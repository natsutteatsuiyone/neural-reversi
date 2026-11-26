import NeuralReversi.PvsParallel.SplitWorkerMir.FieldProgram
import NeuralReversi.PvsParallel.SplitWorkerMir.FieldStores

namespace NeuralReversi.PvsParallel.SplitWorkerMir.FieldStores

open Threading.AtomicWord (Write)
open Threading.SplitWindow (StoreWindow BestStoreWindow MoveStoreWindow)
open Threading.SplitResult (Report)

inductive Value where
  | readAlpha (word : Word)
  | readBest (word : Word)
  | writeAlpha (word : Word)
  | writeBest (word : Word)
  | writeMove (byte : BitVec 8)
  deriving DecidableEq

inductive Physical where
  | readAlpha (word : Word)
  | readBest (word : Word)
  | writeAlpha (write : Write)
  | writeBest (write : Write)
  | writeMove (write : Write)

def Physical.value : Physical → Value
  | .readAlpha a => .readAlpha a
  | .readBest b => .readBest b
  | .writeAlpha w => .writeAlpha (low32 w)
  | .writeBest w => .writeBest (low32 w)
  | .writeMove w => .writeMove (low8 w)

def Access.value (encode : Option (Fin 64) → BitVec 8) (atIndex : Nat → Option (Fin 64)) : Access → Value
  | .readAlpha a => .readAlpha a
  | .readBest b => .readBest b
  | .writeAlpha a => .writeAlpha a
  | .writeBest b => .writeBest b
  | .writeMove i => .writeMove (encode (atIndex i))

def Value.alpha? : Value → Option Word | .writeAlpha a => some a | _ => none
def Value.best? : Value → Option Word | .writeBest a => some a | _ => none
def Value.move? : Value → Option (BitVec 8) | .writeMove m => some m | _ => none

def Physical.alpha? : Physical → Option Write | .writeAlpha a => some a | _ => none
def Physical.best? : Physical → Option Write | .writeBest a => some a | _ => none
def Physical.move? : Physical → Option Write | .writeMove m => some m | _ => none

def writes (ops : List Physical) : Writes :=
  ⟨(ops.filterMap Physical.alpha?).head?,(ops.filterMap Physical.best?).head?,(ops.filterMap Physical.move?).head?⟩

private theorem lists (ops : List Physical) :
      (ops.filterMap Physical.alpha?).map low32 = (ops.map Physical.value).filterMap Value.alpha? ∧
      (ops.filterMap Physical.best?).map low32 = (ops.map Physical.value).filterMap Value.best? ∧
      (ops.filterMap Physical.move?).map low8 = (ops.map Physical.value).filterMap Value.move? := by
  induction ops with
  | nil => exact ⟨rfl,rfl,rfl⟩
  | cons op rest ih =>
    obtain ⟨a,b,m⟩ := ih
    cases op <;> simp only [List.map_cons,List.filterMap_cons,Physical.alpha?,Physical.best?,Physical.move?,
      Physical.value,Value.alpha?,Value.best?,Value.move?,a,b,m,and_self]

private theorem projection (ops : List Physical) :
    (writes ops).alpha.map low32 = ((ops.map Physical.value).filterMap Value.alpha?).head? ∧
    (writes ops).best.map low32 = ((ops.map Physical.value).filterMap Value.best?).head? ∧
    (writes ops).move.map low8 = ((ops.map Physical.value).filterMap Value.move?).head? := by
  obtain ⟨a,b,m⟩ := lists ops
  rw [← a,← b,← m]
  simp [writes]

set_option maxHeartbeats 700000 in
private theorem compiled (i : Input) (s : Locked) (chosen : Word)
    (encode : Option (Fin 64) → BitVec 8) (atIndex : Nat → Option (Fin 64)) :
    let emitted := (accesses i s chosen).map (Access.value encode atIndex)
    let output := commit i s chosen
    emitted.filterMap Value.alpha? = output.alpha.toList ∧
    emitted.filterMap Value.best? = output.best.toList ∧
    emitted.filterMap Value.move? = (if output.moveSet then [encode (atIndex i.index)] else []) := by
  rw [commit_eq]
  cases stop : s.stopped
  · cases root : i.root <;> cases pv : i.pv <;>
      by_cases a : s.alpha.toInt < chosen.toInt <;> by_cases b : s.best.toInt < chosen.toInt <;>
      by_cases z : chosen.toInt < i.beta.toInt <;>
      simp [accesses,Access.value,Value.alpha?,Value.best?,Value.move?,List.filterMap_cons, *]
  · simp [accesses,stop]

structure Raw where
  input : Input
  stopped : Bool
  ops : List Physical

/-- Only loads reached by the compiled branch occur in ops. In particular a stopped
    worker needs no fictitious alpha or best-score read. -/
def Raw.decode (input : Raw) (s : Fields) (pv : Bool) (beta : Int)
    (encode : Option (Fin 64) → BitVec 8) (atIndex : Nat → Option (Fin 64)) : Option Record := do
  let locked : Locked := ⟨input.stopped,s.alpha,s.best⟩
  match h : search input.input with
  | none => none
  | some body =>
    if input.input.pv = pv ∧ input.input.beta.toInt = beta ∧
        input.ops.map Physical.value = (accesses input.input locked body.chosen).map (Access.value encode atIndex) then
      some ⟨input.input,locked,body,h,writes input.ops⟩
    else none

theorem Raw.observed (input : Raw) (ok : input.decode s pv beta encode atIndex = some r) :
    Observed s r pv beta encode atIndex := by
  unfold Raw.decode at ok
  split at ok <;> try contradiction
  rename_i body returned
  dsimp only at ok
  split at ok <;> try contradiction
  rename_i same
  cases ok
  let locked : Locked := ⟨input.stopped,s.alpha,s.best⟩
  have effects := compiled input.input locked body.chosen encode atIndex
  obtain ⟨ea,eb,em⟩ := effects
  obtain ⟨ha,hb,hm⟩ := projection input.ops
  rw [same.2.2,ea] at ha
  rw [same.2.2,eb] at hb
  rw [same.2.2,em] at hm
  refine ⟨same.1,same.2.1,rfl,rfl,?_,?_,?_⟩
  · simpa [Record.update,locked] using ha
  · simpa [Record.update,locked] using hb
  · cases h : (commit input.input locked body.chosen).moveSet <;>
      simpa [Record.update,locked,h] using hm

private theorem cut_iff (i : Input) (s : Locked) (chosen : Word) :
    (commit i s chosen).exit = .cutoff ↔ (commit i s chosen).moveSet = true ∧
      (i.pv = false ∨ i.beta.toInt ≤ chosen.toInt ∨ maximum.toInt ≤ chosen.toInt) := by
  rw [commit_eq]
  cases s.stopped <;> cases i.pv <;> by_cases a : s.alpha.toInt < chosen.toInt <;>
    by_cases b : s.best.toInt < chosen.toInt <;> by_cases lo : chosen.toInt < i.beta.toInt <;>
    by_cases hi : maximum.toInt ≤ chosen.toInt <;> simp [*,Int.not_le.mpr,Int.le_of_not_gt]

/-- Even if a branch skipped an alpha load, its actual move store and the fixed
    search window determine whether it took the cutoff exit. -/
theorem Raw.cut_report (s : Fields) (locked : Locked) (ops : List Physical) (body : Searched)
    (searched : search i = some body) (exit : (commit i locked body.chosen).exit = .cutoff)
    (operands : ops.map Physical.value = (accesses i locked body.chosen).map (Access.value encode atIndex))
    (decoded : (Raw.mk i locked.stopped ops).decode s pv beta encode atIndex = some r) :
    r.report = some (selected i body) ∧
    Threading.SplitWindow.marksCutoff pv beta s.state (selected i body).score = true := by
  unfold Raw.decode at decoded
  split at decoded <;> try contradiction
  rename_i found returned
  change search i = some found at returned
  have eq := Option.some.inj (returned.symm.trans searched)
  subst found
  dsimp only at decoded
  split at decoded <;> try contradiction
  rename_i ok
  cases decoded
  let current : Locked := ⟨locked.stopped,s.alpha,s.best⟩
  have equivalent := congrArg (List.filterMap Value.move?) (operands.symm.trans ok.2.2)
  rw [(compiled i locked body.chosen encode atIndex).2.2,(compiled i current body.chosen encode atIndex).2.2] at equivalent
  obtain ⟨picked,reason⟩ := (cut_iff ..).mp exit
  have picks : (commit i current body.chosen).moveSet = true := by
    cases h : (commit i current body.chosen).moveSet <;> simp_all
  have normal : locked.stopped = false := by
    cases h : locked.stopped <;> simp_all [commit_eq]
  exact ⟨by simp [Record.report,normal],ok.1 ▸ ok.2.1 ▸ cutoff_marks i current body.chosen ((cut_iff ..).mpr ⟨picks,reason⟩)⟩

private theorem head_of_singleton {xs : List α} {f : α → β} {v : Option β} (eq : xs.map f = v.toList) :
    xs.head?.toList = xs := by
  have len : xs.length ≤ 1 := by simpa [← eq] using (Option.length_toList_le (o := v))
  cases xs with
  | nil => rfl
  | cons x xs => cases xs <;> simp_all

theorem Raw.stores (input : Raw) (ok : input.decode s pv beta encode atIndex = some r) :
    r.writes.alpha.toList = input.ops.filterMap Physical.alpha? ∧
    r.writes.best.toList = input.ops.filterMap Physical.best? ∧
    r.writes.move.toList = input.ops.filterMap Physical.move? := by
  unfold Raw.decode at ok
  split at ok <;> try contradiction
  rename_i body _
  dsimp only at ok
  split at ok <;> try contradiction
  rename_i same
  cases ok
  obtain ⟨a,b,m⟩ := lists input.ops
  let lock : Locked := ⟨input.stopped,s.alpha,s.best⟩
  obtain ⟨ca,cb,cm⟩ := compiled input.input lock body.chosen encode atIndex
  rw [same.2.2,ca] at a
  rw [same.2.2,cb] at b
  rw [same.2.2,cm] at m
  refine ⟨head_of_singleton a,head_of_singleton b,?_⟩
  cases move : (commit input.input lock body.chosen).moveSet
  · exact head_of_singleton (f := low8) (v := none) (by simpa [move] using m)
  · exact head_of_singleton (f := low8) (v := some (encode (atIndex input.input.index))) (by simpa [move] using m)

def replay (pv : Bool) (beta : Int) (encode : Option (Fin 64) → BitVec 8) (atIndex : Nat → Option (Fin 64))
    (l : Log) : List Raw → Option Log
  | [] => some l
  | input :: rest => do
    let record ← input.decode l.fields pv beta encode atIndex
    replay pv beta encode atIndex (l.append record atIndex) rest

theorem replay_preserves (calls : List Raw) (start : Invariant pv beta initial encode atIndex l)
    (ok : replay pv beta encode atIndex l calls = some final) : Invariant pv beta initial encode atIndex final := by
  induction calls generalizing l with
  | nil => cases ok; exact start
  | cons input rest ih =>
    obtain ⟨r,read,more⟩ := Option.bind_eq_some_iff.mp ok
    exact ih (append start (input.observed read)) more

theorem replay_window (calls : List Raw) (ok : replay pv beta encode atIndex initial calls = some final)
    (member : c ∈ calls) : c.input.pv = pv ∧ c.input.beta.toInt = beta := by
  induction calls generalizing initial with
  | nil => contradiction
  | cons head rest ih =>
    obtain ⟨rec, read, tail⟩ := Option.bind_eq_some_iff.mp ok
    rcases List.mem_cons.mp member with rfl | more
    · unfold Raw.decode at read
      split at read <;> try contradiction
      dsimp only at read
      split at read <;> try contradiction
      rename_i selected
      exact ⟨selected.1, selected.2.1⟩
    · exact ih tail more

theorem modification_orders (calls : List Raw) (ok : replay pv beta encode atIndex l calls = some final) :
    final.alpha = l.alpha ++ calls.flatMap (fun c => c.ops.filterMap Physical.alpha?) ∧
    final.best = l.best ++ calls.flatMap (fun c => c.ops.filterMap Physical.best?) ∧
    final.moves = l.moves ++ calls.flatMap (fun c => c.ops.filterMap Physical.move?) := by
  induction calls generalizing l with
  | nil => cases ok; simp
  | cons input rest ih =>
    obtain ⟨r,observed,more⟩ := Option.bind_eq_some_iff.mp ok
    obtain ⟨a,b,m⟩ := input.stores observed
    simpa [Log.append,a,b,m] using ih more

theorem windows (calls : List Raw) (a b m : Write) (alpha : low32 a = initial.alpha)
    (best : low32 b = initial.best) (move : low8 m = encode initial.picked)
    (ok : replay pv beta encode atIndex (initialLog initial a b m) calls = some final)
    (aPre aPost bPre bPost mPre mPost : List Write)
    (first lastA : Fin (aPre ++ final.alpha ++ aPost).length)
    (lastB : Fin (bPre ++ final.best ++ bPost).length) (lastM : Fin (mPre ++ final.moves ++ mPost).length)
    (beginA : first.val = aPre.length) (endA : lastA.val+1 = aPre.length+final.alpha.length)
    (endB : lastB.val+1 = bPre.length+final.best.length) (endM : lastM.val+1 = mPre.length+final.moves.length) :
    StoreWindow _ first lastA pv beta initial.state (final.reports.map Report.score) ∧
    BestStoreWindow _ lastB initial.best.toInt (final.reports.map Report.score) ∧
    MoveStoreWindow encode atIndex pv beta initial.state initial.picked final.reports final.fields.picked _ lastM := by
  have verified := replay_preserves calls (seed initial a b m alpha best move) ok
  exact ⟨⟨_,_,_,rfl,beginA,endA,verified.alphaWrites⟩,
    ⟨_,_,_,rfl,endB,verified.bestWrites⟩,⟨_,_,_,rfl,endM,verified.moveWrites⟩⟩

end NeuralReversi.PvsParallel.SplitWorkerMir.FieldStores

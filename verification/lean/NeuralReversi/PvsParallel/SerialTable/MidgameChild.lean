import NeuralReversi.PvsParallel.SerialTable.MidgameMemory
import NeuralReversi.MidgameScoreRange.Search
import NeuralReversi.PvsParallel.SerialTable.MidgamePoll

namespace NeuralReversi.PvsParallel.SerialTable.MidgameChild

open AspirationMir.SerialTable.InitialEstimate (static)
open AspirationMir.SerialTable.Repeated (ReadyMemory)
open AspirationMir.SerialTable.PvInterior (InsideRange)
open EndgameSearch.CacheHistory (Pool)
open EndgameSearch.SerialLeafCache (Call)
open PvsBoardTree (Position GeneratedSquare)
open TranspositionTable (Bound Selectivity)
open SerialCalls (Program Request MoveRequest)
open SerialCalls.ScoreWords (Finite)
open PvsMoveLoop (scoreMax)
open MidgameMemory (penalize save)

abbrev Child (l : Layout) := Request → Memory l → Option (Int × Memory l)

private structure Result (l : Layout) (p : Pool) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (q : Request) where
  exec : Memory l → Int × Memory l
  sound : ∀ m, ReadyMemory p mc fc m → Aspiration.Window q.alpha q.beta →
    ReadyMemory p mc fc (exec m).2 ∧ Finite (exec m).1 ∧ WeakTrial.Range q (exec m).1

private theorem bounded {q : Request} {v : Int} (range : -scoreMax ≤ v ∧ v ≤ scoreMax) :
    Finite v ∧ WeakTrial.Range q v := by
  refine ⟨?_,fun _ _ _ => range⟩
  unfold Finite
  unfold scoreMax at range
  omega

private def scalar (q : Request) (v : Int) (range : -scoreMax ≤ v ∧ v ≤ scoreMax) : Result l p mc fc q :=
  ⟨fun m => (v,m),fun _ ready _ => ⟨ready,bounded range⟩⟩

def passRequest (q : Request) : Request :=
  {q with board := PvsBoardTree.pass q.board,alpha := -q.beta,beta := -q.alpha}

private def passed (cut : Bool) (child : Result l p mc fc {(passRequest q) with cut := cut}) : Result l p mc fc q where
  exec m := let (v,n) := child.exec m; (-v,n)
  sound m ready window := by
    obtain ⟨post,finite,range⟩ := child.sound m ready (MidgameScoreRange.reflected window)
    exact ⟨post,MidgameScoreRange.finite_neg finite,fun pv lo hi => by
      change q.alpha < -(child.exec m).1 at lo
      change -(child.exec m).1 < q.beta at hi
      change -scoreMax ≤ -(child.exec m).1 ∧ -(child.exec m).1 ≤ scoreMax
      obtain ⟨low,high⟩ := range pv (by change -q.beta < _; omega) (by change _ < -q.alpha; omega)
      omega⟩

/-- Evaluate calls in chronological order, including calls whose value only decides whether to retry. -/
private def thenCall (sub : Result l p mc fc r) (window : Aspiration.Window r.alpha r.beta)
    (next : Int → Result l p mc fc q) : Result l p mc fc q where
  exec m := let (v,n) := sub.exec m; (next v).exec n
  sound m ready w := (next (sub.exec m).1).sound _ (sub.sound m ready window).1 w

private def owner (q : Request) (stop : GeneratedSquare mc q.board.val → Bool)
    (slot : Fin 2) (level : Selectivity) (moves : List (GeneratedSquare mc q.board.val))
    (unique : moves.Nodup) (nonempty : moves ≠ [])
    (children : (r : MoveRequest (GeneratedSquare mc q.board.val)) →
      Result l p mc fc (SerialPvCalls.moveRequest q.board mc fc q.depth q.cut r)) : Result l p mc fc q where
  exec m :=
    let code := SerialOwnerCalls.run Subtype.val q.beta moves (RootReplay.start q.alpha)
    match MidgamePoll.run (fun r => (children r).exec) (fun r w => MidgamePoll.afterMove r.pv r.beta w && stop r.move) m code with
    | some (none,n) => (0,n)
    | some (some ans,n) =>
      let v := ans.2.state.best
      let after := save n (l.index q.board slot) q.board v
            (TranspositionTable.classify q.pv (Writes.word v) (Writes.word q.alpha) (Writes.word q.beta))
            q.depth (Writes.pvMove q.beta ans.1) level true
      (v,after)
    | none => (0,m)
  sound m initial window := by
    let code := SerialOwnerCalls.run Subtype.val q.beta moves (RootReplay.start q.alpha)
    let word := fun (r : MoveRequest (GeneratedSquare mc q.board.val)) (v : Int) =>
      Finite v ∧ (r.pv = true → InsideRange r.alpha r.beta v)
    have live := SerialOwnerCalls.run_ready (square := @Subtype.val _ _) (s := RootReplay.start q.alpha) window moves
    obtain ⟨out,after,ran,post,done_⟩ := MidgamePoll.returns (code := code)
      (word := word) (safe := ReadyMemory p mc fc) live (fun r => (children r).exec) (fun r w => MidgamePoll.afterMove r.pv r.beta w && stop r.move)
      (fun r ok prev good => (children r).sound prev good ok.1) m initial
    dsimp only
    rw [ran]
    cases out with
    | none => exact ⟨post,bounded (v := 0) ⟨by decide,by decide⟩⟩
    | some out =>
      obtain ⟨calls,issued,words⟩ := done_ out rfl
      have finite := SerialOwnerCalls.run_finite nonempty issued (fun t ht => (words t ht).1)
      have range : WeakTrial.Range q out.2.state.best := fun _ =>
        AspirationMir.SerialTable.PvInterior.owner unique issued window (fun t ht => (words t ht).2)
      exact ⟨MidgameMemory.save_ready _ post _ _ _ _ _ _ _ _ (MidgameMemory.classified q _ window finite range),
          finite,range⟩

private def noPv (q : Request) (isScout : q.pv = false) (read : Memory l → Int × Memory l)
    (sound : ∀ m, ReadyMemory p mc fc m → Aspiration.Window q.alpha q.beta →
      ReadyMemory p mc fc (read m).2 ∧ Finite (read m).1) : Result l p mc fc q :=
  ⟨read, fun m ok win => ⟨(sound m ok win).1,(sound m ok win).2,fun h => Bool.noConfusion (isScout.symm.trans h)⟩⟩

private def promoted (q : Request) (isScout : q.pv = false) (call : Call)
    (lo : call.alpha.raw.toInt = q.alpha) (hi : call.beta.raw.toInt = q.beta) : Result l p mc fc q :=
  noPv q isScout (fun m => match call.execute p mc fc m.cache with
    | some ((v,_),next) => (v,{m with cache := next})
    | none => (0,m)) (by
      intro m ready win
      obtain ⟨v,next,ran⟩ := EndgameSearch.SerialLeafCache.leaf_step call p mc fc m.cache ready.1.1.2
      rw [ran.executed]
      refine ⟨⟨⟨⟨ready.1.1.1,ran.nextReachable⟩,ready.1.2⟩,ready.2⟩,?_⟩
      apply SerialCalls.ScoreWords.finite_of_replay _ (fun _ => call.null) ran.bound
      rwa [lo,hi])

inductive Store where
  | classified | lower

def Store.tag (where_ : Store) (q : Request) (v : Int) : Bound :=
  match where_ with
  | .classified => TranspositionTable.classify q.pv (Writes.word v) (Writes.word q.alpha) (Writes.word q.beta)
  | .lower => .lower

private def stores (q : Request) (offset : Fin 2) (how : Store) (level : Selectivity) (ttPv : Bool)
    (hint : Option (GeneratedSquare mc q.board.val)) (sub : Result l p mc fc q) : Result l p mc fc q where
  exec m := let (v,n) := sub.exec m; (v,save n (l.index q.board offset) q.board v (how.tag q v) q.depth hint level ttPv)
  sound m good win := by
    obtain ⟨post,finite,range⟩ := sub.sound m good win
    exact ⟨MidgameMemory.save_ready _ post _ _ _ _ _ _ _ _ (by
      cases how with
      | classified => exact MidgameMemory.classified q _ win finite range
      | lower => intro h; contradiction),finite,range⟩

/-- Observable control choices of a serial midgame child. Search depths, move ordering,
    reductions and neural operands may vary between nodes. The program obtains each child's
    returned value from its execution and threads the same table and worker caches. -/
inductive Plan (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) : Request → Type where
  | static {q : Request} (raw : Int) : Plan mc fc q
  | terminal {q : Request} : Plan mc fc q
  | wipeout {q : Request} : Plan mc fc q
  | stopped {q : Request} : Plan mc fc q
  | pass {q : Request} (child : Plan mc fc (passRequest q)) : Plan mc fc q
  | pvsPass {q : Request} (child : Plan mc fc {(passRequest q) with cut := !q.pv && !q.cut}) : Plan mc fc q
  | before {q : Request} (r : Request) (window : Aspiration.Window r.alpha r.beta)
      (sub : Plan mc fc r) (next : Int → Plan mc fc q) : Plan mc fc q
  | principalStoreUntil {q : Request} (offset : Fin 2) (level : Selectivity) (moves : List (GeneratedSquare mc q.board.val))
      (unique : moves.Nodup) (nonempty : moves ≠ []) (stop : GeneratedSquare mc q.board.val → Bool) (each : (r : MoveRequest (GeneratedSquare mc q.board.val)) →
        Plan mc fc (SerialPvCalls.moveRequest q.board mc fc q.depth q.cut r)) : Plan mc fc q
  | observed {q : Request} (board : Position) (next : Fin 2 → Option TranspositionTable.EntryData → Plan mc fc q) : Plan mc fc q
  | lookupAt {q : Request} (board : Position) (next : Option TranspositionTable.EntryData → Plan mc fc q) : Plan mc fc q
  | nonPvWord {q : Request} (isScout : q.pv = false) (v : Int) : Plan mc fc q
  | reindex {q : Request} (depth : TranspositionTable.Depth) (sub : Plan mc fc {q with depth := depth}) : Plan mc fc q
  | writeAt {q : Request} (slot : Fin 2) (how : Store) (level : Selectivity) (ttPv : Bool)
      (hint : Option (GeneratedSquare mc q.board.val)) (sub : Plan mc fc q) : Plan mc fc q
  | promotion {q : Request} (isScout : q.pv = false) (call : Call) (same : call.board = q.board)
      (lo : call.alpha.raw.toInt = q.alpha) (hi : call.beta.raw.toInt = q.beta) : Plan mc fc q
  | penalty {q : Request} (body : Plan mc fc q) : Plan mc fc q

private def compile (l : Layout) (p : Pool) {q : Request} : Plan mc fc q → Result l p mc fc q
  | .static raw => scalar q _ (static (fun _ => raw) q.board).property
  | .terminal => scalar q _ (PvsBoardTree.terminalScore q.board).property
  | .wipeout => scalar q scoreMax ⟨by decide,Int.le_refl _⟩
  | .stopped => scalar q 0 ⟨by decide,by decide⟩
  | .pass child => passed q.cut (compile l p child)
  | .pvsPass child => passed (!q.pv && !q.cut) (compile l p child)
  | .before _ win sub next => thenCall (compile l p sub) win fun v => compile l p (next v)
  | .principalStoreUntil slot level ms un ne stop sub => owner q stop slot level ms un ne fun r => compile l p (sub r)
  | .observed b next =>
    let f := fun (m : Memory l) => match m.tt.image.probe b with
      | .hit slot e => compile l p (next slot (some e))
      | .miss slot => compile l p (next slot none)
    ⟨fun m => (f m).exec m,fun m ok w => (f m).sound m ok w⟩
  | .lookupAt b next =>
    ⟨fun m => (compile l p (next (m.tt.image.lookup b))).exec m,
      fun m ok w => (compile l p (next (m.tt.image.lookup b))).sound m ok w⟩
  | .nonPvWord isScout v =>
    if finite : -32767 < v ∧ v < 32767 then noPv q isScout (fun m => (v,m)) fun _ ok _ => ⟨ok,finite⟩
    else scalar q 0 ⟨by decide,by decide⟩
  | .reindex _ sub =>
    let s := compile l p sub
    ⟨s.exec,fun m ok w => s.sound m ok w⟩
  | .writeAt slot how level ttPv hint sub => stores q slot how level ttPv hint (compile l p sub)
  | .promotion isScout call _ lo hi => promoted q isScout call lo hi
  | .penalty body =>
    let sub := compile l p body
    ⟨fun m => sub.exec (penalize m q.board),fun m ok win => sub.sound _ (MidgameMemory.penalty_ready m q.board ok) win⟩

def eval (p : Pool) {q : Request} (plan : Plan mc fc q) (m : Memory l) : Int × Memory l :=
  (compile l p plan).exec m

theorem eval_nonPvWord (p : Pool) (q : Request) (m : Memory l) (scout : q.pv = false) (v : Int)
    (finite : Finite v) : eval (mc := mc) (fc := fc) p (.nonPvWord scout v) m = (v,m) := by
  unfold Finite at finite
  simp [eval,compile,finite,noPv]

theorem eval_ready (p : Pool) {q : Request} (plan : Plan mc fc q) (m : Memory l)
    (win : Aspiration.Window q.alpha q.beta) (ready : ReadyMemory p mc fc m) :
    ReadyMemory p mc fc (eval p plan m).2 ∧ Finite (eval p plan m).1 ∧ WeakTrial.Range q (eval p plan m).1 :=
  (compile l p plan).sound m ready win

def run (p : Pool) (plan : (q : Request) → Plan mc fc q) : Child l :=
  fun q m => some (eval p (plan q) m)

theorem returns (p : Pool) (plan : (q : Request) → Plan mc fc q) (q : Request) (m : Memory l)
    (win : Aspiration.Window q.alpha q.beta) (ready : ReadyMemory p mc fc m) :
    ∃ v after, run p plan q m = some (v,after) ∧ ReadyMemory p mc fc after ∧ Finite v ∧ WeakTrial.Range q v :=
  ⟨_,_,rfl,(compile l p (plan q)).sound m ready win⟩

end NeuralReversi.PvsParallel.SerialTable.MidgameChild

module
public import NeuralReversi.PvsParallel.NativeFrames.CacheCalls
public import NeuralReversi.EndgameSearch.CacheHistory.Payloads.Instructions

@[expose] public section

namespace NeuralReversi.PvsParallel.NativeFrames.CacheCalls

open EndgameSearch (SafeThreshold)
open EndgameSearch.CacheOwner (Node Edge Flow transfers)
open EndgameSearch.CacheHistory (Pool Local)
open EndgameSearch.CacheHistory.Payloads (Instruction)
open EndgameTreeCache (Table Store soundHistory)

/-- Arguments supplied at a native cache call. A store's signed word is
    recovered from its source score when the enclosing search history is sound. -/
inductive Access where
  | slot (t : Table) (key : BitVec 64)
  | probe (t : Table) (board : Board.Board) (alpha : Int) (safe : SafeThreshold alpha)
  | store (t : Table) (board : Board.Board) (alpha score : Int) (safe : SafeThreshold alpha)

def Access.node : Access → Node
  | .slot .deep _ => .deepSlot | .slot .shallow _ => .shallowSlot
  | .probe .deep .. => .deepProbe | .probe .shallow .. => .shallowProbe
  | .store .deep .. => .deepStore | .store .shallow .. => .shallowStore

def Access.store? : Access → Option Store
  | .store t b a score _ => some ⟨t,b,a,score⟩
  | _ => none

def Access.after (h : List Store) : Access → List Store
  | .store t b a v _ => h ++ [⟨t,b,a,v⟩]
  | _ => h

def Access.instruction : Access → Instruction
  | .slot t k => .slot t k
  | .probe t b a _ => .probe t b (BitVec.ofInt 32 a)
  | .store t b a score _ => .store t b (BitVec.ofInt 32 a) (BitVec.ofInt 32 score)

/-- Input and output are the write histories immediately before and after the
    block. Only a store advances that index, at its actual site in the call tree. -/
inductive Program : Node → List Store → List Store → Type where
  | done : Program node h h
  | call (site : Nat) (child : Node) (flow : Flow) (edge : Edge.mk node site child flow ∈ transfers)
      (routine : EndgameSearch.CacheHistory.Payloads.target child = none)
      (body : Program child h middle) (next : Program node middle last) : Program node h last
  | access (site : Nat) (op : Access) (flow : Flow) (edge : Edge.mk node site op.node flow ∈ transfers)
      (next : Program node (op.after h) last) : Program node h last

def Program.append (code : Program node h middle) (rest : Program node middle last) : Program node h last :=
  match code with
  | .done => rest
  | .call site child flow edge routine body next => .call site child flow edge routine body (next.append rest)
  | .access site op flow edge next => .access site op flow edge (next.append rest)

def Program.forest : Program node h last → Forest node
  | .done => .nil
  | .call site child flow edge _ body next => .call site child flow edge body.forest next.forest
  | .access site op flow edge next => .call site op.node flow edge .nil next.forest

def Program.accesses : Program node h last → List Access
  | .done => []
  | .call _ _ _ _ _ body next => body.accesses ++ next.accesses
  | .access _ op _ _ next => op :: next.accesses

theorem Program.stores (code : Program node h last) : h ++ code.accesses.filterMap Access.store? = last := by
  induction code with
  | done => simp [accesses]
  | call _ _ _ _ _ _ _ first rest => simpa [accesses,←List.append_assoc,first] using rest
  | access _ op _ _ _ ih =>
    cases op <;> simpa [accesses,List.filterMap_cons,Access.store?,Access.after,List.append_assoc] using ih

private theorem thresholdWord {a : Int} (safe : SafeThreshold a) : (BitVec.ofInt 32 a).toInt = a :=
  BitVec.toInt_ofInt_eq_self (by decide) (by have := safe.1; omega) (by have := safe.2; omega)

private theorem filter_same (xs : List α) (f g : α → Option β) (same : ∀ x ∈ xs, f x = g x) :
    xs.filterMap f = xs.filterMap g := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.filterMap_cons,same x List.mem_cons_self,ih (fun y hy => same y (List.mem_cons_of_mem _ hy))]

theorem Program.words_of_mem (code : Program node h last) (sound : soundHistory mc fc last)
    (xs : List Access) (inside : ∀ x ∈ xs, x ∈ code.accesses) :
    (xs.map Access.instruction).filterMap Instruction.write? = xs.filterMap Access.store? := by
  simp only [List.filterMap_map]
  apply filter_same
  intro op member
  cases op with
  | slot | probe => rfl
  | store t b a v safe =>
    have mem : Store.mk t ⟨b,a,v⟩ ∈ last := by
      rw [← code.stores]
      exact List.mem_append_right _ (List.mem_filterMap.mpr ⟨_,inside _ member,rfl⟩)
    have bound := (sound _ mem).1
    change -128 ≤ v ∧ v ≤ 127 at bound
    have value : (BitVec.ofInt 32 v).toInt = v := BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
    simp [Access.instruction,Access.store?,Instruction.write?,thresholdWord safe,value]

theorem Program.words (code : Program node h last) (sound : soundHistory mc fc last) :
    (code.accesses.map Access.instruction).filterMap Instruction.write? = code.accesses.filterMap Access.store? :=
  code.words_of_mem sound _ (fun _ h => h)

/-- The program yields passing full and partial byte-level audits. The cutoff
    can fall between a probe and a store or inside any child. -/
theorem Program.audit_prefix (code : Program node s.history last) (sound : soundHistory mc fc last) (p : Pool)
    (sizes : ∀ t, (s.cells t).size = p.sizes t) (n : Nat) :
    let ops := (code.accesses.map Access.instruction).take n
    EndgameSearch.CacheHistory.Payloads.check p false s last (EndgameSearch.CacheHistory.Payloads.observeRun p s sizes ops) =
      some ⟨EndgameSearch.CacheHistory.Payloads.observeRun p s sizes ops,ops.foldl (fun st op => op.step p st) s⟩ := by
  apply EndgameSearch.CacheHistory.Payloads.check_run
  let all := code.accesses.map Access.instruction
  have whole : s.history ++ all.filterMap Instruction.write? = last := by
    exact (congrArg (s.history ++ ·) (code.words sound)).trans code.stores
  have parts := congrArg (List.filterMap Instruction.write?) (List.take_append_drop n all)
  rw [List.filterMap_append] at parts
  let prefixHistory := s.history ++ (all.take n).filterMap Instruction.write?
  have growth : prefixHistory ++ (all.drop n).filterMap Instruction.write? = last := by
    exact (List.append_assoc ..).trans ((congrArg (s.history ++ ·) parts).trans whole)
  change prefixHistory = last.take prefixHistory.length
  calc prefixHistory = (prefixHistory ++ (all.drop n).filterMap Instruction.write?).take prefixHistory.length := by simp
       _ = _ := congrArg (fun h => h.take prefixHistory.length) growth

theorem Program.audit (code : Program node s.history last) (sound : soundHistory mc fc last) (p : Pool)
    (sizes : ∀ t, (s.cells t).size = p.sizes t) :
    let ops := code.accesses.map Access.instruction
    EndgameSearch.CacheHistory.Payloads.check p true s last (EndgameSearch.CacheHistory.Payloads.observeRun p s sizes ops) =
      some ⟨EndgameSearch.CacheHistory.Payloads.observeRun p s sizes ops,ops.foldl (fun st op => op.step p st) s⟩ := by
  apply EndgameSearch.CacheHistory.Payloads.check_run
  simpa only [ite_true,code.words sound] using code.stores

end NeuralReversi.PvsParallel.NativeFrames.CacheCalls

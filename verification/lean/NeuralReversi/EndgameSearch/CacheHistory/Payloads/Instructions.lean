module
public import NeuralReversi.EndgameSearch.CacheHistory.Payloads

@[expose] public section

namespace NeuralReversi.EndgameSearch.CacheHistory.Payloads

open EndgameTreeCache (Store Table)
open EndgameCache (Entry)
open Board (Board)

/-- The operands at the three MIR calls. Probe and store use the slot computed
    for this board on this worker's table. Its length stays fixed across calls. -/
inductive Instruction where
  | slot (table : Table) (key : BitVec 64)
  | probe (table : Table) (board : Board) (alpha : BitVec 32)
  | store (table : Table) (board : Board) (alpha score : BitVec 32)

def probeWord (e : Entry) (b : Board) (a : BitVec 32) : Option (BitVec 32) :=
  if e.board.player ≠ b.player ∨ e.board.opponent ≠ b.opponent then none
  else if e.lower.toInt > a.toInt then some (e.lower.toBitVec.signExtend 32)
  else if e.upper.toInt ≤ a.toInt then some (e.upper.toBitVec.signExtend 32)
  else none

private theorem probeWord_eq (e : Entry) (b : Board) (a : BitVec 32) :
    (probeWord e b a).map BitVec.toInt = CacheMir.probeCell e b a.toInt := by
  by_cases hp : e.board.player = b.player <;> by_cases ho : e.board.opponent = b.opponent <;>
    simp [probeWord,CacheMir.probeCell,hp,ho,CacheMir.scoreForAlpha]
  split <;> simp [BitVec.toInt_signExtend_of_le (by decide : 8 ≤ 32)]

theorem inside (p : Pool) (s : Local) (shape : ∀ t, (s.cells t).size = p.sizes t) (t : Table) (b : Board) :
    p.address t b < (s.cells t).size := by
  rw [shape]
  have fit := CacheMir.budget_fits p.size p.width p.word t
  exact (CacheMir.index_eq _ _ _ fit.1 fit.2.1 fit.2.2).2

def Instruction.observe (p : Pool) (s : Local) (shape : ∀ t, (s.cells t).size = p.sizes t) : Instruction → Op
  | .slot t key => ⟨t,.slot key (CacheMir.slot p.width (s.cells t).size key)⟩
  | .probe t b a =>
    let i := p.address t b
    let e := (s.cells t)[i]'(inside p s shape t b)
    ⟨t,.probe i b a (words e) (probeWord e b a)⟩
  | .store t b a v =>
    let i := p.address t b
    let e := (s.cells t)[i]'(inside p s shape t b)
    ⟨t,.store i b a v (words e) (words (CacheMir.storeCell e b a.toInt v.toInt))⟩

/-- The machine word consumed by a cache branch is the result of probing the
    entry at the same address in the logical write history, including misses. -/
theorem observe_probe_words (p : Pool) (s : Local) (shape : ∀ t, (s.cells t).size = p.sizes t)
    (aligned : CacheState.represents p.address s.history s.cells) (t : Table) (b : Board) (a : BitVec 32) :
    let logical := EndgameTreeCache.entryFor p.address s.history t b
    ∃ raw, (Instruction.probe t b a).observe p s shape = ⟨t,.probe (p.address t b) b a (words logical) raw⟩ ∧
      raw.map BitVec.toInt = EndgameCache.probe logical b a.toInt := by
  let e := (s.cells t)[p.address t b]'(inside p s shape t b)
  have old := aligned t b
  rw [Array.getElem?_eq_getElem (inside p s shape t b)] at old
  have cell : e = EndgameTreeCache.entryFor p.address s.history t b := Option.some.inj old
  refine ⟨probeWord e b a,?_,?_⟩
  · change Op.mk t (.probe _ b a (words e) (probeWord e b a)) = _
    rw [cell]
  · rw [probeWord_eq,CacheMir.probeCell_eq,cell]

def Instruction.write? : Instruction → Option Store
  | .store t b a v => some ⟨t,b,a.toInt,v.toInt⟩
  | _ => none

def Instruction.step (p : Pool) (s : Local) (op : Instruction) : Local :=
  op.write?.elim s (applyWrite p s)

theorem Instruction.execute_observe (op : Instruction) (p : Pool) (s : Local)
    (shape : ∀ t, (s.cells t).size = p.sizes t) :
    execute p s (op.observe p s shape) = some (op.step p s) := by
  have index (t b) : CacheMir.slot p.width (s.cells t).size (p.key b) = p.address t b := by rw [shape]; rfl
  cases op with
  | slot t k => simp [observe,execute,step,Instruction.write?]
  | probe t b a =>
    simp [observe,execute,CacheMir.probe,Array.getElem?_eq_getElem (inside p s shape t b),probeWord_eq,index,step,Instruction.write?]
  | store t b a v =>
    have update := CacheMir.write_eq (hash:=p.key) (Store.mk t ⟨b,a.toInt,v.toInt⟩) shape
      (CacheMir.budget_fits p.size p.width p.word)
    change CacheMir.writeCells p.width p.key s.cells _ = some (CacheState.write p.address s.cells _) at update
    simp [observe,execute,Array.getElem?_eq_getElem (inside p s shape t b),index,update,step,Instruction.write?,applyWrite]

theorem Instruction.shape (op : Instruction) (p : Pool) (s : Local) (sizes : ∀ t, (s.cells t).size = p.sizes t) :
    ∀ t, ((op.step p s).cells t).size = p.sizes t := by
  intro t
  unfold step
  cases h : op.write? with
  | none => exact sizes t
  | some w => by_cases same : t = w.table <;> simp [applyWrite,CacheState.write,same,sizes]

/-- Capture each operand from the array left by the preceding instruction. -/
def observeRun (p : Pool) (s : Local) (shape : ∀ t, (s.cells t).size = p.sizes t) : List Instruction → List Op
  | [] => []
  | op :: rest => op.observe p s shape :: observeRun p (op.step p s) (op.shape p s shape) rest

theorem execute_run (p : Pool) (ops : List Instruction) (s : Local) (shape : ∀ t, (s.cells t).size = p.sizes t) :
    (observeRun p s shape ops).foldlM (execute p) s = some (ops.foldl (fun st op => op.step p st) s) := by
  induction ops generalizing s with
  | nil => rfl
  | cons op rest ih => simpa [observeRun,List.foldlM_cons,op.execute_observe] using ih (op.step p s) (op.shape p s shape)

theorem observe_writes (p : Pool) (ops : List Instruction) (s : Local) (shape : ∀ t, (s.cells t).size = p.sizes t) :
    writes (observeRun p s shape ops) = ops.filterMap Instruction.write? := by
  induction ops generalizing s with
  | nil => rfl
  | cons op rest ih =>
    have tail := ih (op.step p s) (op.shape p s shape)
    cases op <;> simp only [writes,observeRun,List.filterMap_cons,Instruction.observe,Op.write?,Instruction.write?] at tail ⊢ <;>
      simp [tail]

theorem check_run (p : Pool) (ops : List Instruction) (s : Local) (shape : ∀ t, (s.cells t).size = p.sizes t)
    (complete : Bool) (expected : List Store)
    (matched : let h := s.history ++ ops.filterMap Instruction.write?
               h = if complete then expected else expected.take h.length) :
    check p complete s expected (observeRun p s shape ops) =
      some ⟨observeRun p s shape ops,ops.foldl (fun st op => op.step p st) s⟩ := by
  have ran := execute_run p ops s shape
  have history := (trace_writes p _ shape ran).1.trans (congrArg (s.history ++ ·) (observe_writes p ops s shape))
  unfold check
  rw [ran]
  let last := ops.foldl (fun st op => op.step p st) s
  change (if last.history = (if complete then expected else expected.take last.history.length) then some _ else none) = _
  rw [ite_eq_left (show last.history = (if complete then expected else expected.take last.history.length) by
    change last.history = _ at history
    rw [history]; exact matched)]

theorem check_direct (p : Pool) (i : Input) (s : Local) (complete : Bool)
    (direct : (∃ f, i.arm = .fixed f) ∨ (∃ e, i.arm = .table e)) :
    check p complete s (i.run mc fc p.address s.history).history [] = some ⟨[],s⟩ := by
  rcases direct with ⟨f,arm⟩ | ⟨e,arm⟩ <;> cases complete <;> simp [check,Input.run,arm,Input.eval]

end NeuralReversi.EndgameSearch.CacheHistory.Payloads

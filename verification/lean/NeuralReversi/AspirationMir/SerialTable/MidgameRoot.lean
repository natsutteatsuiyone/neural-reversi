import NeuralReversi.AspirationMir.SerialTable.MidgameRootRows
import NeuralReversi.AspirationMir.SerialTable.MidgameRows
import NeuralReversi.AspirationMir.SerialTable.Repeated

namespace NeuralReversi.AspirationMir.SerialTable.MidgameRoot

open PvsParallel
open PvsParallel.SerialTable (Memory Layout)
open PvsParallel.SerialTable.WeakTrial (Range)
open PvsParallel.SerialTable.Writes (Proposal)
open PvsBoardTree (Position)
open PvsMoveLoop (scoreMax)
open SerialCalls (Program Request)
open SerialCalls.ScoreWords (Finite)
open EndgameSearch.CacheHistory (Pool)
open Repeated (ReadyMemory)
open RootMoveStorage (Row)

abbrev Child (l : Layout) := Request → Memory l → Option (Int × Memory l)

def ChildReturns (p : Pool) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (child : Child l) : Prop :=
  ∀ q m, Aspiration.Window q.alpha q.beta → (q.pv = false → q.beta = q.alpha + 1) → ReadyMemory p mc fc m →
    ∃ v next, child q m = some (v,next) ∧ ReadyMemory p mc fc next ∧ Finite v ∧ Range q v

/-- Invoke each child on the table and worker caches returned by its predecessor. -/
def interpret (child : Child l) : Program Request α → Memory l → Option (α × Memory l)
  | .fail, _ => none
  | .pure v, m => some (v,m)
  | .call q cont, m => do
    let (v,next) ← child q m
    interpret child (cont v) next

private theorem interpret_returns {valid : Memory l → Prop} {allowed : Request → Prop} {code : Program Request α}
    (ready : Program.Ready allowed code) (child : Child l) (children : ∀ q, allowed q → ∀ m, valid m →
      ∃ v next, child q m = some (v,next) ∧ valid next ∧ Finite v ∧ Range q v) (m : Memory l) (ok : valid m) :
    ∃ out next, interpret child code m = some (out,next) ∧ valid next ∧
      ∃ cs, Program.Runs code cs out ∧ ∀ t ∈ cs, Finite t.2 ∧ Range t.1 t.2 := by
  induction ready generalizing m with
  | pure => exact ⟨_,m,rfl,ok,[],.pure,by simp⟩
  | @call q _ issued _ ih =>
    obtain ⟨v,after,ran,safe,fin,range⟩ := children q issued m ok
    obtain ⟨out,next,more,g,cs,run,words⟩ := ih v after safe
    refine ⟨out,next,by simpa [interpret,ran] using more,g,(q,v)::cs,.call run,?_⟩
    intro t present
    rcases List.mem_cons.mp present with rfl | later
    · exact ⟨fin,range⟩
    · exact words t later

def save (m : Memory l) (slot : l.Index) (lvl : TranspositionTable.Selectivity) : Option Proposal → Memory l
  | none => m
  | some w =>
    ⟨m.tt.store slot w.board (PvsParallel.SerialTable.Writes.word w.score) w.bound w.depth w.move lvl false w.ttPv,
      m.tt.store_supported m.supported .., m.cache⟩

theorem save_ready (after : Memory l) (safe : ReadyMemory p mc fc after) (slot : l.Index)
    (lvl : TranspositionTable.Selectivity)
    (issued : Program.Runs (PvsParallel.SerialTable.Root.frame mc fc b d paired scratch rows idx order a z) cs out)
    (finite : Finite out.answer.value) (range : PvInterior.InsideRange a.toInt z.toInt out.answer.value) :
    ReadyMemory p mc fc (save after slot lvl out.proposal) := by
  cases written : out.proposal with
  | none => exact safe
  | some w =>
    obtain ⟨_,board,_,score,_,tag,legal⟩ := PvsParallel.SerialTable.Root.writes issued written
    refine ⟨⟨⟨after.tt.store_other safe.1.1.1 _ _ _ _ _ _ _ false _ (Or.inl rfl) (board ▸ legal),safe.1.1.2⟩,
      after.tt.store_finite after.supported safe.1.2 _ _ _ _ _ _ _ _ _⟩,after.tt.store_exactScores after.supported safe.2 _ _ _ _ _ _ _ _ _ ?_⟩
    intro exactTag
    have inside := ((TranspositionTable.classify_contract ..).2.1).mp (tag.symm.trans exactTag)
    have ⟨lo,hi⟩ := finite
    have fits : (BitVec.ofInt 32 out.answer.value).toInt = out.answer.value :=
      BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
    change true = true ∧ a.toInt < (BitVec.ofInt 32 out.answer.value).toInt ∧ (BitVec.ofInt 32 out.answer.value).toInt < z.toInt at inside
    rw [fits] at inside
    change -scoreMax ≤ (BitVec.ofInt 32 w.score).toInt ∧ (BitVec.ofInt 32 w.score).toInt ≤ scoreMax
    rw [score,fits]
    exact range inside.2.1 inside.2.2

/-- The pre-child probe selects the root slot. The epilogue stores into the physical table
    left by the children and always writes the midgame flag, even at depth = empties. -/
def runAt (idx : Nat) (mc fc) (b : Position) (depth : ExactDepth.Depth) (paired : Bool)
    (scratch : MoveList.Initialization.Memory Move.Move) (order : SerialRecursion.RootInput)
    (lvl : TranspositionTable.Selectivity) (child : Child l) (rows : List Row) (a z : Word) (m : Memory l) := do
  let code := PvsParallel.SerialTable.Root.frame mc fc b depth paired scratch rows idx order a z
  let (out,after) ← interpret child code m
  return (out.answer, save after (m.tt.image.slot b) lvl out.proposal)

/-- The final accepted row score and TT exact entry are computed from this trial's actual
    replies. A fail-high or fail-low trial still preserves the candidate and memory invariants. -/
theorem returnsAt (idx : Nat) (p : Pool) (mc fc) (b : Position) (depth : ExactDepth.Depth) (paired : Bool)
    (scratch : MoveList.Initialization.Memory Move.Move) (order : SerialRecursion.RootInput)
    (lvl : TranspositionTable.Selectivity) (child : Child l) (rows : List Row) (a z : Word) (m : Memory l)
    (children : ChildReturns p mc fc child) (ok : ReadyMemory p mc fc m)
    (cover : Root.Covers mc b (rows.map Row.square)) (positive : idx < rows.length) (window : Aspiration.Window a.toInt z.toInt) :
    ∃ ans next, runAt idx mc fc b depth paired scratch order lvl child rows a z m = some (ans,next) ∧
      ReadyMemory p mc fc next ∧ Finite ans.value ∧ PvInterior.InsideRange a.toInt z.toInt ans.value ∧
      ans.rows.map Row.square = rows.map Row.square ∧ (MidgameRows.Valid rows → MidgameRows.Valid ans.rows) ∧
      ans.rows.take idx = rows.take idx ∧ (a.toInt < ans.value → RowMaximum (ans.rows.drop idx) ans.value) := by
  let code := PvsParallel.SerialTable.Root.frame mc fc b depth paired scratch rows idx order a z
  have ready := SerialRootCalls.frame_ready_at_depth (fc := fc) (key := order.key) (sort := order.sort) (pv := order.pvLine)
    (depth := depth) (paired := paired) (mem := scratch) cover positive window
  rw [← PvsParallel.SerialTable.Root.erase mc fc b depth paired scratch rows idx order] at ready
  obtain ⟨out,after,ran,safe,cs,issued,words⟩ := interpret_returns (code := code) (valid := ReadyMemory p mc fc)
    (PvsParallel.SerialTable.Stack.ready_of_map ready) child (fun q issued m ok => children q m issued.2.1 issued.2.2 ok) m ok
  have finiteChild t (h : t ∈ cs) := (words t h).1
  obtain ⟨finite,keys⟩ := PvsParallel.SerialTable.WeakRoot.result issued cover positive finiteChild
  have range := PvsParallel.SerialTable.WeakTrial.root_range issued window (fun t h => (words t h).2)
  refine ⟨out.answer,save after (m.tt.image.slot b) lvl out.proposal,?_,?_,finite,range,keys,
    (fun h => MidgameRows.frame issued h finiteChild),root_suffix_rows issued cover window⟩
  · dsimp only [code] at ran
    simp only [runAt,ran,bind,Option.bind]
    rfl
  · exact save_ready after safe _ lvl issued finite range

def run (mc fc) (b : Position) (depth : ExactDepth.Depth) (paired : Bool)
    (scratch : MoveList.Initialization.Memory Move.Move) (order : SerialRecursion.RootInput)
    (lvl : TranspositionTable.Selectivity) (child : Child l) (rows : List Row) (a z : Word) (m : Memory l) :=
  runAt 0 mc fc b depth paired scratch order lvl child rows a z m

theorem returns (p : Pool) (mc fc) (b : Position) (depth : ExactDepth.Depth) (paired : Bool)
    (scratch : MoveList.Initialization.Memory Move.Move) (order : SerialRecursion.RootInput)
    (lvl : TranspositionTable.Selectivity) (child : Child l) (rows : List Row) (a z : Word) (m : Memory l)
    (children : ChildReturns p mc fc child) (ok : ReadyMemory p mc fc m)
    (cover : Root.Covers mc b (rows.map Row.square)) (positive : 0 < rows.length) (window : Aspiration.Window a.toInt z.toInt) :
    ∃ ans next, run mc fc b depth paired scratch order lvl child rows a z m = some (ans,next) ∧
      ReadyMemory p mc fc next ∧ Finite ans.value ∧ PvInterior.InsideRange a.toInt z.toInt ans.value ∧
      ans.rows.map Row.square = rows.map Row.square ∧ (MidgameRows.Valid rows → MidgameRows.Valid ans.rows) ∧
      (a.toInt < ans.value → RowMaximum ans.rows ans.value) := by
  obtain ⟨ans,next,ran,safe,finite,range,keys,hist,_,best⟩ := returnsAt 0 p mc fc b depth paired scratch order lvl child rows a z m children ok cover positive window
  exact ⟨ans,next,ran,safe,finite,range,keys,hist,by simpa using best⟩

end NeuralReversi.AspirationMir.SerialTable.MidgameRoot

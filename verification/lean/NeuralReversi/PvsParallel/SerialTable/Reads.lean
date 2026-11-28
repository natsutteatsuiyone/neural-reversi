module
public import NeuralReversi.PvsParallel.SerialTable.StableSample
public import NeuralReversi.PvsParallel.SerialTable.Slots
public import NeuralReversi.PvsParallel.SerialCacheStack.Program

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialTable

open TranspositionTable
open PvsBoardTree (Position GeneratedSquare play legalSquares)
open Cutoffs (History FlipConfig)
open Bitboard.Moves (X86Config)
open TableReads (ChildLookup ChildLookupFrom)

theorem Image.probe_reads (im : Image l) (supported : im.Supported history) (b : Position) :
    TableReads.ProbeReads history (im.attempts b) := by
  intro a member i
  have same : a = im.round b := by
    have inside := List.mem_of_mem_take member
    unfold Image.attempts at inside
    dsimp only at inside
    split at inside <;> simpa using inside
  subst a
  have observed (j : Fin 2) : TableReads.SnapshotRead history (sample l.bigEndian (im.cell b j)) :=
    StableSample.snapshot (supported.even (l.index b j)) (supported.resident (l.index b j))
  by_cases front : i = 0
  · simpa only [Image.round,front,ite_true] using observed 0
  · unfold Image.round
    simp only [front,ite_false]
    split
    · exact observed 1
    · exact .unavailable

theorem Image.lookup_reads (im : Image l) (supported : im.Supported history) (b : Position) (i : Fin 2) :
    TableReads.SlotLookup history b (im.lookupSlot b i) :=
  StableSample.lookup (supported.even (l.index b i)) (supported.resident (l.index b i)) b

/-- In the hit branch Rust does not load the back slot. -/
def Image.child (im : Image l) (supported : im.Supported history) (mc : X86Config) (fc : FlipConfig)
    (b : Position) (move : GeneratedSquare mc b.val) : ChildLookup mc fc b history :=
  let next := (play mc fc b move).val
  match read : im.lookupSlot next 0 with
  | some e => ⟨move,some e,none,read ▸ im.lookup_reads supported next 0,.miss⟩
  | none => ⟨move,none,im.lookupSlot next 1,.miss,im.lookup_reads supported next 1⟩

theorem Image.child_observation (im : Image l) (supported : im.Supported history) (mc fc)
    (b : Position) (m : GeneratedSquare mc b.val) :
    (im.child supported mc fc b m).observation = (m.val,im.lookup (play mc fc b m).val) := by
  unfold Image.child Image.lookup
  dsimp only
  split <;> simp_all [ChildLookupFrom.observation]

theorem Image.child_move (im : Image l) (supported : im.Supported history) (mc fc)
    (b : Position) (m : GeneratedSquare mc b.val) : (im.child supported mc fc b m).move = m := by
  unfold Image.child
  dsimp only
  split <;> rfl

def Image.children (im : Image l) (supported : im.Supported history) (mc : X86Config) (fc : FlipConfig)
    (b : Position) (q : PvsEnhancedCutoff.Query) : List (GeneratedSquare mc b.val) → List (ChildLookup mc fc b history)
  | [] => []
  | m :: ms =>
    let read := im.child supported mc fc b m
    read :: (if (PvsEnhancedCutoff.candidate q read.observation).isSome then []
      else im.children supported mc fc b q ms)

theorem Image.children_scan (im : Image l) (supported : im.Supported history) (mc fc)
    (b : Position) (q : PvsEnhancedCutoff.Query) (moves : List (GeneratedSquare mc b.val)) :
    PvsEnhancedCutoff.scan q ((im.children supported mc fc b q moves).map ChildLookupFrom.observation) =
      PvsEnhancedCutoff.scan q (moves.map fun m => (m.val,im.lookup (play mc fc b m).val)) := by
  induction moves with
  | nil => rfl
  | cons m rest ih =>
    unfold PvsEnhancedCutoff.scan at ih ⊢
    simp only [Image.children, List.map_cons, List.findSome?_cons, im.child_observation]
    cases found : PvsEnhancedCutoff.candidate q (m.val,im.lookup (play mc fc b m).val) <;> simp [ih]

theorem Image.children_moves (im : Image l) (supported : im.Supported history) (mc fc)
    (b : Position) (q : PvsEnhancedCutoff.Query) (moves : List (GeneratedSquare mc b.val)) :
    let misses := fun m : GeneratedSquare mc b.val =>
      (PvsEnhancedCutoff.candidate q (m.val,im.lookup (play mc fc b m).val)).isNone
    (im.children supported mc fc b q moves).map ChildLookupFrom.move =
      moves.take ((moves.takeWhile misses).length + 1) := by
  induction moves with
  | nil => rfl
  | cons m ms ih =>
    simp only [Image.children,List.map_cons,im.child_observation,im.child_move]
    cases found : PvsEnhancedCutoff.candidate q (m.val,im.lookup (play mc fc b m).val) <;>
      simp [List.takeWhile,found,ih]

theorem generation_order (mc : X86Config) (fc : FlipConfig) (b : Position) :
    MoveList.Generation.records fc b.val (Board.getMoves mc b.val) =
      (legalSquares mc b.val).map fun m => MoveList.Generation.entry fc b.val m.val := by
  unfold MoveList.Generation.records
  rw [show MoveList.Generation.squares (Board.getMoves mc b.val) = (legalSquares mc b.val).map Subtype.val from
    (PvsBoardTree.legalSquares_values mc b.val).symm,List.map_map]
  rfl

abbrev raw := SerialCalls.ScoreWords.encode

def Image.inputs (im : Image l) (supported : im.Supported history) (c : SerialScout.Config)
    (base : SerialExecution.Inputs mc fc) : SerialExecution.Inputs mc fc := fun i ctx =>
  let r := ctx.request
  let op := base i ctx
  let empty : List (ChildLookup mc fc r.board history) := []
  let action := fun attempts => GeneralNonPvMir.entryAction mc fc r.board r.depth (raw r.alpha) (raw r.beta)
    c.paired c.checked op.memory attempts empty
  let needsProbe := if r.pv then
    let moves := PvEntryMir.generated mc fc r.board c.paired op.memory
    PvEntryMir.dispatch r.depth.toNat (moves.len.getD 0 != 0) (Board.hasLegalMoves mc (PvsBoardTree.pass r.board).val)
      moves.wipeout.join.isSome == .continueSearch
    else action [] == .rest
  let attempts := if needsProbe then im.attempts r.board else []
  let q : PvsEnhancedCutoff.Query := ⟨r.depth,Cutoffs.exactSelectivity,true,⟨(raw r.alpha).raw⟩⟩
  let children := if !r.pv && 14 ≤ r.depth.toNat && action attempts == .rest then
    im.children supported mc fc r.board q (legalSquares mc r.board.val) else []
  ⟨op.memory,op.cache,history,attempts,children,op.ordering,op.tie⟩

theorem Image.inputs_shared (im : Image l) (supported : im.Supported history) (c : SerialScout.Config)
    (base : SerialExecution.Inputs mc fc) (pub : ExactTableHistory.Publications mc fc history)
    (hints : TableReads.LegalHints history) :
    SerialCacheStack.SharedFacts (im.inputs supported c base) limit := by
  refine ⟨?_,fun _ _ _ _ _ => pub,fun _ _ _ _ _ => hints⟩
  intro i ctx role _ _
  simp only [Image.inputs,role,Bool.false_eq_true,ite_false,beq_iff_eq]
  split
  · exact im.probe_reads supported ctx.request.board
  · simp [TableReads.ProbeReads,TableReads.ProbeReadsFrom]

end NeuralReversi.PvsParallel.SerialTable

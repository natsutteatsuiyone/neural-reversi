module
public import NeuralReversi.AspirationMir.SmallRoot
public import NeuralReversi.PvsParallel.RootOwnerMir.Serial

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialRoot

open Aspiration (Window SoundResult)
open AspirationMir (applyRootWrite)
open AspirationMir.SmallRoot (Answer candidates)
open Bitboard (Square)
open Bitboard.Moves (X86Config)
open Cutoffs (FlipConfig)
open PvsBoardTree (Position GeneratedSquare build play minimax)
open PvsMoveLoop (ChildSearch ChildRelation MoveCalls ScanCalls moveScore scoreInf)
open RootMoveStorage (Row)
open RootOwnerMir (Replies observed)
open RootOwnerMir.Slice (Exit Path Supplies Visit)
open RootReplay (Snapshot start serial)
open Threading.SplitWindow (marksCutoff)

open RootOwnerMir.Serial (schedule owner_returns)

def trial (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : ExactDepth.Depth)
    (paired : Bool) (mem : MoveList.Initialization.Memory Move.Move) (initial : List Row) (idx : Nat)
    (key : Square → BitVec 32) (sort : RootFrameMir.LibraryOrder)
    (pv : Square → List Square) (a z : BitVec 32) (replies : Replies (GeneratedSquare mc b.val)) :
    Option Answer := do
  let roots := initial.map Row.square
  let generated := MoveList.Generation.fill false paired fc b.val (Root.remainingMask mc b.val roots idx) mem
  let count ← generated.len
  let cached ← generated.wipeout
  if h : 0 < count then
    match rootEntry count cached.isSome h, cached with
    | .wipeout, some sq =>
      let (writes, value) := rootWipeoutExit sq
      return ⟨value, writes.foldl (applyRootWrite pv) (RootMoveStorage.resetFrom initial idx)⟩
    | .header _ _ _ _, none =>
      let moves := candidates mc fc b roots idx key sort
      let start := RootReplay.start a.toInt
      let final ← RootOwnerMir.Slice.run Subtype.val replies schedule depth.toNat rootMinimumSplitDepth
        z.toInt 0 0 moves.length moves start
      guard (!final.handoff)
      return ⟨(rootEpilogue idx final.snapshot.state.best).2,
        RootSerialMir.rowsAfter Subtype.val (fun m => pv m.val) (observed replies) z.toInt moves start
          (RootMoveStorage.resetFrom initial idx)⟩
    | _, _ => none
  else none

theorem trial_returns (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : ExactDepth.Depth)
    (paired : Bool) (mem : MoveList.Initialization.Memory Move.Move) (initial : List Row) (idx : Nat)
    (key : Square → BitVec 32) (sort : RootFrameMir.LibraryOrder)
    (pv : Square → List Square) (a z : BitVec 32) (replies : Replies (GeneratedSquare mc b.val))
    (covers : Root.Covers mc b (initial.map Row.square)) (within : idx < initial.length)
    (full : ExactDepth.Full b depth) (window : Window a.toInt z.toInt)
    (ih : ∀ m p lo hi, Window lo hi → (p = false → hi = lo + 1) →
      ∃ score, replies m p lo hi = some score ∧ Replayed (build mc fc (play mc fc b m).val) p lo hi score) :
    ∃ answer snap, trial mc fc b depth paired mem initial idx key sort pv a z replies = some answer ∧
      answer.value = snap.state.best ∧
      RootSuffix.Writes initial idx a.toInt z.toInt answer.rows snap ∧
      RootEvidence.Attempt mc fc b (initial.map Row.square) idx a.toInt z.toInt answer.value snap.scores ∧
      RootOrder.SoundAttempt mc fc b (initial.map Row.square) idx a.toInt z.toInt answer.value snap.scores := by
  let roots := initial.map Row.square
  have positive : idx < roots.length := by simpa [roots] using within
  let selected := Root.remainingMask mc b.val roots idx
  have subset : MoveList.Generation.IsSubset selected (Board.getMoves mc b.val) := by
    intro i bit
    simp only [selected, Root.remainingMask, BitVec.getLsbD_and, Bool.and_eq_true] at bit
    exact bit.1
  obtain ⟨_, _, count, nonempty, length⟩ :=
    ConcurrentReplay.generatedRoot_positive mc fc b roots idx depth paired mem covers positive full
  have cache := (MoveList.Generation.generationContract_proven mc fc b.val selected false
    paired mem b.property subset (by simp)).2.2.2.2
  generalize value : ((MoveList.Generation.candidatesForCache (MoveList.Generation.records fc b.val selected)).reverse.find?
    (MoveList.Wipeout.isMatch b.val.opponent)).map MoveList.Wipeout.Move.square = hit at cache
  let start := RootReplay.start a.toInt
  have cleared := RootSuffix.Writes.reset (initial := initial) (idx := idx) (alpha := a.toInt) (beta := z.toInt)
  cases hit with
  | none =>
    let moves := candidates mc fc b roots idx key sort
    have order := (RootFrameMir.ordered_candidates mc fc b roots idx key sort).1
    obtain ⟨visits, result, path, supplied, returned, noSplit, snapshot, calls⟩ :=
      owner_returns (Move := GeneratedSquare mc b.val) Subtype.val replies
        (fun m p lo hi s => Replayed (build mc fc (play mc fc b m).val) p lo hi s)
        depth.toNat rootMinimumSplitDepth a.toInt z.toInt moves window ih
    have attempt : RootEvidence.Attempt mc fc b roots idx a.toInt z.toInt
        result.snapshot.state.best result.snapshot.scores := by
      rw [snapshot]
      exact .sequential moves (observed replies) order calls
    have writes := RootSerialMir.rowsAfter_writes Subtype.val (fun m => pv m.val) (observed replies) moves start cleared
      (Root.roots_nodup covers) (by
        intro m member
        rw [List.map_drop]
        exact (Root.choices_squares_mem covers m.val).mp (List.mem_map_of_mem (order.mem_iff.mp member)))
    refine ⟨⟨result.snapshot.state.best, _⟩, result.snapshot, ?_, rfl,
      snapshot.symm ▸ writes, attempt, attempt.sound covers positive window⟩
    dsimp only [roots, selected] at length cache
    simp [trial, length, cache, nonempty, rootEntry, rootEpilogue, roots, moves, start] at returned ⊢
    simp [returned, noSplit, guard]
  | some sq =>
    obtain ⟨move, member, square, wipe⟩ :=
      ConcurrentReplay.generatedWipeout mc fc b roots idx paired mem sq covers cache
    have legal := (Root.choices_squares_mem covers move.val).mp (List.mem_map_of_mem member)
    have shape := (cleared.scores (Root.roots_nodup covers)).1
    have inRows : sq ∈ ((RootMoveStorage.resetFrom initial idx).drop idx).map Row.square := by
      rw [List.map_drop, shape]
      simpa [roots, square] using legal
    have written := RootSuffix.Writes.record cleared (⟨sq, a.toInt, PvsMoveLoop.scoreMax⟩ : RootOrder.Report)
      true (pv sq) inRows rfl
    have attempt : RootEvidence.Attempt mc fc b roots idx a.toInt z.toInt PvsMoveLoop.scoreMax
        (start.record z.toInt ⟨sq, a.toInt, PvsMoveLoop.scoreMax⟩).scores := by
      rw [← square]
      exact .wipeout move member wipe
    let final := start.record z.toInt ⟨sq, a.toInt, PvsMoveLoop.scoreMax⟩
    refine ⟨⟨PvsMoveLoop.scoreMax, _⟩, final, ?_, rfl, written, attempt, attempt.sound covers positive window⟩
    dsimp only [roots, selected] at length cache
    simp [trial, nonempty, rootEntry, length, cache, rootWipeoutExit, applyRootWrite, PvsMoveLoop.scoreMax]

end NeuralReversi.PvsParallel.SerialRoot

import NeuralReversi.PvsParallel.ConcurrentReplay
import NeuralReversi.MoveList.Generation
import NeuralReversi.PvsParallel.ExactDepth
import NeuralReversi.PvsParallel.WipeoutExit

namespace NeuralReversi.PvsParallel.ConcurrentReplay

open Bitboard (Square)
open PvsBoardTree (Position)
open PvsMoveLoop (scoreMax scoreInf)
open MoveList

theorem generatedRoot_positive
    (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config) (b : Position)
    (roots : List Square) (idx : Nat) (depth : ExactDepth.Depth) (paired : Bool)
    (mem : MoveList.Initialization.Memory Move.Move) (covers : Root.Covers mc b roots)
    (inRange : idx < roots.length) (full : ExactDepth.Full b depth) :
    0 < depth.toNat ∧ Root.remainingMask mc b.val roots idx ≠ 0 ∧ ∃ size, 0 < size ∧
      (Generation.fill false paired fc b.val (Root.remainingMask mc b.val roots idx) mem).len = some size := by
  let selected := Root.remainingMask mc b.val roots idx
  let sq := roots[idx]
  have inSuffix : sq ∈ roots.drop idx := by
    have : (roots.drop idx)[0]? = some sq := by simp [sq, List.getElem?_drop, inRange]
    exact List.mem_of_getElem? this
  have bit := (Root.maskContract_proven mc b roots idx covers).1 sq |>.mpr inSuffix
  obtain ⟨move, _, _⟩ := List.mem_map.mp ((Root.choices_squares_mem covers sq).mpr inSuffix)
  have subset : Generation.IsSubset selected (Board.getMoves mc b.val) := by
    intro i
    simp only [selected, Root.remainingMask, BitVec.getLsbD_and, Bool.and_eq_true]
    exact And.left
  have length := (Generation.generationContract_proven mc fc b.val selected false paired mem
    b.property subset (by simp)).1
  have nonzero : selected ≠ 0 := by
    intro zero
    change selected.getLsbD sq.val = true at bit
    simp [zero] at bit
  refine ⟨(ExactDepth.move_depth mc fc b depth move full).1, nonzero, _, ?_, length⟩
  have member : sq ∈ Generation.squares selected := List.mem_filter.mpr ⟨by simp, bit⟩
  exact List.length_pos_of_mem (List.mem_map_of_mem (f := Generation.entry fc b.val) member)

theorem generatedWipeout
    (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config)
    (b : Position) (roots : List Square) (idx : Nat)
    (paired : Bool) (mem : MoveList.Initialization.Memory Move.Move) (square : Square)
    (covers : Root.Covers mc b roots)
    (cache : (Generation.fill false paired fc b.val (Root.remainingMask mc b.val roots idx) mem).wipeout =
      some (some square)) :
    ∃ move ∈ Root.choices mc b roots idx, move.val = square ∧ EndgameCutoffs.wipesOut fc b.val move.val = true := by
  let selected := Root.remainingMask mc b.val roots idx
  have subset : Generation.IsSubset selected (Board.getMoves mc b.val) := by
    intro i
    simp only [selected, Root.remainingMask, BitVec.getLsbD_and, Bool.and_eq_true]
    exact And.left
  have hit := (Generation.generationContract_proven mc fc b.val selected false paired mem
    b.property subset (by simp)).2.2.2.2
  rw [cache] at hit
  obtain ⟨candidate, found, chosen⟩ := Option.map_eq_some_iff.mp (Option.some.inj hit).symm
  have comparison := List.find?_some found
  have entryMem := List.mem_reverse.mp (List.mem_of_find?_eq_some found)
  obtain ⟨entry, included, rfl⟩ := List.mem_map.mp entryMem
  obtain ⟨sq, member, rfl⟩ := List.mem_map.mp included
  change sq = square at chosen
  subst square
  have bit : selected.getLsbD sq.val = true := (List.mem_filter.mp member).2
  have suffix := (Root.maskContract_proven mc b roots idx covers).1 sq |>.mp bit
  obtain ⟨move, legal, same⟩ := List.mem_map.mp ((Root.choices_squares_mem covers sq).mpr suffix)
  have wipe : EndgameCutoffs.wipesOut fc b.val move.val = true := by
    rw [same]
    exact comparison
  exact ⟨move, legal, same, wipe⟩

theorem rootAttempt_of_generated_wipeout
    (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config) (endian : Bool)
    (b : Position) (roots : List Square) (idx : Nat) (alpha beta : Int)
    (paired : Bool) (mem : MoveList.Initialization.Memory Move.Move) (square : Square)
    (covers : Root.Covers mc b roots)
    (cache : (Generation.fill false paired fc b.val (Root.remainingMask mc b.val roots idx) mem).wipeout =
      some (some square)) :
    let (writes, returned) := rootWipeoutExit square
    let payload := writes.foldl (fun rows (sq, s, pv) =>
      RootOrder.writeScore rows sq (if pv then s else -scoreInf)) (fun _ => -scoreInf)
    RootAttempt mc fc endian b roots idx alpha beta returned payload := by
  change RootAttempt mc fc endian b roots idx alpha beta scoreMax
    (fun query => if query = square then scoreMax else -scoreInf)
  obtain ⟨move, legal, same, wipe⟩ := generatedWipeout mc fc b roots idx paired mem square covers cache
  let observed : Observations mc fc endian [] := {
    identify := fun _ => 0
    before := fun _ _ => False
    irrefl := fun _ h => h
    trans := fun h _ => h
    words := by simp
    tablePublications := by simp
    evidence := by simp }
  refine ⟨[], observed, 0, ?_⟩
  have evidence := RootEvidence.AttemptWithUses.wipeout
    (completed := UsedChild observed.identify observed.before [] 0)
    (idx := idx) (alpha := alpha) (beta := beta) move legal wipe
  rw [← same]
  exact evidence

end NeuralReversi.PvsParallel.ConcurrentReplay

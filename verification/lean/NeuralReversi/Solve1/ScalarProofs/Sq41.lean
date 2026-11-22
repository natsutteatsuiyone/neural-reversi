import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq40
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_41_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 41 0)
      (laneIndex 41 0 (Projection.scatter [1,9,17,25,33,41,49,57] (BitVec.ofNat 8 n.val))) =
    laneReference 41 0 (Projection.scatter [1,9,17,25,33,41,49,57] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_41_0 (p : Word) :
    LineTable.reference (lanePos 41 0) (laneIndex 41 0 p) = laneReference 41 0 p := by
  let cells : List Nat := [1,9,17,25,33,41,49,57]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 41 0) (by decide)
  have hh := finite_41_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_41_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 41 1)
      (laneIndex 41 1 (Projection.scatter [40,41,42,43,44,45,46,47] (BitVec.ofNat 8 n.val))) =
    laneReference 41 1 (Projection.scatter [40,41,42,43,44,45,46,47] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_41_1 (p : Word) :
    LineTable.reference (lanePos 41 1) (laneIndex 41 1 p) = laneReference 41 1 p := by
  let cells : List Nat := [40,41,42,43,44,45,46,47]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 41 1) (by decide)
  have hh := finite_41_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_41_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 41 2)
      (laneIndex 41 2 (Projection.scatter [6,13,20,27,34,41,50,59] (BitVec.ofNat 8 n.val))) =
    laneReference 41 2 (Projection.scatter [6,13,20,27,34,41,50,59] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_41_2 (p : Word) :
    LineTable.reference (lanePos 41 2) (laneIndex 41 2 p) = laneReference 41 2 p := by
  let cells : List Nat := [6,13,20,27,34,41,50,59]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 41 2) (by decide)
  have hh := finite_41_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_41_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 41 3)
      (laneIndex 41 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 41 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_41_3 (p : Word) :
    LineTable.reference (lanePos 41 3) (laneIndex 41 3 p) = laneReference 41 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 41 3) (by decide)
  have hh := finite_41_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

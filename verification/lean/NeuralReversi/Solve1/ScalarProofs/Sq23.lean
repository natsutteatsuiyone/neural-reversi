import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq22
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_23_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 23 0)
      (laneIndex 23 0 (Projection.scatter [7,15,23,31,39,47,55,63] (BitVec.ofNat 8 n.val))) =
    laneReference 23 0 (Projection.scatter [7,15,23,31,39,47,55,63] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_23_0 (p : Word) :
    LineTable.reference (lanePos 23 0) (laneIndex 23 0 p) = laneReference 23 0 p := by
  let cells : List Nat := [7,15,23,31,39,47,55,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 23 0) (by decide)
  have hh := finite_23_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_23_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 23 1)
      (laneIndex 23 1 (Projection.scatter [16,17,18,19,20,21,22,23] (BitVec.ofNat 8 n.val))) =
    laneReference 23 1 (Projection.scatter [16,17,18,19,20,21,22,23] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_23_1 (p : Word) :
    LineTable.reference (lanePos 23 1) (laneIndex 23 1 p) = laneReference 23 1 p := by
  let cells : List Nat := [16,17,18,19,20,21,22,23]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 23 1) (by decide)
  have hh := finite_23_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_23_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 23 2)
      (laneIndex 23 2 (Projection.scatter [5,14,23,30,37,44,51,58] (BitVec.ofNat 8 n.val))) =
    laneReference 23 2 (Projection.scatter [5,14,23,30,37,44,51,58] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_23_2 (p : Word) :
    LineTable.reference (lanePos 23 2) (laneIndex 23 2 p) = laneReference 23 2 p := by
  let cells : List Nat := [5,14,23,30,37,44,51,58]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 23 2) (by decide)
  have hh := finite_23_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_23_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 23 3)
      (laneIndex 23 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 23 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_23_3 (p : Word) :
    LineTable.reference (lanePos 23 3) (laneIndex 23 3 p) = laneReference 23 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 23 3) (by decide)
  have hh := finite_23_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

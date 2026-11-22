import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq04
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_5_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 5 0)
      (laneIndex 5 0 (Projection.scatter [5,13,21,29,37,45,53,61] (BitVec.ofNat 8 n.val))) =
    laneReference 5 0 (Projection.scatter [5,13,21,29,37,45,53,61] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_5_0 (p : Word) :
    LineTable.reference (lanePos 5 0) (laneIndex 5 0 p) = laneReference 5 0 p := by
  let cells : List Nat := [5,13,21,29,37,45,53,61]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 5 0) (by decide)
  have hh := finite_5_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_5_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 5 1)
      (laneIndex 5 1 (Projection.scatter [0,1,2,3,4,5,6,7] (BitVec.ofNat 8 n.val))) =
    laneReference 5 1 (Projection.scatter [0,1,2,3,4,5,6,7] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_5_1 (p : Word) :
    LineTable.reference (lanePos 5 1) (laneIndex 5 1 p) = laneReference 5 1 p := by
  let cells : List Nat := [0,1,2,3,4,5,6,7]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 5 1) (by decide)
  have hh := finite_5_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_5_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 5 2)
      (laneIndex 5 2 (Projection.scatter [5,12,14,19,23,26,33,40] (BitVec.ofNat 8 n.val))) =
    laneReference 5 2 (Projection.scatter [5,12,14,19,23,26,33,40] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_5_2 (p : Word) :
    LineTable.reference (lanePos 5 2) (laneIndex 5 2 p) = laneReference 5 2 p := by
  let cells : List Nat := [5,12,14,19,23,26,33,40]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 5 2) (by decide)
  have hh := finite_5_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_5_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 5 3)
      (laneIndex 5 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 5 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_5_3 (p : Word) :
    LineTable.reference (lanePos 5 3) (laneIndex 5 3 p) = laneReference 5 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 5 3) (by decide)
  have hh := finite_5_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq20
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_21_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 21 0)
      (laneIndex 21 0 (Projection.scatter [5,13,21,29,37,45,53,61] (BitVec.ofNat 8 n.val))) =
    laneReference 21 0 (Projection.scatter [5,13,21,29,37,45,53,61] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_21_0 (p : Word) :
    LineTable.reference (lanePos 21 0) (laneIndex 21 0 p) = laneReference 21 0 p := by
  let cells : List Nat := [5,13,21,29,37,45,53,61]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 21 0) (by decide)
  have hh := finite_21_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_21_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 21 1)
      (laneIndex 21 1 (Projection.scatter [16,17,18,19,20,21,22,23] (BitVec.ofNat 8 n.val))) =
    laneReference 21 1 (Projection.scatter [16,17,18,19,20,21,22,23] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_21_1 (p : Word) :
    LineTable.reference (lanePos 21 1) (laneIndex 21 1 p) = laneReference 21 1 p := by
  let cells : List Nat := [16,17,18,19,20,21,22,23]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 21 1) (by decide)
  have hh := finite_21_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_21_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 21 2)
      (laneIndex 21 2 (Projection.scatter [7,14,21,28,35,42,49,56] (BitVec.ofNat 8 n.val))) =
    laneReference 21 2 (Projection.scatter [7,14,21,28,35,42,49,56] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_21_2 (p : Word) :
    LineTable.reference (lanePos 21 2) (laneIndex 21 2 p) = laneReference 21 2 p := by
  let cells : List Nat := [7,14,21,28,35,42,49,56]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 21 2) (by decide)
  have hh := finite_21_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_21_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 21 3)
      (laneIndex 21 3 (Projection.scatter [3,12,21,30,39] (BitVec.ofNat 8 n.val))) =
    laneReference 21 3 (Projection.scatter [3,12,21,30,39] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_21_3 (p : Word) :
    LineTable.reference (lanePos 21 3) (laneIndex 21 3 p) = laneReference 21 3 p := by
  let cells : List Nat := [3,12,21,30,39]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 21 3) (by decide)
  have hh := finite_21_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

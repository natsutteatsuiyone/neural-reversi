import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq12
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_13_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 13 0)
      (laneIndex 13 0 (Projection.scatter [5,13,21,29,37,45,53,61] (BitVec.ofNat 8 n.val))) =
    laneReference 13 0 (Projection.scatter [5,13,21,29,37,45,53,61] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_13_0 (p : Word) :
    LineTable.reference (lanePos 13 0) (laneIndex 13 0 p) = laneReference 13 0 p := by
  let cells : List Nat := [5,13,21,29,37,45,53,61]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 13 0) (by decide)
  have hh := finite_13_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_13_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 13 1)
      (laneIndex 13 1 (Projection.scatter [8,9,10,11,12,13,14,15] (BitVec.ofNat 8 n.val))) =
    laneReference 13 1 (Projection.scatter [8,9,10,11,12,13,14,15] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_13_1 (p : Word) :
    LineTable.reference (lanePos 13 1) (laneIndex 13 1 p) = laneReference 13 1 p := by
  let cells : List Nat := [8,9,10,11,12,13,14,15]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 13 1) (by decide)
  have hh := finite_13_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_13_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 13 2)
      (laneIndex 13 2 (Projection.scatter [13,20,22,27,31,34,41,48] (BitVec.ofNat 8 n.val))) =
    laneReference 13 2 (Projection.scatter [13,20,22,27,31,34,41,48] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_13_2 (p : Word) :
    LineTable.reference (lanePos 13 2) (laneIndex 13 2 p) = laneReference 13 2 p := by
  let cells : List Nat := [13,20,22,27,31,34,41,48]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 13 2) (by decide)
  have hh := finite_13_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_13_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 13 3)
      (laneIndex 13 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 13 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_13_3 (p : Word) :
    LineTable.reference (lanePos 13 3) (laneIndex 13 3 p) = laneReference 13 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 13 3) (by decide)
  have hh := finite_13_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

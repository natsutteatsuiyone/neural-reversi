import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq09
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_10_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 10 0)
      (laneIndex 10 0 (Projection.scatter [2,10,18,26,34,42,50,58] (BitVec.ofNat 8 n.val))) =
    laneReference 10 0 (Projection.scatter [2,10,18,26,34,42,50,58] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_10_0 (p : Word) :
    LineTable.reference (lanePos 10 0) (laneIndex 10 0 p) = laneReference 10 0 p := by
  let cells : List Nat := [2,10,18,26,34,42,50,58]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 10 0) (by decide)
  have hh := finite_10_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_10_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 10 1)
      (laneIndex 10 1 (Projection.scatter [8,9,10,11,12,13,14,15] (BitVec.ofNat 8 n.val))) =
    laneReference 10 1 (Projection.scatter [8,9,10,11,12,13,14,15] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_10_1 (p : Word) :
    LineTable.reference (lanePos 10 1) (laneIndex 10 1 p) = laneReference 10 1 p := by
  let cells : List Nat := [8,9,10,11,12,13,14,15]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 10 1) (by decide)
  have hh := finite_10_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_10_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 10 2)
      (laneIndex 10 2 (Projection.scatter [10,17,19,24,28,37,46,55] (BitVec.ofNat 8 n.val))) =
    laneReference 10 2 (Projection.scatter [10,17,19,24,28,37,46,55] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_10_2 (p : Word) :
    LineTable.reference (lanePos 10 2) (laneIndex 10 2 p) = laneReference 10 2 p := by
  let cells : List Nat := [10,17,19,24,28,37,46,55]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 10 2) (by decide)
  have hh := finite_10_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_10_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 10 3)
      (laneIndex 10 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 10 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_10_3 (p : Word) :
    LineTable.reference (lanePos 10 3) (laneIndex 10 3 p) = laneReference 10 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 10 3) (by decide)
  have hh := finite_10_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq15
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_16_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 16 0)
      (laneIndex 16 0 (Projection.scatter [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 n.val))) =
    laneReference 16 0 (Projection.scatter [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_16_0 (p : Word) :
    LineTable.reference (lanePos 16 0) (laneIndex 16 0 p) = laneReference 16 0 p := by
  let cells : List Nat := [0,8,16,24,32,40,48,56]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 16 0) (by decide)
  have hh := finite_16_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_16_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 16 1)
      (laneIndex 16 1 (Projection.scatter [16,17,18,19,20,21,22,23] (BitVec.ofNat 8 n.val))) =
    laneReference 16 1 (Projection.scatter [16,17,18,19,20,21,22,23] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_16_1 (p : Word) :
    LineTable.reference (lanePos 16 1) (laneIndex 16 1 p) = laneReference 16 1 p := by
  let cells : List Nat := [16,17,18,19,20,21,22,23]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 16 1) (by decide)
  have hh := finite_16_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_16_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 16 2)
      (laneIndex 16 2 (Projection.scatter [2,9,16,25,34,43,52,61] (BitVec.ofNat 8 n.val))) =
    laneReference 16 2 (Projection.scatter [2,9,16,25,34,43,52,61] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_16_2 (p : Word) :
    LineTable.reference (lanePos 16 2) (laneIndex 16 2 p) = laneReference 16 2 p := by
  let cells : List Nat := [2,9,16,25,34,43,52,61]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 16 2) (by decide)
  have hh := finite_16_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_16_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 16 3)
      (laneIndex 16 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 16 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_16_3 (p : Word) :
    LineTable.reference (lanePos 16 3) (laneIndex 16 3 p) = laneReference 16 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 16 3) (by decide)
  have hh := finite_16_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

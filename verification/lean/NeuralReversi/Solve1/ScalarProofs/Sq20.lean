import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq19
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_20_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 20 0)
      (laneIndex 20 0 (Projection.scatter [4,12,20,28,36,44,52,60] (BitVec.ofNat 8 n.val))) =
    laneReference 20 0 (Projection.scatter [4,12,20,28,36,44,52,60] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_20_0 (p : Word) :
    LineTable.reference (lanePos 20 0) (laneIndex 20 0 p) = laneReference 20 0 p := by
  let cells : List Nat := [4,12,20,28,36,44,52,60]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 20 0) (by decide)
  have hh := finite_20_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_20_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 20 1)
      (laneIndex 20 1 (Projection.scatter [16,17,18,19,20,21,22,23] (BitVec.ofNat 8 n.val))) =
    laneReference 20 1 (Projection.scatter [16,17,18,19,20,21,22,23] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_20_1 (p : Word) :
    LineTable.reference (lanePos 20 1) (laneIndex 20 1 p) = laneReference 20 1 p := by
  let cells : List Nat := [16,17,18,19,20,21,22,23]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 20 1) (by decide)
  have hh := finite_20_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_20_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 20 2)
      (laneIndex 20 2 (Projection.scatter [6,13,20,27,34,41,48] (BitVec.ofNat 8 n.val))) =
    laneReference 20 2 (Projection.scatter [6,13,20,27,34,41,48] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_20_2 (p : Word) :
    LineTable.reference (lanePos 20 2) (laneIndex 20 2 p) = laneReference 20 2 p := by
  let cells : List Nat := [6,13,20,27,34,41,48]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 20 2) (by decide)
  have hh := finite_20_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_20_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 20 3)
      (laneIndex 20 3 (Projection.scatter [2,11,20,29,38,47] (BitVec.ofNat 8 n.val))) =
    laneReference 20 3 (Projection.scatter [2,11,20,29,38,47] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_20_3 (p : Word) :
    LineTable.reference (lanePos 20 3) (laneIndex 20 3 p) = laneReference 20 3 p := by
  let cells : List Nat := [2,11,20,29,38,47]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 20 3) (by decide)
  have hh := finite_20_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

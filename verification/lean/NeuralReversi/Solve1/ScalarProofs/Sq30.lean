import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq29
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_30_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 30 0)
      (laneIndex 30 0 (Projection.scatter [6,14,22,30,38,46,54,62] (BitVec.ofNat 8 n.val))) =
    laneReference 30 0 (Projection.scatter [6,14,22,30,38,46,54,62] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_30_0 (p : Word) :
    LineTable.reference (lanePos 30 0) (laneIndex 30 0 p) = laneReference 30 0 p := by
  let cells : List Nat := [6,14,22,30,38,46,54,62]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 30 0) (by decide)
  have hh := finite_30_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_30_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 30 1)
      (laneIndex 30 1 (Projection.scatter [24,25,26,27,28,29,30,31] (BitVec.ofNat 8 n.val))) =
    laneReference 30 1 (Projection.scatter [24,25,26,27,28,29,30,31] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_30_1 (p : Word) :
    LineTable.reference (lanePos 30 1) (laneIndex 30 1 p) = laneReference 30 1 p := by
  let cells : List Nat := [24,25,26,27,28,29,30,31]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 30 1) (by decide)
  have hh := finite_30_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_30_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 30 2)
      (laneIndex 30 2 (Projection.scatter [3,12,21,30,37,44,51,58] (BitVec.ofNat 8 n.val))) =
    laneReference 30 2 (Projection.scatter [3,12,21,30,37,44,51,58] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_30_2 (p : Word) :
    LineTable.reference (lanePos 30 2) (laneIndex 30 2 p) = laneReference 30 2 p := by
  let cells : List Nat := [3,12,21,30,37,44,51,58]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 30 2) (by decide)
  have hh := finite_30_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_30_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 30 3)
      (laneIndex 30 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 30 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_30_3 (p : Word) :
    LineTable.reference (lanePos 30 3) (laneIndex 30 3 p) = laneReference 30 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 30 3) (by decide)
  have hh := finite_30_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

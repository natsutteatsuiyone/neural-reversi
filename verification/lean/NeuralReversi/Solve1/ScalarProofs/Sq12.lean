import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq11
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_12_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 12 0)
      (laneIndex 12 0 (Projection.scatter [4,12,20,28,36,44,52,60] (BitVec.ofNat 8 n.val))) =
    laneReference 12 0 (Projection.scatter [4,12,20,28,36,44,52,60] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_12_0 (p : Word) :
    LineTable.reference (lanePos 12 0) (laneIndex 12 0 p) = laneReference 12 0 p := by
  let cells : List Nat := [4,12,20,28,36,44,52,60]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 12 0) (by decide)
  have hh := finite_12_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_12_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 12 1)
      (laneIndex 12 1 (Projection.scatter [8,9,10,11,12,13,14,15] (BitVec.ofNat 8 n.val))) =
    laneReference 12 1 (Projection.scatter [8,9,10,11,12,13,14,15] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_12_1 (p : Word) :
    LineTable.reference (lanePos 12 1) (laneIndex 12 1 p) = laneReference 12 1 p := by
  let cells : List Nat := [8,9,10,11,12,13,14,15]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 12 1) (by decide)
  have hh := finite_12_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_12_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 12 2)
      (laneIndex 12 2 (Projection.scatter [12,19,21,26,30,33,39,40] (BitVec.ofNat 8 n.val))) =
    laneReference 12 2 (Projection.scatter [12,19,21,26,30,33,39,40] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_12_2 (p : Word) :
    LineTable.reference (lanePos 12 2) (laneIndex 12 2 p) = laneReference 12 2 p := by
  let cells : List Nat := [12,19,21,26,30,33,39,40]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 12 2) (by decide)
  have hh := finite_12_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_12_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 12 3)
      (laneIndex 12 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 12 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_12_3 (p : Word) :
    LineTable.reference (lanePos 12 3) (laneIndex 12 3 p) = laneReference 12 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 12 3) (by decide)
  have hh := finite_12_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

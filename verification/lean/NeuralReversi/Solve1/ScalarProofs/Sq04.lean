import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq03
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_4_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 4 0)
      (laneIndex 4 0 (Projection.scatter [4,12,20,28,36,44,52,60] (BitVec.ofNat 8 n.val))) =
    laneReference 4 0 (Projection.scatter [4,12,20,28,36,44,52,60] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_4_0 (p : Word) :
    LineTable.reference (lanePos 4 0) (laneIndex 4 0 p) = laneReference 4 0 p := by
  let cells : List Nat := [4,12,20,28,36,44,52,60]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 4 0) (by decide)
  have hh := finite_4_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_4_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 4 1)
      (laneIndex 4 1 (Projection.scatter [0,1,2,3,4,5,6,7] (BitVec.ofNat 8 n.val))) =
    laneReference 4 1 (Projection.scatter [0,1,2,3,4,5,6,7] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_4_1 (p : Word) :
    LineTable.reference (lanePos 4 1) (laneIndex 4 1 p) = laneReference 4 1 p := by
  let cells : List Nat := [0,1,2,3,4,5,6,7]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 4 1) (by decide)
  have hh := finite_4_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_4_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 4 2)
      (laneIndex 4 2 (Projection.scatter [4,11,13,18,22,25,31,32] (BitVec.ofNat 8 n.val))) =
    laneReference 4 2 (Projection.scatter [4,11,13,18,22,25,31,32] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_4_2 (p : Word) :
    LineTable.reference (lanePos 4 2) (laneIndex 4 2 p) = laneReference 4 2 p := by
  let cells : List Nat := [4,11,13,18,22,25,31,32]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 4 2) (by decide)
  have hh := finite_4_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_4_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 4 3)
      (laneIndex 4 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 4 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_4_3 (p : Word) :
    LineTable.reference (lanePos 4 3) (laneIndex 4 3 p) = laneReference 4 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 4 3) (by decide)
  have hh := finite_4_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

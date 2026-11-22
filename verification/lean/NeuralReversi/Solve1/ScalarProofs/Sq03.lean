import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq02
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_3_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 3 0)
      (laneIndex 3 0 (Projection.scatter [3,11,19,27,35,43,51,59] (BitVec.ofNat 8 n.val))) =
    laneReference 3 0 (Projection.scatter [3,11,19,27,35,43,51,59] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_3_0 (p : Word) :
    LineTable.reference (lanePos 3 0) (laneIndex 3 0 p) = laneReference 3 0 p := by
  let cells : List Nat := [3,11,19,27,35,43,51,59]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 3 0) (by decide)
  have hh := finite_3_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_3_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 3 1)
      (laneIndex 3 1 (Projection.scatter [0,1,2,3,4,5,6,7] (BitVec.ofNat 8 n.val))) =
    laneReference 3 1 (Projection.scatter [0,1,2,3,4,5,6,7] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_3_1 (p : Word) :
    LineTable.reference (lanePos 3 1) (laneIndex 3 1 p) = laneReference 3 1 p := by
  let cells : List Nat := [0,1,2,3,4,5,6,7]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 3 1) (by decide)
  have hh := finite_3_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_3_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 3 2)
      (laneIndex 3 2 (Projection.scatter [3,10,12,17,21,24,30,39] (BitVec.ofNat 8 n.val))) =
    laneReference 3 2 (Projection.scatter [3,10,12,17,21,24,30,39] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_3_2 (p : Word) :
    LineTable.reference (lanePos 3 2) (laneIndex 3 2 p) = laneReference 3 2 p := by
  let cells : List Nat := [3,10,12,17,21,24,30,39]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 3 2) (by decide)
  have hh := finite_3_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_3_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 3 3)
      (laneIndex 3 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 3 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_3_3 (p : Word) :
    LineTable.reference (lanePos 3 3) (laneIndex 3 3 p) = laneReference 3 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 3 3) (by decide)
  have hh := finite_3_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq01
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_2_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 2 0)
      (laneIndex 2 0 (Projection.scatter [2,10,18,26,34,42,50,58] (BitVec.ofNat 8 n.val))) =
    laneReference 2 0 (Projection.scatter [2,10,18,26,34,42,50,58] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_2_0 (p : Word) :
    LineTable.reference (lanePos 2 0) (laneIndex 2 0 p) = laneReference 2 0 p := by
  let cells : List Nat := [2,10,18,26,34,42,50,58]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 2 0) (by decide)
  have hh := finite_2_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_2_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 2 1)
      (laneIndex 2 1 (Projection.scatter [0,1,2,3,4,5,6,7] (BitVec.ofNat 8 n.val))) =
    laneReference 2 1 (Projection.scatter [0,1,2,3,4,5,6,7] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_2_1 (p : Word) :
    LineTable.reference (lanePos 2 1) (laneIndex 2 1 p) = laneReference 2 1 p := by
  let cells : List Nat := [0,1,2,3,4,5,6,7]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 2 1) (by decide)
  have hh := finite_2_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_2_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 2 2)
      (laneIndex 2 2 (Projection.scatter [2,9,11,16,20,29,38,47] (BitVec.ofNat 8 n.val))) =
    laneReference 2 2 (Projection.scatter [2,9,11,16,20,29,38,47] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_2_2 (p : Word) :
    LineTable.reference (lanePos 2 2) (laneIndex 2 2 p) = laneReference 2 2 p := by
  let cells : List Nat := [2,9,11,16,20,29,38,47]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 2 2) (by decide)
  have hh := finite_2_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_2_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 2 3)
      (laneIndex 2 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 2 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_2_3 (p : Word) :
    LineTable.reference (lanePos 2 3) (laneIndex 2 3 p) = laneReference 2 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 2 3) (by decide)
  have hh := finite_2_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

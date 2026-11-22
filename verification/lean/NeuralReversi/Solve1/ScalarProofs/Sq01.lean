import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq00
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_1_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 1 0)
      (laneIndex 1 0 (Projection.scatter [1,9,17,25,33,41,49,57] (BitVec.ofNat 8 n.val))) =
    laneReference 1 0 (Projection.scatter [1,9,17,25,33,41,49,57] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_1_0 (p : Word) :
    LineTable.reference (lanePos 1 0) (laneIndex 1 0 p) = laneReference 1 0 p := by
  let cells : List Nat := [1,9,17,25,33,41,49,57]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 1 0) (by decide)
  have hh := finite_1_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_1_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 1 1)
      (laneIndex 1 1 (Projection.scatter [0,1,2,3,4,5,6,7] (BitVec.ofNat 8 n.val))) =
    laneReference 1 1 (Projection.scatter [0,1,2,3,4,5,6,7] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_1_1 (p : Word) :
    LineTable.reference (lanePos 1 1) (laneIndex 1 1 p) = laneReference 1 1 p := by
  let cells : List Nat := [0,1,2,3,4,5,6,7]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 1 1) (by decide)
  have hh := finite_1_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_1_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 1 2)
      (laneIndex 1 2 (Projection.scatter [1,10,19,28,37,46,55] (BitVec.ofNat 8 n.val))) =
    laneReference 1 2 (Projection.scatter [1,10,19,28,37,46,55] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_1_2 (p : Word) :
    LineTable.reference (lanePos 1 2) (laneIndex 1 2 p) = laneReference 1 2 p := by
  let cells : List Nat := [1,10,19,28,37,46,55]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 1 2) (by decide)
  have hh := finite_1_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_1_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 1 3)
      (laneIndex 1 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 1 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_1_3 (p : Word) :
    LineTable.reference (lanePos 1 3) (laneIndex 1 3 p) = laneReference 1 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 1 3) (by decide)
  have hh := finite_1_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

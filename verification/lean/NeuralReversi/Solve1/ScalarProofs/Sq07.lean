import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq06
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_7_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 7 0)
      (laneIndex 7 0 (Projection.scatter [7,15,23,31,39,47,55,63] (BitVec.ofNat 8 n.val))) =
    laneReference 7 0 (Projection.scatter [7,15,23,31,39,47,55,63] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_7_0 (p : Word) :
    LineTable.reference (lanePos 7 0) (laneIndex 7 0 p) = laneReference 7 0 p := by
  let cells : List Nat := [7,15,23,31,39,47,55,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 7 0) (by decide)
  have hh := finite_7_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_7_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 7 1)
      (laneIndex 7 1 (Projection.scatter [0,1,2,3,4,5,6,7] (BitVec.ofNat 8 n.val))) =
    laneReference 7 1 (Projection.scatter [0,1,2,3,4,5,6,7] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_7_1 (p : Word) :
    LineTable.reference (lanePos 7 1) (laneIndex 7 1 p) = laneReference 7 1 p := by
  let cells : List Nat := [0,1,2,3,4,5,6,7]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 7 1) (by decide)
  have hh := finite_7_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_7_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 7 2)
      (laneIndex 7 2 (Projection.scatter [7,14,21,28,35,42,49,56] (BitVec.ofNat 8 n.val))) =
    laneReference 7 2 (Projection.scatter [7,14,21,28,35,42,49,56] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_7_2 (p : Word) :
    LineTable.reference (lanePos 7 2) (laneIndex 7 2 p) = laneReference 7 2 p := by
  let cells : List Nat := [7,14,21,28,35,42,49,56]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 7 2) (by decide)
  have hh := finite_7_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_7_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 7 3)
      (laneIndex 7 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 7 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_7_3 (p : Word) :
    LineTable.reference (lanePos 7 3) (laneIndex 7 3 p) = laneReference 7 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 7 3) (by decide)
  have hh := finite_7_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

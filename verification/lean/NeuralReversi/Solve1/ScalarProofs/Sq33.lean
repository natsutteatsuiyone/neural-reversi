import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq32
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_33_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 33 0)
      (laneIndex 33 0 (Projection.scatter [1,9,17,25,33,41,49,57] (BitVec.ofNat 8 n.val))) =
    laneReference 33 0 (Projection.scatter [1,9,17,25,33,41,49,57] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_33_0 (p : Word) :
    LineTable.reference (lanePos 33 0) (laneIndex 33 0 p) = laneReference 33 0 p := by
  let cells : List Nat := [1,9,17,25,33,41,49,57]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 33 0) (by decide)
  have hh := finite_33_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_33_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 33 1)
      (laneIndex 33 1 (Projection.scatter [32,33,34,35,36,37,38,39] (BitVec.ofNat 8 n.val))) =
    laneReference 33 1 (Projection.scatter [32,33,34,35,36,37,38,39] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_33_1 (p : Word) :
    LineTable.reference (lanePos 33 1) (laneIndex 33 1 p) = laneReference 33 1 p := by
  let cells : List Nat := [32,33,34,35,36,37,38,39]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 33 1) (by decide)
  have hh := finite_33_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_33_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 33 2)
      (laneIndex 33 2 (Projection.scatter [5,12,19,26,33,42,51,60] (BitVec.ofNat 8 n.val))) =
    laneReference 33 2 (Projection.scatter [5,12,19,26,33,42,51,60] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_33_2 (p : Word) :
    LineTable.reference (lanePos 33 2) (laneIndex 33 2 p) = laneReference 33 2 p := by
  let cells : List Nat := [5,12,19,26,33,42,51,60]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 33 2) (by decide)
  have hh := finite_33_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_33_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 33 3)
      (laneIndex 33 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 33 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_33_3 (p : Word) :
    LineTable.reference (lanePos 33 3) (laneIndex 33 3 p) = laneReference 33 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 33 3) (by decide)
  have hh := finite_33_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

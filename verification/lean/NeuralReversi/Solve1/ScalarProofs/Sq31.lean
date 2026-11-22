import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq30
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_31_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 31 0)
      (laneIndex 31 0 (Projection.scatter [7,15,23,31,39,47,55,63] (BitVec.ofNat 8 n.val))) =
    laneReference 31 0 (Projection.scatter [7,15,23,31,39,47,55,63] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_31_0 (p : Word) :
    LineTable.reference (lanePos 31 0) (laneIndex 31 0 p) = laneReference 31 0 p := by
  let cells : List Nat := [7,15,23,31,39,47,55,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 31 0) (by decide)
  have hh := finite_31_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_31_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 31 1)
      (laneIndex 31 1 (Projection.scatter [24,25,26,27,28,29,30,31] (BitVec.ofNat 8 n.val))) =
    laneReference 31 1 (Projection.scatter [24,25,26,27,28,29,30,31] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_31_1 (p : Word) :
    LineTable.reference (lanePos 31 1) (laneIndex 31 1 p) = laneReference 31 1 p := by
  let cells : List Nat := [24,25,26,27,28,29,30,31]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 31 1) (by decide)
  have hh := finite_31_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_31_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 31 2)
      (laneIndex 31 2 (Projection.scatter [4,13,22,31,38,45,52,59] (BitVec.ofNat 8 n.val))) =
    laneReference 31 2 (Projection.scatter [4,13,22,31,38,45,52,59] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_31_2 (p : Word) :
    LineTable.reference (lanePos 31 2) (laneIndex 31 2 p) = laneReference 31 2 p := by
  let cells : List Nat := [4,13,22,31,38,45,52,59]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 31 2) (by decide)
  have hh := finite_31_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_31_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 31 3)
      (laneIndex 31 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 31 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_31_3 (p : Word) :
    LineTable.reference (lanePos 31 3) (laneIndex 31 3 p) = laneReference 31 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 31 3) (by decide)
  have hh := finite_31_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

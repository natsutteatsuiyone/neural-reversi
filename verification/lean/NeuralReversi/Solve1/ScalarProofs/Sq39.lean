import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq38
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_39_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 39 0)
      (laneIndex 39 0 (Projection.scatter [7,15,23,31,39,47,55,63] (BitVec.ofNat 8 n.val))) =
    laneReference 39 0 (Projection.scatter [7,15,23,31,39,47,55,63] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_39_0 (p : Word) :
    LineTable.reference (lanePos 39 0) (laneIndex 39 0 p) = laneReference 39 0 p := by
  let cells : List Nat := [7,15,23,31,39,47,55,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 39 0) (by decide)
  have hh := finite_39_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_39_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 39 1)
      (laneIndex 39 1 (Projection.scatter [32,33,34,35,36,37,38,39] (BitVec.ofNat 8 n.val))) =
    laneReference 39 1 (Projection.scatter [32,33,34,35,36,37,38,39] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_39_1 (p : Word) :
    LineTable.reference (lanePos 39 1) (laneIndex 39 1 p) = laneReference 39 1 p := by
  let cells : List Nat := [32,33,34,35,36,37,38,39]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 39 1) (by decide)
  have hh := finite_39_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_39_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 39 2)
      (laneIndex 39 2 (Projection.scatter [3,12,21,30,39,46,53,60] (BitVec.ofNat 8 n.val))) =
    laneReference 39 2 (Projection.scatter [3,12,21,30,39,46,53,60] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_39_2 (p : Word) :
    LineTable.reference (lanePos 39 2) (laneIndex 39 2 p) = laneReference 39 2 p := by
  let cells : List Nat := [3,12,21,30,39,46,53,60]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 39 2) (by decide)
  have hh := finite_39_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_39_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 39 3)
      (laneIndex 39 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 39 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_39_3 (p : Word) :
    LineTable.reference (lanePos 39 3) (laneIndex 39 3 p) = laneReference 39 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 39 3) (by decide)
  have hh := finite_39_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

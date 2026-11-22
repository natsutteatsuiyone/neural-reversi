import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq18
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_19_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 19 0)
      (laneIndex 19 0 (Projection.scatter [3,11,19,27,35,43,51,59] (BitVec.ofNat 8 n.val))) =
    laneReference 19 0 (Projection.scatter [3,11,19,27,35,43,51,59] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_19_0 (p : Word) :
    LineTable.reference (lanePos 19 0) (laneIndex 19 0 p) = laneReference 19 0 p := by
  let cells : List Nat := [3,11,19,27,35,43,51,59]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 19 0) (by decide)
  have hh := finite_19_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_19_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 19 1)
      (laneIndex 19 1 (Projection.scatter [16,17,18,19,20,21,22,23] (BitVec.ofNat 8 n.val))) =
    laneReference 19 1 (Projection.scatter [16,17,18,19,20,21,22,23] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_19_1 (p : Word) :
    LineTable.reference (lanePos 19 1) (laneIndex 19 1 p) = laneReference 19 1 p := by
  let cells : List Nat := [16,17,18,19,20,21,22,23]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 19 1) (by decide)
  have hh := finite_19_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_19_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 19 2)
      (laneIndex 19 2 (Projection.scatter [5,12,19,26,33,40] (BitVec.ofNat 8 n.val))) =
    laneReference 19 2 (Projection.scatter [5,12,19,26,33,40] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_19_2 (p : Word) :
    LineTable.reference (lanePos 19 2) (laneIndex 19 2 p) = laneReference 19 2 p := by
  let cells : List Nat := [5,12,19,26,33,40]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 19 2) (by decide)
  have hh := finite_19_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_19_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 19 3)
      (laneIndex 19 3 (Projection.scatter [1,10,19,28,37,46,55] (BitVec.ofNat 8 n.val))) =
    laneReference 19 3 (Projection.scatter [1,10,19,28,37,46,55] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_19_3 (p : Word) :
    LineTable.reference (lanePos 19 3) (laneIndex 19 3 p) = laneReference 19 3 p := by
  let cells : List Nat := [1,10,19,28,37,46,55]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 19 3) (by decide)
  have hh := finite_19_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

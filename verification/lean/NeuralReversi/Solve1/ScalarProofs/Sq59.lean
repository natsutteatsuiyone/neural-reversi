import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq58
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_59_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 59 0)
      (laneIndex 59 0 (Projection.scatter [3,11,19,27,35,43,51,59] (BitVec.ofNat 8 n.val))) =
    laneReference 59 0 (Projection.scatter [3,11,19,27,35,43,51,59] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_59_0 (p : Word) :
    LineTable.reference (lanePos 59 0) (laneIndex 59 0 p) = laneReference 59 0 p := by
  let cells : List Nat := [3,11,19,27,35,43,51,59]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 59 0) (by decide)
  have hh := finite_59_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_59_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 59 1)
      (laneIndex 59 1 (Projection.scatter [56,57,58,59,60,61,62,63] (BitVec.ofNat 8 n.val))) =
    laneReference 59 1 (Projection.scatter [56,57,58,59,60,61,62,63] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_59_1 (p : Word) :
    LineTable.reference (lanePos 59 1) (laneIndex 59 1 p) = laneReference 59 1 p := by
  let cells : List Nat := [56,57,58,59,60,61,62,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 59 1) (by decide)
  have hh := finite_59_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_59_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 59 2)
      (laneIndex 59 2 (Projection.scatter [31,32,38,41,45,50,52,59] (BitVec.ofNat 8 n.val))) =
    laneReference 59 2 (Projection.scatter [31,32,38,41,45,50,52,59] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_59_2 (p : Word) :
    LineTable.reference (lanePos 59 2) (laneIndex 59 2 p) = laneReference 59 2 p := by
  let cells : List Nat := [31,32,38,41,45,50,52,59]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 59 2) (by decide)
  have hh := finite_59_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_59_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 59 3)
      (laneIndex 59 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 59 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_59_3 (p : Word) :
    LineTable.reference (lanePos 59 3) (laneIndex 59 3 p) = laneReference 59 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 59 3) (by decide)
  have hh := finite_59_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

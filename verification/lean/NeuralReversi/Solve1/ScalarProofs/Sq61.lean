import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq60
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_61_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 61 0)
      (laneIndex 61 0 (Projection.scatter [5,13,21,29,37,45,53,61] (BitVec.ofNat 8 n.val))) =
    laneReference 61 0 (Projection.scatter [5,13,21,29,37,45,53,61] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_61_0 (p : Word) :
    LineTable.reference (lanePos 61 0) (laneIndex 61 0 p) = laneReference 61 0 p := by
  let cells : List Nat := [5,13,21,29,37,45,53,61]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 61 0) (by decide)
  have hh := finite_61_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_61_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 61 1)
      (laneIndex 61 1 (Projection.scatter [56,57,58,59,60,61,62,63] (BitVec.ofNat 8 n.val))) =
    laneReference 61 1 (Projection.scatter [56,57,58,59,60,61,62,63] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_61_1 (p : Word) :
    LineTable.reference (lanePos 61 1) (laneIndex 61 1 p) = laneReference 61 1 p := by
  let cells : List Nat := [56,57,58,59,60,61,62,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 61 1) (by decide)
  have hh := finite_61_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_61_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 61 2)
      (laneIndex 61 2 (Projection.scatter [16,25,34,43,47,52,54,61] (BitVec.ofNat 8 n.val))) =
    laneReference 61 2 (Projection.scatter [16,25,34,43,47,52,54,61] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_61_2 (p : Word) :
    LineTable.reference (lanePos 61 2) (laneIndex 61 2 p) = laneReference 61 2 p := by
  let cells : List Nat := [16,25,34,43,47,52,54,61]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 61 2) (by decide)
  have hh := finite_61_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_61_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 61 3)
      (laneIndex 61 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 61 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_61_3 (p : Word) :
    LineTable.reference (lanePos 61 3) (laneIndex 61 3 p) = laneReference 61 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 61 3) (by decide)
  have hh := finite_61_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

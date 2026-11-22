import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq57
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_58_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 58 0)
      (laneIndex 58 0 (Projection.scatter [2,10,18,26,34,42,50,58] (BitVec.ofNat 8 n.val))) =
    laneReference 58 0 (Projection.scatter [2,10,18,26,34,42,50,58] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_58_0 (p : Word) :
    LineTable.reference (lanePos 58 0) (laneIndex 58 0 p) = laneReference 58 0 p := by
  let cells : List Nat := [2,10,18,26,34,42,50,58]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 58 0) (by decide)
  have hh := finite_58_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_58_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 58 1)
      (laneIndex 58 1 (Projection.scatter [56,57,58,59,60,61,62,63] (BitVec.ofNat 8 n.val))) =
    laneReference 58 1 (Projection.scatter [56,57,58,59,60,61,62,63] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_58_1 (p : Word) :
    LineTable.reference (lanePos 58 1) (laneIndex 58 1 p) = laneReference 58 1 p := by
  let cells : List Nat := [56,57,58,59,60,61,62,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 58 1) (by decide)
  have hh := finite_58_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_58_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 58 2)
      (laneIndex 58 2 (Projection.scatter [23,30,37,40,44,49,51,58] (BitVec.ofNat 8 n.val))) =
    laneReference 58 2 (Projection.scatter [23,30,37,40,44,49,51,58] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_58_2 (p : Word) :
    LineTable.reference (lanePos 58 2) (laneIndex 58 2 p) = laneReference 58 2 p := by
  let cells : List Nat := [23,30,37,40,44,49,51,58]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 58 2) (by decide)
  have hh := finite_58_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_58_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 58 3)
      (laneIndex 58 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 58 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_58_3 (p : Word) :
    LineTable.reference (lanePos 58 3) (laneIndex 58 3 p) = laneReference 58 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 58 3) (by decide)
  have hh := finite_58_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq07
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_8_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 8 0)
      (laneIndex 8 0 (Projection.scatter [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 n.val))) =
    laneReference 8 0 (Projection.scatter [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_8_0 (p : Word) :
    LineTable.reference (lanePos 8 0) (laneIndex 8 0 p) = laneReference 8 0 p := by
  let cells : List Nat := [0,8,16,24,32,40,48,56]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 8 0) (by decide)
  have hh := finite_8_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_8_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 8 1)
      (laneIndex 8 1 (Projection.scatter [8,9,10,11,12,13,14,15] (BitVec.ofNat 8 n.val))) =
    laneReference 8 1 (Projection.scatter [8,9,10,11,12,13,14,15] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_8_1 (p : Word) :
    LineTable.reference (lanePos 8 1) (laneIndex 8 1 p) = laneReference 8 1 p := by
  let cells : List Nat := [8,9,10,11,12,13,14,15]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 8 1) (by decide)
  have hh := finite_8_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_8_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 8 2)
      (laneIndex 8 2 (Projection.scatter [8,17,26,35,44,53,62] (BitVec.ofNat 8 n.val))) =
    laneReference 8 2 (Projection.scatter [8,17,26,35,44,53,62] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_8_2 (p : Word) :
    LineTable.reference (lanePos 8 2) (laneIndex 8 2 p) = laneReference 8 2 p := by
  let cells : List Nat := [8,17,26,35,44,53,62]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 8 2) (by decide)
  have hh := finite_8_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_8_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 8 3)
      (laneIndex 8 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 8 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_8_3 (p : Word) :
    LineTable.reference (lanePos 8 3) (laneIndex 8 3 p) = laneReference 8 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 8 3) (by decide)
  have hh := finite_8_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

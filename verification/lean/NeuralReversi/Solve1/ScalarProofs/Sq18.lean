import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq17
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_18_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 18 0)
      (laneIndex 18 0 (Projection.scatter [2,10,18,26,34,42,50,58] (BitVec.ofNat 8 n.val))) =
    laneReference 18 0 (Projection.scatter [2,10,18,26,34,42,50,58] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_18_0 (p : Word) :
    LineTable.reference (lanePos 18 0) (laneIndex 18 0 p) = laneReference 18 0 p := by
  let cells : List Nat := [2,10,18,26,34,42,50,58]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 18 0) (by decide)
  have hh := finite_18_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_18_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 18 1)
      (laneIndex 18 1 (Projection.scatter [16,17,18,19,20,21,22,23] (BitVec.ofNat 8 n.val))) =
    laneReference 18 1 (Projection.scatter [16,17,18,19,20,21,22,23] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_18_1 (p : Word) :
    LineTable.reference (lanePos 18 1) (laneIndex 18 1 p) = laneReference 18 1 p := by
  let cells : List Nat := [16,17,18,19,20,21,22,23]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 18 1) (by decide)
  have hh := finite_18_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_18_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 18 2)
      (laneIndex 18 2 (Projection.scatter [4,11,18,25,32] (BitVec.ofNat 8 n.val))) =
    laneReference 18 2 (Projection.scatter [4,11,18,25,32] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_18_2 (p : Word) :
    LineTable.reference (lanePos 18 2) (laneIndex 18 2 p) = laneReference 18 2 p := by
  let cells : List Nat := [4,11,18,25,32]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 18 2) (by decide)
  have hh := finite_18_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_18_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 18 3)
      (laneIndex 18 3 (Projection.scatter [0,9,18,27,36,45,54,63] (BitVec.ofNat 8 n.val))) =
    laneReference 18 3 (Projection.scatter [0,9,18,27,36,45,54,63] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_18_3 (p : Word) :
    LineTable.reference (lanePos 18 3) (laneIndex 18 3 p) = laneReference 18 3 p := by
  let cells : List Nat := [0,9,18,27,36,45,54,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 18 3) (by decide)
  have hh := finite_18_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

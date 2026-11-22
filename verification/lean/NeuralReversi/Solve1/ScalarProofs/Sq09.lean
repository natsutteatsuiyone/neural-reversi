import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq08
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_9_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 9 0)
      (laneIndex 9 0 (Projection.scatter [1,9,17,25,33,41,49,57] (BitVec.ofNat 8 n.val))) =
    laneReference 9 0 (Projection.scatter [1,9,17,25,33,41,49,57] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_9_0 (p : Word) :
    LineTable.reference (lanePos 9 0) (laneIndex 9 0 p) = laneReference 9 0 p := by
  let cells : List Nat := [1,9,17,25,33,41,49,57]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 9 0) (by decide)
  have hh := finite_9_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_9_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 9 1)
      (laneIndex 9 1 (Projection.scatter [8,9,10,11,12,13,14,15] (BitVec.ofNat 8 n.val))) =
    laneReference 9 1 (Projection.scatter [8,9,10,11,12,13,14,15] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_9_1 (p : Word) :
    LineTable.reference (lanePos 9 1) (laneIndex 9 1 p) = laneReference 9 1 p := by
  let cells : List Nat := [8,9,10,11,12,13,14,15]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 9 1) (by decide)
  have hh := finite_9_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_9_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 9 2)
      (laneIndex 9 2 (Projection.scatter [0,9,18,27,36,45,54,63] (BitVec.ofNat 8 n.val))) =
    laneReference 9 2 (Projection.scatter [0,9,18,27,36,45,54,63] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_9_2 (p : Word) :
    LineTable.reference (lanePos 9 2) (laneIndex 9 2 p) = laneReference 9 2 p := by
  let cells : List Nat := [0,9,18,27,36,45,54,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 9 2) (by decide)
  have hh := finite_9_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_9_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 9 3)
      (laneIndex 9 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 9 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_9_3 (p : Word) :
    LineTable.reference (lanePos 9 3) (laneIndex 9 3 p) = laneReference 9 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 9 3) (by decide)
  have hh := finite_9_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

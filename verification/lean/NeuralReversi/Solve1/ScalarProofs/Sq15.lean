import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq14
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_15_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 15 0)
      (laneIndex 15 0 (Projection.scatter [7,15,23,31,39,47,55,63] (BitVec.ofNat 8 n.val))) =
    laneReference 15 0 (Projection.scatter [7,15,23,31,39,47,55,63] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_15_0 (p : Word) :
    LineTable.reference (lanePos 15 0) (laneIndex 15 0 p) = laneReference 15 0 p := by
  let cells : List Nat := [7,15,23,31,39,47,55,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 15 0) (by decide)
  have hh := finite_15_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_15_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 15 1)
      (laneIndex 15 1 (Projection.scatter [8,9,10,11,12,13,14,15] (BitVec.ofNat 8 n.val))) =
    laneReference 15 1 (Projection.scatter [8,9,10,11,12,13,14,15] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_15_1 (p : Word) :
    LineTable.reference (lanePos 15 1) (laneIndex 15 1 p) = laneReference 15 1 p := by
  let cells : List Nat := [8,9,10,11,12,13,14,15]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 15 1) (by decide)
  have hh := finite_15_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_15_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 15 2)
      (laneIndex 15 2 (Projection.scatter [15,22,29,36,43,50,57] (BitVec.ofNat 8 n.val))) =
    laneReference 15 2 (Projection.scatter [15,22,29,36,43,50,57] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_15_2 (p : Word) :
    LineTable.reference (lanePos 15 2) (laneIndex 15 2 p) = laneReference 15 2 p := by
  let cells : List Nat := [15,22,29,36,43,50,57]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 15 2) (by decide)
  have hh := finite_15_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_15_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 15 3)
      (laneIndex 15 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 15 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_15_3 (p : Word) :
    LineTable.reference (lanePos 15 3) (laneIndex 15 3 p) = laneReference 15 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 15 3) (by decide)
  have hh := finite_15_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

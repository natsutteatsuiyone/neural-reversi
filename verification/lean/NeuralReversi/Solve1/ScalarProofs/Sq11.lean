import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq10
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_11_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 11 0)
      (laneIndex 11 0 (Projection.scatter [3,11,19,27,35,43,51,59] (BitVec.ofNat 8 n.val))) =
    laneReference 11 0 (Projection.scatter [3,11,19,27,35,43,51,59] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_11_0 (p : Word) :
    LineTable.reference (lanePos 11 0) (laneIndex 11 0 p) = laneReference 11 0 p := by
  let cells : List Nat := [3,11,19,27,35,43,51,59]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 11 0) (by decide)
  have hh := finite_11_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_11_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 11 1)
      (laneIndex 11 1 (Projection.scatter [8,9,10,11,12,13,14,15] (BitVec.ofNat 8 n.val))) =
    laneReference 11 1 (Projection.scatter [8,9,10,11,12,13,14,15] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_11_1 (p : Word) :
    LineTable.reference (lanePos 11 1) (laneIndex 11 1 p) = laneReference 11 1 p := by
  let cells : List Nat := [8,9,10,11,12,13,14,15]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 11 1) (by decide)
  have hh := finite_11_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_11_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 11 2)
      (laneIndex 11 2 (Projection.scatter [11,18,20,25,29,32,38,47] (BitVec.ofNat 8 n.val))) =
    laneReference 11 2 (Projection.scatter [11,18,20,25,29,32,38,47] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_11_2 (p : Word) :
    LineTable.reference (lanePos 11 2) (laneIndex 11 2 p) = laneReference 11 2 p := by
  let cells : List Nat := [11,18,20,25,29,32,38,47]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 11 2) (by decide)
  have hh := finite_11_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_11_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 11 3)
      (laneIndex 11 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 11 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_11_3 (p : Word) :
    LineTable.reference (lanePos 11 3) (laneIndex 11 3 p) = laneReference 11 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 11 3) (by decide)
  have hh := finite_11_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

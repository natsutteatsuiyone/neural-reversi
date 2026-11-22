import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq21
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_22_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 22 0)
      (laneIndex 22 0 (Projection.scatter [6,14,22,30,38,46,54,62] (BitVec.ofNat 8 n.val))) =
    laneReference 22 0 (Projection.scatter [6,14,22,30,38,46,54,62] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_22_0 (p : Word) :
    LineTable.reference (lanePos 22 0) (laneIndex 22 0 p) = laneReference 22 0 p := by
  let cells : List Nat := [6,14,22,30,38,46,54,62]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 22 0) (by decide)
  have hh := finite_22_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_22_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 22 1)
      (laneIndex 22 1 (Projection.scatter [16,17,18,19,20,21,22,23] (BitVec.ofNat 8 n.val))) =
    laneReference 22 1 (Projection.scatter [16,17,18,19,20,21,22,23] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_22_1 (p : Word) :
    LineTable.reference (lanePos 22 1) (laneIndex 22 1 p) = laneReference 22 1 p := by
  let cells : List Nat := [16,17,18,19,20,21,22,23]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 22 1) (by decide)
  have hh := finite_22_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_22_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 22 2)
      (laneIndex 22 2 (Projection.scatter [4,13,22,29,36,43,50,57] (BitVec.ofNat 8 n.val))) =
    laneReference 22 2 (Projection.scatter [4,13,22,29,36,43,50,57] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_22_2 (p : Word) :
    LineTable.reference (lanePos 22 2) (laneIndex 22 2 p) = laneReference 22 2 p := by
  let cells : List Nat := [4,13,22,29,36,43,50,57]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 22 2) (by decide)
  have hh := finite_22_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_22_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 22 3)
      (laneIndex 22 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 22 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_22_3 (p : Word) :
    LineTable.reference (lanePos 22 3) (laneIndex 22 3 p) = laneReference 22 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 22 3) (by decide)
  have hh := finite_22_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

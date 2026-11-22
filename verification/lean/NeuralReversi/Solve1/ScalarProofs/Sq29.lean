import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq28
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_29_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 29 0)
      (laneIndex 29 0 (Projection.scatter [5,13,21,29,37,45,53,61] (BitVec.ofNat 8 n.val))) =
    laneReference 29 0 (Projection.scatter [5,13,21,29,37,45,53,61] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_29_0 (p : Word) :
    LineTable.reference (lanePos 29 0) (laneIndex 29 0 p) = laneReference 29 0 p := by
  let cells : List Nat := [5,13,21,29,37,45,53,61]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 29 0) (by decide)
  have hh := finite_29_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_29_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 29 1)
      (laneIndex 29 1 (Projection.scatter [24,25,26,27,28,29,30,31] (BitVec.ofNat 8 n.val))) =
    laneReference 29 1 (Projection.scatter [24,25,26,27,28,29,30,31] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_29_1 (p : Word) :
    LineTable.reference (lanePos 29 1) (laneIndex 29 1 p) = laneReference 29 1 p := by
  let cells : List Nat := [24,25,26,27,28,29,30,31]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 29 1) (by decide)
  have hh := finite_29_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_29_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 29 2)
      (laneIndex 29 2 (Projection.scatter [15,22,29,36,43,50,57] (BitVec.ofNat 8 n.val))) =
    laneReference 29 2 (Projection.scatter [15,22,29,36,43,50,57] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_29_2 (p : Word) :
    LineTable.reference (lanePos 29 2) (laneIndex 29 2 p) = laneReference 29 2 p := by
  let cells : List Nat := [15,22,29,36,43,50,57]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 29 2) (by decide)
  have hh := finite_29_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_29_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 29 3)
      (laneIndex 29 3 (Projection.scatter [2,11,20,29,38,47] (BitVec.ofNat 8 n.val))) =
    laneReference 29 3 (Projection.scatter [2,11,20,29,38,47] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_29_3 (p : Word) :
    LineTable.reference (lanePos 29 3) (laneIndex 29 3 p) = laneReference 29 3 p := by
  let cells : List Nat := [2,11,20,29,38,47]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 29 3) (by decide)
  have hh := finite_29_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

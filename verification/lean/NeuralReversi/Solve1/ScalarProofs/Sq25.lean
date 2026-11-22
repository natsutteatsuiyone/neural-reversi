import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq24
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_25_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 25 0)
      (laneIndex 25 0 (Projection.scatter [1,9,17,25,33,41,49,57] (BitVec.ofNat 8 n.val))) =
    laneReference 25 0 (Projection.scatter [1,9,17,25,33,41,49,57] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_25_0 (p : Word) :
    LineTable.reference (lanePos 25 0) (laneIndex 25 0 p) = laneReference 25 0 p := by
  let cells : List Nat := [1,9,17,25,33,41,49,57]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 25 0) (by decide)
  have hh := finite_25_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_25_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 25 1)
      (laneIndex 25 1 (Projection.scatter [24,25,26,27,28,29,30,31] (BitVec.ofNat 8 n.val))) =
    laneReference 25 1 (Projection.scatter [24,25,26,27,28,29,30,31] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_25_1 (p : Word) :
    LineTable.reference (lanePos 25 1) (laneIndex 25 1 p) = laneReference 25 1 p := by
  let cells : List Nat := [24,25,26,27,28,29,30,31]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 25 1) (by decide)
  have hh := finite_25_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_25_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 25 2)
      (laneIndex 25 2 (Projection.scatter [4,11,18,25,34,43,52,61] (BitVec.ofNat 8 n.val))) =
    laneReference 25 2 (Projection.scatter [4,11,18,25,34,43,52,61] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_25_2 (p : Word) :
    LineTable.reference (lanePos 25 2) (laneIndex 25 2 p) = laneReference 25 2 p := by
  let cells : List Nat := [4,11,18,25,34,43,52,61]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 25 2) (by decide)
  have hh := finite_25_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_25_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 25 3)
      (laneIndex 25 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 25 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_25_3 (p : Word) :
    LineTable.reference (lanePos 25 3) (laneIndex 25 3 p) = laneReference 25 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 25 3) (by decide)
  have hh := finite_25_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

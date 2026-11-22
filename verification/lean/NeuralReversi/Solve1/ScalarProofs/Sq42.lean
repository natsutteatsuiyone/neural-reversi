import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq41
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_42_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 42 0)
      (laneIndex 42 0 (Projection.scatter [2,10,18,26,34,42,50,58] (BitVec.ofNat 8 n.val))) =
    laneReference 42 0 (Projection.scatter [2,10,18,26,34,42,50,58] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_42_0 (p : Word) :
    LineTable.reference (lanePos 42 0) (laneIndex 42 0 p) = laneReference 42 0 p := by
  let cells : List Nat := [2,10,18,26,34,42,50,58]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 42 0) (by decide)
  have hh := finite_42_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_42_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 42 1)
      (laneIndex 42 1 (Projection.scatter [40,41,42,43,44,45,46,47] (BitVec.ofNat 8 n.val))) =
    laneReference 42 1 (Projection.scatter [40,41,42,43,44,45,46,47] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_42_1 (p : Word) :
    LineTable.reference (lanePos 42 1) (laneIndex 42 1 p) = laneReference 42 1 p := by
  let cells : List Nat := [40,41,42,43,44,45,46,47]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 42 1) (by decide)
  have hh := finite_42_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_42_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 42 2)
      (laneIndex 42 2 (Projection.scatter [7,14,21,28,35,42,49,56] (BitVec.ofNat 8 n.val))) =
    laneReference 42 2 (Projection.scatter [7,14,21,28,35,42,49,56] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_42_2 (p : Word) :
    LineTable.reference (lanePos 42 2) (laneIndex 42 2 p) = laneReference 42 2 p := by
  let cells : List Nat := [7,14,21,28,35,42,49,56]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 42 2) (by decide)
  have hh := finite_42_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_42_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 42 3)
      (laneIndex 42 3 (Projection.scatter [24,33,42,51,60] (BitVec.ofNat 8 n.val))) =
    laneReference 42 3 (Projection.scatter [24,33,42,51,60] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_42_3 (p : Word) :
    LineTable.reference (lanePos 42 3) (laneIndex 42 3 p) = laneReference 42 3 p := by
  let cells : List Nat := [24,33,42,51,60]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 42 3) (by decide)
  have hh := finite_42_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

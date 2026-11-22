import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq45
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_46_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 46 0)
      (laneIndex 46 0 (Projection.scatter [6,14,22,30,38,46,54,62] (BitVec.ofNat 8 n.val))) =
    laneReference 46 0 (Projection.scatter [6,14,22,30,38,46,54,62] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_46_0 (p : Word) :
    LineTable.reference (lanePos 46 0) (laneIndex 46 0 p) = laneReference 46 0 p := by
  let cells : List Nat := [6,14,22,30,38,46,54,62]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 46 0) (by decide)
  have hh := finite_46_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_46_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 46 1)
      (laneIndex 46 1 (Projection.scatter [40,41,42,43,44,45,46,47] (BitVec.ofNat 8 n.val))) =
    laneReference 46 1 (Projection.scatter [40,41,42,43,44,45,46,47] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_46_1 (p : Word) :
    LineTable.reference (lanePos 46 1) (laneIndex 46 1 p) = laneReference 46 1 p := by
  let cells : List Nat := [40,41,42,43,44,45,46,47]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 46 1) (by decide)
  have hh := finite_46_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_46_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 46 2)
      (laneIndex 46 2 (Projection.scatter [1,10,19,28,37,46,53,60] (BitVec.ofNat 8 n.val))) =
    laneReference 46 2 (Projection.scatter [1,10,19,28,37,46,53,60] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_46_2 (p : Word) :
    LineTable.reference (lanePos 46 2) (laneIndex 46 2 p) = laneReference 46 2 p := by
  let cells : List Nat := [1,10,19,28,37,46,53,60]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 46 2) (by decide)
  have hh := finite_46_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_46_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 46 3)
      (laneIndex 46 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 46 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_46_3 (p : Word) :
    LineTable.reference (lanePos 46 3) (laneIndex 46 3 p) = laneReference 46 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 46 3) (by decide)
  have hh := finite_46_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

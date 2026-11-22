import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq25
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_26_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 26 0)
      (laneIndex 26 0 (Projection.scatter [2,10,18,26,34,42,50,58] (BitVec.ofNat 8 n.val))) =
    laneReference 26 0 (Projection.scatter [2,10,18,26,34,42,50,58] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_26_0 (p : Word) :
    LineTable.reference (lanePos 26 0) (laneIndex 26 0 p) = laneReference 26 0 p := by
  let cells : List Nat := [2,10,18,26,34,42,50,58]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 26 0) (by decide)
  have hh := finite_26_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_26_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 26 1)
      (laneIndex 26 1 (Projection.scatter [24,25,26,27,28,29,30,31] (BitVec.ofNat 8 n.val))) =
    laneReference 26 1 (Projection.scatter [24,25,26,27,28,29,30,31] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_26_1 (p : Word) :
    LineTable.reference (lanePos 26 1) (laneIndex 26 1 p) = laneReference 26 1 p := by
  let cells : List Nat := [24,25,26,27,28,29,30,31]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 26 1) (by decide)
  have hh := finite_26_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_26_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 26 2)
      (laneIndex 26 2 (Projection.scatter [5,12,19,26,33,40] (BitVec.ofNat 8 n.val))) =
    laneReference 26 2 (Projection.scatter [5,12,19,26,33,40] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_26_2 (p : Word) :
    LineTable.reference (lanePos 26 2) (laneIndex 26 2 p) = laneReference 26 2 p := by
  let cells : List Nat := [5,12,19,26,33,40]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 26 2) (by decide)
  have hh := finite_26_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_26_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 26 3)
      (laneIndex 26 3 (Projection.scatter [8,17,26,35,44,53,62] (BitVec.ofNat 8 n.val))) =
    laneReference 26 3 (Projection.scatter [8,17,26,35,44,53,62] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_26_3 (p : Word) :
    LineTable.reference (lanePos 26 3) (laneIndex 26 3 p) = laneReference 26 3 p := by
  let cells : List Nat := [8,17,26,35,44,53,62]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 26 3) (by decide)
  have hh := finite_26_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

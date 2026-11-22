import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq52
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_53_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 53 0)
      (laneIndex 53 0 (Projection.scatter [5,13,21,29,37,45,53,61] (BitVec.ofNat 8 n.val))) =
    laneReference 53 0 (Projection.scatter [5,13,21,29,37,45,53,61] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_53_0 (p : Word) :
    LineTable.reference (lanePos 53 0) (laneIndex 53 0 p) = laneReference 53 0 p := by
  let cells : List Nat := [5,13,21,29,37,45,53,61]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 53 0) (by decide)
  have hh := finite_53_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_53_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 53 1)
      (laneIndex 53 1 (Projection.scatter [48,49,50,51,52,53,54,55] (BitVec.ofNat 8 n.val))) =
    laneReference 53 1 (Projection.scatter [48,49,50,51,52,53,54,55] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_53_1 (p : Word) :
    LineTable.reference (lanePos 53 1) (laneIndex 53 1 p) = laneReference 53 1 p := by
  let cells : List Nat := [48,49,50,51,52,53,54,55]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 53 1) (by decide)
  have hh := finite_53_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_53_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 53 2)
      (laneIndex 53 2 (Projection.scatter [8,17,26,35,39,44,46,53] (BitVec.ofNat 8 n.val))) =
    laneReference 53 2 (Projection.scatter [8,17,26,35,39,44,46,53] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_53_2 (p : Word) :
    LineTable.reference (lanePos 53 2) (laneIndex 53 2 p) = laneReference 53 2 p := by
  let cells : List Nat := [8,17,26,35,39,44,46,53]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 53 2) (by decide)
  have hh := finite_53_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_53_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 53 3)
      (laneIndex 53 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 53 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_53_3 (p : Word) :
    LineTable.reference (lanePos 53 3) (laneIndex 53 3 p) = laneReference 53 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 53 3) (by decide)
  have hh := finite_53_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

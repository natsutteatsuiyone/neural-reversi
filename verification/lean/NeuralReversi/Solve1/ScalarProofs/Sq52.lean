import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq51
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_52_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 52 0)
      (laneIndex 52 0 (Projection.scatter [4,12,20,28,36,44,52,60] (BitVec.ofNat 8 n.val))) =
    laneReference 52 0 (Projection.scatter [4,12,20,28,36,44,52,60] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_52_0 (p : Word) :
    LineTable.reference (lanePos 52 0) (laneIndex 52 0 p) = laneReference 52 0 p := by
  let cells : List Nat := [4,12,20,28,36,44,52,60]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 52 0) (by decide)
  have hh := finite_52_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_52_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 52 1)
      (laneIndex 52 1 (Projection.scatter [48,49,50,51,52,53,54,55] (BitVec.ofNat 8 n.val))) =
    laneReference 52 1 (Projection.scatter [48,49,50,51,52,53,54,55] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_52_1 (p : Word) :
    LineTable.reference (lanePos 52 1) (laneIndex 52 1 p) = laneReference 52 1 p := by
  let cells : List Nat := [48,49,50,51,52,53,54,55]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 52 1) (by decide)
  have hh := finite_52_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_52_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 52 2)
      (laneIndex 52 2 (Projection.scatter [16,25,31,34,38,43,45,52] (BitVec.ofNat 8 n.val))) =
    laneReference 52 2 (Projection.scatter [16,25,31,34,38,43,45,52] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_52_2 (p : Word) :
    LineTable.reference (lanePos 52 2) (laneIndex 52 2 p) = laneReference 52 2 p := by
  let cells : List Nat := [16,25,31,34,38,43,45,52]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 52 2) (by decide)
  have hh := finite_52_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_52_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 52 3)
      (laneIndex 52 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 52 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_52_3 (p : Word) :
    LineTable.reference (lanePos 52 3) (laneIndex 52 3 p) = laneReference 52 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 52 3) (by decide)
  have hh := finite_52_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

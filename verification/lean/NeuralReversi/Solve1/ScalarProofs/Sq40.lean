import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq39
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_40_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 40 0)
      (laneIndex 40 0 (Projection.scatter [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 n.val))) =
    laneReference 40 0 (Projection.scatter [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_40_0 (p : Word) :
    LineTable.reference (lanePos 40 0) (laneIndex 40 0 p) = laneReference 40 0 p := by
  let cells : List Nat := [0,8,16,24,32,40,48,56]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 40 0) (by decide)
  have hh := finite_40_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_40_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 40 1)
      (laneIndex 40 1 (Projection.scatter [40,41,42,43,44,45,46,47] (BitVec.ofNat 8 n.val))) =
    laneReference 40 1 (Projection.scatter [40,41,42,43,44,45,46,47] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_40_1 (p : Word) :
    LineTable.reference (lanePos 40 1) (laneIndex 40 1 p) = laneReference 40 1 p := by
  let cells : List Nat := [40,41,42,43,44,45,46,47]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 40 1) (by decide)
  have hh := finite_40_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_40_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 40 2)
      (laneIndex 40 2 (Projection.scatter [5,12,19,26,33,40,49,58] (BitVec.ofNat 8 n.val))) =
    laneReference 40 2 (Projection.scatter [5,12,19,26,33,40,49,58] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_40_2 (p : Word) :
    LineTable.reference (lanePos 40 2) (laneIndex 40 2 p) = laneReference 40 2 p := by
  let cells : List Nat := [5,12,19,26,33,40,49,58]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 40 2) (by decide)
  have hh := finite_40_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_40_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 40 3)
      (laneIndex 40 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 40 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_40_3 (p : Word) :
    LineTable.reference (lanePos 40 3) (laneIndex 40 3 p) = laneReference 40 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 40 3) (by decide)
  have hh := finite_40_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

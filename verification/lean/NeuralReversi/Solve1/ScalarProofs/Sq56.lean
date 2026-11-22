import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq55
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_56_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 56 0)
      (laneIndex 56 0 (Projection.scatter [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 n.val))) =
    laneReference 56 0 (Projection.scatter [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_56_0 (p : Word) :
    LineTable.reference (lanePos 56 0) (laneIndex 56 0 p) = laneReference 56 0 p := by
  let cells : List Nat := [0,8,16,24,32,40,48,56]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 56 0) (by decide)
  have hh := finite_56_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_56_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 56 1)
      (laneIndex 56 1 (Projection.scatter [56,57,58,59,60,61,62,63] (BitVec.ofNat 8 n.val))) =
    laneReference 56 1 (Projection.scatter [56,57,58,59,60,61,62,63] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_56_1 (p : Word) :
    LineTable.reference (lanePos 56 1) (laneIndex 56 1 p) = laneReference 56 1 p := by
  let cells : List Nat := [56,57,58,59,60,61,62,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 56 1) (by decide)
  have hh := finite_56_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_56_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 56 2)
      (laneIndex 56 2 (Projection.scatter [7,14,21,28,35,42,49,56] (BitVec.ofNat 8 n.val))) =
    laneReference 56 2 (Projection.scatter [7,14,21,28,35,42,49,56] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_56_2 (p : Word) :
    LineTable.reference (lanePos 56 2) (laneIndex 56 2 p) = laneReference 56 2 p := by
  let cells : List Nat := [7,14,21,28,35,42,49,56]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 56 2) (by decide)
  have hh := finite_56_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_56_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 56 3)
      (laneIndex 56 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 56 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_56_3 (p : Word) :
    LineTable.reference (lanePos 56 3) (laneIndex 56 3 p) = laneReference 56 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 56 3) (by decide)
  have hh := finite_56_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

import NeuralReversi.Solve1.ScalarModel
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_0_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 0 0)
      (laneIndex 0 0 (Projection.scatter [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 n.val))) =
    laneReference 0 0 (Projection.scatter [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_0_0 (p : Word) :
    LineTable.reference (lanePos 0 0) (laneIndex 0 0 p) = laneReference 0 0 p := by
  let cells : List Nat := [0,8,16,24,32,40,48,56]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 0 0) (by decide)
  have hh := finite_0_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_0_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 0 1)
      (laneIndex 0 1 (Projection.scatter [0,1,2,3,4,5,6,7] (BitVec.ofNat 8 n.val))) =
    laneReference 0 1 (Projection.scatter [0,1,2,3,4,5,6,7] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_0_1 (p : Word) :
    LineTable.reference (lanePos 0 1) (laneIndex 0 1 p) = laneReference 0 1 p := by
  let cells : List Nat := [0,1,2,3,4,5,6,7]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 0 1) (by decide)
  have hh := finite_0_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_0_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 0 2)
      (laneIndex 0 2 (Projection.scatter [0,9,18,27,36,45,54,63] (BitVec.ofNat 8 n.val))) =
    laneReference 0 2 (Projection.scatter [0,9,18,27,36,45,54,63] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_0_2 (p : Word) :
    LineTable.reference (lanePos 0 2) (laneIndex 0 2 p) = laneReference 0 2 p := by
  let cells : List Nat := [0,9,18,27,36,45,54,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 0 2) (by decide)
  have hh := finite_0_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_0_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 0 3)
      (laneIndex 0 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 0 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_0_3 (p : Word) :
    LineTable.reference (lanePos 0 3) (laneIndex 0 3 p) = laneReference 0 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 0 3) (by decide)
  have hh := finite_0_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq34
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_35_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 35 0)
      (laneIndex 35 0 (Projection.scatter [3,11,19,27,35,43,51,59] (BitVec.ofNat 8 n.val))) =
    laneReference 35 0 (Projection.scatter [3,11,19,27,35,43,51,59] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_35_0 (p : Word) :
    LineTable.reference (lanePos 35 0) (laneIndex 35 0 p) = laneReference 35 0 p := by
  let cells : List Nat := [3,11,19,27,35,43,51,59]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 35 0) (by decide)
  have hh := finite_35_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_35_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 35 1)
      (laneIndex 35 1 (Projection.scatter [32,33,34,35,36,37,38,39] (BitVec.ofNat 8 n.val))) =
    laneReference 35 1 (Projection.scatter [32,33,34,35,36,37,38,39] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_35_1 (p : Word) :
    LineTable.reference (lanePos 35 1) (laneIndex 35 1 p) = laneReference 35 1 p := by
  let cells : List Nat := [32,33,34,35,36,37,38,39]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 35 1) (by decide)
  have hh := finite_35_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_35_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 35 2)
      (laneIndex 35 2 (Projection.scatter [7,14,21,28,35,42,49,56] (BitVec.ofNat 8 n.val))) =
    laneReference 35 2 (Projection.scatter [7,14,21,28,35,42,49,56] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_35_2 (p : Word) :
    LineTable.reference (lanePos 35 2) (laneIndex 35 2 p) = laneReference 35 2 p := by
  let cells : List Nat := [7,14,21,28,35,42,49,56]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 35 2) (by decide)
  have hh := finite_35_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_35_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 35 3)
      (laneIndex 35 3 (Projection.scatter [8,17,26,35,44,53,62] (BitVec.ofNat 8 n.val))) =
    laneReference 35 3 (Projection.scatter [8,17,26,35,44,53,62] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_35_3 (p : Word) :
    LineTable.reference (lanePos 35 3) (laneIndex 35 3 p) = laneReference 35 3 p := by
  let cells : List Nat := [8,17,26,35,44,53,62]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 35 3) (by decide)
  have hh := finite_35_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

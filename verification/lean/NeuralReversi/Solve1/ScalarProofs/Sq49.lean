import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq48
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_49_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 49 0)
      (laneIndex 49 0 (Projection.scatter [1,9,17,25,33,41,49,57] (BitVec.ofNat 8 n.val))) =
    laneReference 49 0 (Projection.scatter [1,9,17,25,33,41,49,57] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_49_0 (p : Word) :
    LineTable.reference (lanePos 49 0) (laneIndex 49 0 p) = laneReference 49 0 p := by
  let cells : List Nat := [1,9,17,25,33,41,49,57]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 49 0) (by decide)
  have hh := finite_49_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_49_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 49 1)
      (laneIndex 49 1 (Projection.scatter [48,49,50,51,52,53,54,55] (BitVec.ofNat 8 n.val))) =
    laneReference 49 1 (Projection.scatter [48,49,50,51,52,53,54,55] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_49_1 (p : Word) :
    LineTable.reference (lanePos 49 1) (laneIndex 49 1 p) = laneReference 49 1 p := by
  let cells : List Nat := [48,49,50,51,52,53,54,55]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 49 1) (by decide)
  have hh := finite_49_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_49_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 49 2)
      (laneIndex 49 2 (Projection.scatter [7,14,21,28,35,42,49,56] (BitVec.ofNat 8 n.val))) =
    laneReference 49 2 (Projection.scatter [7,14,21,28,35,42,49,56] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_49_2 (p : Word) :
    LineTable.reference (lanePos 49 2) (laneIndex 49 2 p) = laneReference 49 2 p := by
  let cells : List Nat := [7,14,21,28,35,42,49,56]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 49 2) (by decide)
  have hh := finite_49_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_49_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 49 3)
      (laneIndex 49 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 49 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_49_3 (p : Word) :
    LineTable.reference (lanePos 49 3) (laneIndex 49 3 p) = laneReference 49 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 49 3) (by decide)
  have hh := finite_49_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

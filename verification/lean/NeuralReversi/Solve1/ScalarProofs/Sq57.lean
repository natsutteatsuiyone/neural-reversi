import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq56
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_57_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 57 0)
      (laneIndex 57 0 (Projection.scatter [1,9,17,25,33,41,49,57] (BitVec.ofNat 8 n.val))) =
    laneReference 57 0 (Projection.scatter [1,9,17,25,33,41,49,57] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_57_0 (p : Word) :
    LineTable.reference (lanePos 57 0) (laneIndex 57 0 p) = laneReference 57 0 p := by
  let cells : List Nat := [1,9,17,25,33,41,49,57]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 57 0) (by decide)
  have hh := finite_57_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_57_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 57 1)
      (laneIndex 57 1 (Projection.scatter [56,57,58,59,60,61,62,63] (BitVec.ofNat 8 n.val))) =
    laneReference 57 1 (Projection.scatter [56,57,58,59,60,61,62,63] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_57_1 (p : Word) :
    LineTable.reference (lanePos 57 1) (laneIndex 57 1 p) = laneReference 57 1 p := by
  let cells : List Nat := [56,57,58,59,60,61,62,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 57 1) (by decide)
  have hh := finite_57_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_57_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 57 2)
      (laneIndex 57 2 (Projection.scatter [15,22,29,36,43,50,57] (BitVec.ofNat 8 n.val))) =
    laneReference 57 2 (Projection.scatter [15,22,29,36,43,50,57] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_57_2 (p : Word) :
    LineTable.reference (lanePos 57 2) (laneIndex 57 2 p) = laneReference 57 2 p := by
  let cells : List Nat := [15,22,29,36,43,50,57]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 57 2) (by decide)
  have hh := finite_57_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_57_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 57 3)
      (laneIndex 57 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 57 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_57_3 (p : Word) :
    LineTable.reference (lanePos 57 3) (laneIndex 57 3 p) = laneReference 57 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 57 3) (by decide)
  have hh := finite_57_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq26
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_27_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 27 0)
      (laneIndex 27 0 (Projection.scatter [3,11,19,27,35,43,51,59] (BitVec.ofNat 8 n.val))) =
    laneReference 27 0 (Projection.scatter [3,11,19,27,35,43,51,59] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_27_0 (p : Word) :
    LineTable.reference (lanePos 27 0) (laneIndex 27 0 p) = laneReference 27 0 p := by
  let cells : List Nat := [3,11,19,27,35,43,51,59]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 27 0) (by decide)
  have hh := finite_27_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_27_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 27 1)
      (laneIndex 27 1 (Projection.scatter [24,25,26,27,28,29,30,31] (BitVec.ofNat 8 n.val))) =
    laneReference 27 1 (Projection.scatter [24,25,26,27,28,29,30,31] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_27_1 (p : Word) :
    LineTable.reference (lanePos 27 1) (laneIndex 27 1 p) = laneReference 27 1 p := by
  let cells : List Nat := [24,25,26,27,28,29,30,31]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 27 1) (by decide)
  have hh := finite_27_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_27_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 27 2)
      (laneIndex 27 2 (Projection.scatter [6,13,20,27,34,41,48] (BitVec.ofNat 8 n.val))) =
    laneReference 27 2 (Projection.scatter [6,13,20,27,34,41,48] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_27_2 (p : Word) :
    LineTable.reference (lanePos 27 2) (laneIndex 27 2 p) = laneReference 27 2 p := by
  let cells : List Nat := [6,13,20,27,34,41,48]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 27 2) (by decide)
  have hh := finite_27_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_27_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 27 3)
      (laneIndex 27 3 (Projection.scatter [0,9,18,27,36,45,54,63] (BitVec.ofNat 8 n.val))) =
    laneReference 27 3 (Projection.scatter [0,9,18,27,36,45,54,63] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_27_3 (p : Word) :
    LineTable.reference (lanePos 27 3) (laneIndex 27 3 p) = laneReference 27 3 p := by
  let cells : List Nat := [0,9,18,27,36,45,54,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 27 3) (by decide)
  have hh := finite_27_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

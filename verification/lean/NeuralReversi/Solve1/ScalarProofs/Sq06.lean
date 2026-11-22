import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq05
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_6_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 6 0)
      (laneIndex 6 0 (Projection.scatter [6,14,22,30,38,46,54,62] (BitVec.ofNat 8 n.val))) =
    laneReference 6 0 (Projection.scatter [6,14,22,30,38,46,54,62] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_6_0 (p : Word) :
    LineTable.reference (lanePos 6 0) (laneIndex 6 0 p) = laneReference 6 0 p := by
  let cells : List Nat := [6,14,22,30,38,46,54,62]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 6 0) (by decide)
  have hh := finite_6_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_6_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 6 1)
      (laneIndex 6 1 (Projection.scatter [0,1,2,3,4,5,6,7] (BitVec.ofNat 8 n.val))) =
    laneReference 6 1 (Projection.scatter [0,1,2,3,4,5,6,7] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_6_1 (p : Word) :
    LineTable.reference (lanePos 6 1) (laneIndex 6 1 p) = laneReference 6 1 p := by
  let cells : List Nat := [0,1,2,3,4,5,6,7]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 6 1) (by decide)
  have hh := finite_6_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_6_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 6 2)
      (laneIndex 6 2 (Projection.scatter [6,13,20,27,34,41,48] (BitVec.ofNat 8 n.val))) =
    laneReference 6 2 (Projection.scatter [6,13,20,27,34,41,48] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_6_2 (p : Word) :
    LineTable.reference (lanePos 6 2) (laneIndex 6 2 p) = laneReference 6 2 p := by
  let cells : List Nat := [6,13,20,27,34,41,48]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 6 2) (by decide)
  have hh := finite_6_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_6_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 6 3)
      (laneIndex 6 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 6 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_6_3 (p : Word) :
    LineTable.reference (lanePos 6 3) (laneIndex 6 3 p) = laneReference 6 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 6 3) (by decide)
  have hh := finite_6_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

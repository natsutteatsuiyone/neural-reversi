import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq33
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_34_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 34 0)
      (laneIndex 34 0 (Projection.scatter [2,10,18,26,34,42,50,58] (BitVec.ofNat 8 n.val))) =
    laneReference 34 0 (Projection.scatter [2,10,18,26,34,42,50,58] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_34_0 (p : Word) :
    LineTable.reference (lanePos 34 0) (laneIndex 34 0 p) = laneReference 34 0 p := by
  let cells : List Nat := [2,10,18,26,34,42,50,58]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 34 0) (by decide)
  have hh := finite_34_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_34_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 34 1)
      (laneIndex 34 1 (Projection.scatter [32,33,34,35,36,37,38,39] (BitVec.ofNat 8 n.val))) =
    laneReference 34 1 (Projection.scatter [32,33,34,35,36,37,38,39] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_34_1 (p : Word) :
    LineTable.reference (lanePos 34 1) (laneIndex 34 1 p) = laneReference 34 1 p := by
  let cells : List Nat := [32,33,34,35,36,37,38,39]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 34 1) (by decide)
  have hh := finite_34_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_34_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 34 2)
      (laneIndex 34 2 (Projection.scatter [6,13,20,27,34,41,48] (BitVec.ofNat 8 n.val))) =
    laneReference 34 2 (Projection.scatter [6,13,20,27,34,41,48] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_34_2 (p : Word) :
    LineTable.reference (lanePos 34 2) (laneIndex 34 2 p) = laneReference 34 2 p := by
  let cells : List Nat := [6,13,20,27,34,41,48]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 34 2) (by decide)
  have hh := finite_34_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_34_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 34 3)
      (laneIndex 34 3 (Projection.scatter [16,25,34,43,52,61] (BitVec.ofNat 8 n.val))) =
    laneReference 34 3 (Projection.scatter [16,25,34,43,52,61] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_34_3 (p : Word) :
    LineTable.reference (lanePos 34 3) (laneIndex 34 3 p) = laneReference 34 3 p := by
  let cells : List Nat := [16,25,34,43,52,61]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 34 3) (by decide)
  have hh := finite_34_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

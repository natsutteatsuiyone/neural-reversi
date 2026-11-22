import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq37
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_38_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 38 0)
      (laneIndex 38 0 (Projection.scatter [6,14,22,30,38,46,54,62] (BitVec.ofNat 8 n.val))) =
    laneReference 38 0 (Projection.scatter [6,14,22,30,38,46,54,62] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_38_0 (p : Word) :
    LineTable.reference (lanePos 38 0) (laneIndex 38 0 p) = laneReference 38 0 p := by
  let cells : List Nat := [6,14,22,30,38,46,54,62]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 38 0) (by decide)
  have hh := finite_38_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_38_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 38 1)
      (laneIndex 38 1 (Projection.scatter [32,33,34,35,36,37,38,39] (BitVec.ofNat 8 n.val))) =
    laneReference 38 1 (Projection.scatter [32,33,34,35,36,37,38,39] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_38_1 (p : Word) :
    LineTable.reference (lanePos 38 1) (laneIndex 38 1 p) = laneReference 38 1 p := by
  let cells : List Nat := [32,33,34,35,36,37,38,39]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 38 1) (by decide)
  have hh := finite_38_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_38_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 38 2)
      (laneIndex 38 2 (Projection.scatter [2,11,20,29,38,45,52,59] (BitVec.ofNat 8 n.val))) =
    laneReference 38 2 (Projection.scatter [2,11,20,29,38,45,52,59] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_38_2 (p : Word) :
    LineTable.reference (lanePos 38 2) (laneIndex 38 2 p) = laneReference 38 2 p := by
  let cells : List Nat := [2,11,20,29,38,45,52,59]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 38 2) (by decide)
  have hh := finite_38_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_38_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 38 3)
      (laneIndex 38 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 38 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_38_3 (p : Word) :
    LineTable.reference (lanePos 38 3) (laneIndex 38 3 p) = laneReference 38 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 38 3) (by decide)
  have hh := finite_38_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

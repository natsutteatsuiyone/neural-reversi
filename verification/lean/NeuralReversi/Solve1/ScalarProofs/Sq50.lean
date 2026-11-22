import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq49
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_50_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 50 0)
      (laneIndex 50 0 (Projection.scatter [2,10,18,26,34,42,50,58] (BitVec.ofNat 8 n.val))) =
    laneReference 50 0 (Projection.scatter [2,10,18,26,34,42,50,58] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_50_0 (p : Word) :
    LineTable.reference (lanePos 50 0) (laneIndex 50 0 p) = laneReference 50 0 p := by
  let cells : List Nat := [2,10,18,26,34,42,50,58]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 50 0) (by decide)
  have hh := finite_50_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_50_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 50 1)
      (laneIndex 50 1 (Projection.scatter [48,49,50,51,52,53,54,55] (BitVec.ofNat 8 n.val))) =
    laneReference 50 1 (Projection.scatter [48,49,50,51,52,53,54,55] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_50_1 (p : Word) :
    LineTable.reference (lanePos 50 1) (laneIndex 50 1 p) = laneReference 50 1 p := by
  let cells : List Nat := [48,49,50,51,52,53,54,55]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 50 1) (by decide)
  have hh := finite_50_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_50_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 50 2)
      (laneIndex 50 2 (Projection.scatter [15,22,29,32,36,41,43,50] (BitVec.ofNat 8 n.val))) =
    laneReference 50 2 (Projection.scatter [15,22,29,32,36,41,43,50] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_50_2 (p : Word) :
    LineTable.reference (lanePos 50 2) (laneIndex 50 2 p) = laneReference 50 2 p := by
  let cells : List Nat := [15,22,29,32,36,41,43,50]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 50 2) (by decide)
  have hh := finite_50_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_50_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 50 3)
      (laneIndex 50 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 50 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_50_3 (p : Word) :
    LineTable.reference (lanePos 50 3) (laneIndex 50 3 p) = laneReference 50 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 50 3) (by decide)
  have hh := finite_50_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

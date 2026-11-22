import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq31
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_32_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 32 0)
      (laneIndex 32 0 (Projection.scatter [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 n.val))) =
    laneReference 32 0 (Projection.scatter [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_32_0 (p : Word) :
    LineTable.reference (lanePos 32 0) (laneIndex 32 0 p) = laneReference 32 0 p := by
  let cells : List Nat := [0,8,16,24,32,40,48,56]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 32 0) (by decide)
  have hh := finite_32_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_32_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 32 1)
      (laneIndex 32 1 (Projection.scatter [32,33,34,35,36,37,38,39] (BitVec.ofNat 8 n.val))) =
    laneReference 32 1 (Projection.scatter [32,33,34,35,36,37,38,39] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_32_1 (p : Word) :
    LineTable.reference (lanePos 32 1) (laneIndex 32 1 p) = laneReference 32 1 p := by
  let cells : List Nat := [32,33,34,35,36,37,38,39]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 32 1) (by decide)
  have hh := finite_32_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_32_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 32 2)
      (laneIndex 32 2 (Projection.scatter [4,11,18,25,32,41,50,59] (BitVec.ofNat 8 n.val))) =
    laneReference 32 2 (Projection.scatter [4,11,18,25,32,41,50,59] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_32_2 (p : Word) :
    LineTable.reference (lanePos 32 2) (laneIndex 32 2 p) = laneReference 32 2 p := by
  let cells : List Nat := [4,11,18,25,32,41,50,59]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 32 2) (by decide)
  have hh := finite_32_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_32_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 32 3)
      (laneIndex 32 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 32 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_32_3 (p : Word) :
    LineTable.reference (lanePos 32 3) (laneIndex 32 3 p) = laneReference 32 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 32 3) (by decide)
  have hh := finite_32_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

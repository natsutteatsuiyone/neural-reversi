import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq47
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_48_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 48 0)
      (laneIndex 48 0 (Projection.scatter [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 n.val))) =
    laneReference 48 0 (Projection.scatter [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_48_0 (p : Word) :
    LineTable.reference (lanePos 48 0) (laneIndex 48 0 p) = laneReference 48 0 p := by
  let cells : List Nat := [0,8,16,24,32,40,48,56]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 48 0) (by decide)
  have hh := finite_48_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_48_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 48 1)
      (laneIndex 48 1 (Projection.scatter [48,49,50,51,52,53,54,55] (BitVec.ofNat 8 n.val))) =
    laneReference 48 1 (Projection.scatter [48,49,50,51,52,53,54,55] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_48_1 (p : Word) :
    LineTable.reference (lanePos 48 1) (laneIndex 48 1 p) = laneReference 48 1 p := by
  let cells : List Nat := [48,49,50,51,52,53,54,55]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 48 1) (by decide)
  have hh := finite_48_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_48_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 48 2)
      (laneIndex 48 2 (Projection.scatter [6,13,20,27,34,41,48] (BitVec.ofNat 8 n.val))) =
    laneReference 48 2 (Projection.scatter [6,13,20,27,34,41,48] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_48_2 (p : Word) :
    LineTable.reference (lanePos 48 2) (laneIndex 48 2 p) = laneReference 48 2 p := by
  let cells : List Nat := [6,13,20,27,34,41,48]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 48 2) (by decide)
  have hh := finite_48_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_48_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 48 3)
      (laneIndex 48 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 48 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_48_3 (p : Word) :
    LineTable.reference (lanePos 48 3) (laneIndex 48 3 p) = laneReference 48 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 48 3) (by decide)
  have hh := finite_48_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

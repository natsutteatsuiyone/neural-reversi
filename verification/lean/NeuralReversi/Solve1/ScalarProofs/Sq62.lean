import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq61
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_62_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 62 0)
      (laneIndex 62 0 (Projection.scatter [6,14,22,30,38,46,54,62] (BitVec.ofNat 8 n.val))) =
    laneReference 62 0 (Projection.scatter [6,14,22,30,38,46,54,62] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_62_0 (p : Word) :
    LineTable.reference (lanePos 62 0) (laneIndex 62 0 p) = laneReference 62 0 p := by
  let cells : List Nat := [6,14,22,30,38,46,54,62]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 62 0) (by decide)
  have hh := finite_62_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_62_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 62 1)
      (laneIndex 62 1 (Projection.scatter [56,57,58,59,60,61,62,63] (BitVec.ofNat 8 n.val))) =
    laneReference 62 1 (Projection.scatter [56,57,58,59,60,61,62,63] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_62_1 (p : Word) :
    LineTable.reference (lanePos 62 1) (laneIndex 62 1 p) = laneReference 62 1 p := by
  let cells : List Nat := [56,57,58,59,60,61,62,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 62 1) (by decide)
  have hh := finite_62_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_62_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 62 2)
      (laneIndex 62 2 (Projection.scatter [8,17,26,35,44,53,62] (BitVec.ofNat 8 n.val))) =
    laneReference 62 2 (Projection.scatter [8,17,26,35,44,53,62] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_62_2 (p : Word) :
    LineTable.reference (lanePos 62 2) (laneIndex 62 2 p) = laneReference 62 2 p := by
  let cells : List Nat := [8,17,26,35,44,53,62]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 62 2) (by decide)
  have hh := finite_62_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_62_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 62 3)
      (laneIndex 62 3 (Projection.scatter [] (BitVec.ofNat 8 n.val))) =
    laneReference 62 3 (Projection.scatter [] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_62_3 (p : Word) :
    LineTable.reference (lanePos 62 3) (laneIndex 62 3 p) = laneReference 62 3 p := by
  let cells : List Nat := []
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 62 3) (by decide)
  have hh := finite_62_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

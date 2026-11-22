import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq44
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_45_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 45 0)
      (laneIndex 45 0 (Projection.scatter [5,13,21,29,37,45,53,61] (BitVec.ofNat 8 n.val))) =
    laneReference 45 0 (Projection.scatter [5,13,21,29,37,45,53,61] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_45_0 (p : Word) :
    LineTable.reference (lanePos 45 0) (laneIndex 45 0 p) = laneReference 45 0 p := by
  let cells : List Nat := [5,13,21,29,37,45,53,61]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 45 0) (by decide)
  have hh := finite_45_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_45_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 45 1)
      (laneIndex 45 1 (Projection.scatter [40,41,42,43,44,45,46,47] (BitVec.ofNat 8 n.val))) =
    laneReference 45 1 (Projection.scatter [40,41,42,43,44,45,46,47] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_45_1 (p : Word) :
    LineTable.reference (lanePos 45 1) (laneIndex 45 1 p) = laneReference 45 1 p := by
  let cells : List Nat := [40,41,42,43,44,45,46,47]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 45 1) (by decide)
  have hh := finite_45_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_45_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 45 2)
      (laneIndex 45 2 (Projection.scatter [31,38,45,52,59] (BitVec.ofNat 8 n.val))) =
    laneReference 45 2 (Projection.scatter [31,38,45,52,59] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_45_2 (p : Word) :
    LineTable.reference (lanePos 45 2) (laneIndex 45 2 p) = laneReference 45 2 p := by
  let cells : List Nat := [31,38,45,52,59]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 45 2) (by decide)
  have hh := finite_45_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_45_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 45 3)
      (laneIndex 45 3 (Projection.scatter [0,9,18,27,36,45,54,63] (BitVec.ofNat 8 n.val))) =
    laneReference 45 3 (Projection.scatter [0,9,18,27,36,45,54,63] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_45_3 (p : Word) :
    LineTable.reference (lanePos 45 3) (laneIndex 45 3 p) = laneReference 45 3 p := by
  let cells : List Nat := [0,9,18,27,36,45,54,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 45 3) (by decide)
  have hh := finite_45_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

import NeuralReversi.Solve1.ScalarModel
import NeuralReversi.Solve1.ScalarProofs.Sq43
namespace NeuralReversi.Solve1.Scalar
set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem finite_44_0 : ∀ n : Fin 256,
    LineTable.reference (lanePos 44 0)
      (laneIndex 44 0 (Projection.scatter [4,12,20,28,36,44,52,60] (BitVec.ofNat 8 n.val))) =
    laneReference 44 0 (Projection.scatter [4,12,20,28,36,44,52,60] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_44_0 (p : Word) :
    LineTable.reference (lanePos 44 0) (laneIndex 44 0 p) = laneReference 44 0 p := by
  let cells : List Nat := [4,12,20,28,36,44,52,60]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 44 0) (by decide)
  have hh := finite_44_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_44_1 : ∀ n : Fin 256,
    LineTable.reference (lanePos 44 1)
      (laneIndex 44 1 (Projection.scatter [40,41,42,43,44,45,46,47] (BitVec.ofNat 8 n.val))) =
    laneReference 44 1 (Projection.scatter [40,41,42,43,44,45,46,47] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_44_1 (p : Word) :
    LineTable.reference (lanePos 44 1) (laneIndex 44 1 p) = laneReference 44 1 p := by
  let cells : List Nat := [40,41,42,43,44,45,46,47]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 44 1) (by decide)
  have hh := finite_44_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_44_2 : ∀ n : Fin 256,
    LineTable.reference (lanePos 44 2)
      (laneIndex 44 2 (Projection.scatter [23,30,37,44,51,58] (BitVec.ofNat 8 n.val))) =
    laneReference 44 2 (Projection.scatter [23,30,37,44,51,58] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_44_2 (p : Word) :
    LineTable.reference (lanePos 44 2) (laneIndex 44 2 p) = laneReference 44 2 p := by
  let cells : List Nat := [23,30,37,44,51,58]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 44 2) (by decide)
  have hh := finite_44_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

private theorem finite_44_3 : ∀ n : Fin 256,
    LineTable.reference (lanePos 44 3)
      (laneIndex 44 3 (Projection.scatter [8,17,26,35,44,53,62] (BitVec.ofNat 8 n.val))) =
    laneReference 44 3 (Projection.scatter [8,17,26,35,44,53,62] (BitVec.ofNat 8 n.val)) := by decide +kernel

theorem lane_44_3 (p : Word) :
    LineTable.reference (lanePos 44 3) (laneIndex 44 3 p) = laneReference 44 3 p := by
  let cells : List Nat := [8,17,26,35,44,53,62]
  have hp := Projection.masked_eq_scatter cells p (by decide) (laneMask 44 3) (by decide)
  have hh := finite_44_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simp [laneIndex, laneMask, params, laneReference, Geometry.ray, Geometry.raySquares,
    List.range_succ, BitVec.and_assoc, BitVec.ushiftRight_and_distrib, and255, -BitVec.getLsbD_eq_getElem] at hh ⊢
  simpa only [and255] using hh

end NeuralReversi.Solve1.Scalar

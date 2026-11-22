import NeuralReversi.Solve1.NeonModel
import NeuralReversi.Solve1.NeonProofs.Part4
namespace NeuralReversi.Solve1.Neon
set_option maxRecDepth 100000
set_option maxHeartbeats 0
private theorem finite_xor_40_0 : ∀ b : Fin 256,
    Scalar.laneIndex 40 0 (Projection.scatter [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 b.val)) ^^^ laneXor 40 0 =
    Scalar.laneIndex 40 0 (~~~(Projection.scatter [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_40_0 (p : Word) :
    Scalar.laneIndex 40 0 p ^^^ laneXor 40 0 = Scalar.laneIndex 40 0 (~~~p) := by
  let cells : List Nat := [0,8,16,24,32,40,48,56]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 40 0) (by decide)
  have hh := finite_xor_40_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_40_1 : ∀ b : Fin 256,
    Scalar.laneIndex 40 1 (Projection.scatter [40,41,42,43,44,45,46,47] (BitVec.ofNat 8 b.val)) ^^^ laneXor 40 1 =
    Scalar.laneIndex 40 1 (~~~(Projection.scatter [40,41,42,43,44,45,46,47] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_40_1 (p : Word) :
    Scalar.laneIndex 40 1 p ^^^ laneXor 40 1 = Scalar.laneIndex 40 1 (~~~p) := by
  let cells : List Nat := [40,41,42,43,44,45,46,47]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 40 1) (by decide)
  have hh := finite_xor_40_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_40_2 : ∀ b : Fin 256,
    Scalar.laneIndex 40 2 (Projection.scatter [5,12,19,26,33,40,49,58] (BitVec.ofNat 8 b.val)) ^^^ laneXor 40 2 =
    Scalar.laneIndex 40 2 (~~~(Projection.scatter [5,12,19,26,33,40,49,58] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_40_2 (p : Word) :
    Scalar.laneIndex 40 2 p ^^^ laneXor 40 2 = Scalar.laneIndex 40 2 (~~~p) := by
  let cells : List Nat := [5,12,19,26,33,40,49,58]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 40 2) (by decide)
  have hh := finite_xor_40_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_41_0 : ∀ b : Fin 256,
    Scalar.laneIndex 41 0 (Projection.scatter [1,9,17,25,33,41,49,57] (BitVec.ofNat 8 b.val)) ^^^ laneXor 41 0 =
    Scalar.laneIndex 41 0 (~~~(Projection.scatter [1,9,17,25,33,41,49,57] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_41_0 (p : Word) :
    Scalar.laneIndex 41 0 p ^^^ laneXor 41 0 = Scalar.laneIndex 41 0 (~~~p) := by
  let cells : List Nat := [1,9,17,25,33,41,49,57]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 41 0) (by decide)
  have hh := finite_xor_41_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_41_1 : ∀ b : Fin 256,
    Scalar.laneIndex 41 1 (Projection.scatter [40,41,42,43,44,45,46,47] (BitVec.ofNat 8 b.val)) ^^^ laneXor 41 1 =
    Scalar.laneIndex 41 1 (~~~(Projection.scatter [40,41,42,43,44,45,46,47] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_41_1 (p : Word) :
    Scalar.laneIndex 41 1 p ^^^ laneXor 41 1 = Scalar.laneIndex 41 1 (~~~p) := by
  let cells : List Nat := [40,41,42,43,44,45,46,47]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 41 1) (by decide)
  have hh := finite_xor_41_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_41_2 : ∀ b : Fin 256,
    Scalar.laneIndex 41 2 (Projection.scatter [6,13,20,27,34,41,50,59] (BitVec.ofNat 8 b.val)) ^^^ laneXor 41 2 =
    Scalar.laneIndex 41 2 (~~~(Projection.scatter [6,13,20,27,34,41,50,59] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_41_2 (p : Word) :
    Scalar.laneIndex 41 2 p ^^^ laneXor 41 2 = Scalar.laneIndex 41 2 (~~~p) := by
  let cells : List Nat := [6,13,20,27,34,41,50,59]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 41 2) (by decide)
  have hh := finite_xor_41_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_42_0 : ∀ b : Fin 256,
    Scalar.laneIndex 42 0 (Projection.scatter [2,10,18,26,34,42,50,58] (BitVec.ofNat 8 b.val)) ^^^ laneXor 42 0 =
    Scalar.laneIndex 42 0 (~~~(Projection.scatter [2,10,18,26,34,42,50,58] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_42_0 (p : Word) :
    Scalar.laneIndex 42 0 p ^^^ laneXor 42 0 = Scalar.laneIndex 42 0 (~~~p) := by
  let cells : List Nat := [2,10,18,26,34,42,50,58]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 42 0) (by decide)
  have hh := finite_xor_42_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_42_1 : ∀ b : Fin 256,
    Scalar.laneIndex 42 1 (Projection.scatter [40,41,42,43,44,45,46,47] (BitVec.ofNat 8 b.val)) ^^^ laneXor 42 1 =
    Scalar.laneIndex 42 1 (~~~(Projection.scatter [40,41,42,43,44,45,46,47] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_42_1 (p : Word) :
    Scalar.laneIndex 42 1 p ^^^ laneXor 42 1 = Scalar.laneIndex 42 1 (~~~p) := by
  let cells : List Nat := [40,41,42,43,44,45,46,47]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 42 1) (by decide)
  have hh := finite_xor_42_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_42_2 : ∀ b : Fin 256,
    Scalar.laneIndex 42 2 (Projection.scatter [7,14,21,28,35,42,49,56] (BitVec.ofNat 8 b.val)) ^^^ laneXor 42 2 =
    Scalar.laneIndex 42 2 (~~~(Projection.scatter [7,14,21,28,35,42,49,56] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_42_2 (p : Word) :
    Scalar.laneIndex 42 2 p ^^^ laneXor 42 2 = Scalar.laneIndex 42 2 (~~~p) := by
  let cells : List Nat := [7,14,21,28,35,42,49,56]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 42 2) (by decide)
  have hh := finite_xor_42_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_42_3 : ∀ b : Fin 256,
    Scalar.laneIndex 42 3 (Projection.scatter [24,33,42,51,60] (BitVec.ofNat 8 b.val)) ^^^ laneXor 42 3 =
    Scalar.laneIndex 42 3 (~~~(Projection.scatter [24,33,42,51,60] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_42_3 (p : Word) :
    Scalar.laneIndex 42 3 p ^^^ laneXor 42 3 = Scalar.laneIndex 42 3 (~~~p) := by
  let cells : List Nat := [24,33,42,51,60]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 42 3) (by decide)
  have hh := finite_xor_42_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_43_0 : ∀ b : Fin 256,
    Scalar.laneIndex 43 0 (Projection.scatter [3,11,19,27,35,43,51,59] (BitVec.ofNat 8 b.val)) ^^^ laneXor 43 0 =
    Scalar.laneIndex 43 0 (~~~(Projection.scatter [3,11,19,27,35,43,51,59] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_43_0 (p : Word) :
    Scalar.laneIndex 43 0 p ^^^ laneXor 43 0 = Scalar.laneIndex 43 0 (~~~p) := by
  let cells : List Nat := [3,11,19,27,35,43,51,59]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 43 0) (by decide)
  have hh := finite_xor_43_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_43_1 : ∀ b : Fin 256,
    Scalar.laneIndex 43 1 (Projection.scatter [40,41,42,43,44,45,46,47] (BitVec.ofNat 8 b.val)) ^^^ laneXor 43 1 =
    Scalar.laneIndex 43 1 (~~~(Projection.scatter [40,41,42,43,44,45,46,47] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_43_1 (p : Word) :
    Scalar.laneIndex 43 1 p ^^^ laneXor 43 1 = Scalar.laneIndex 43 1 (~~~p) := by
  let cells : List Nat := [40,41,42,43,44,45,46,47]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 43 1) (by decide)
  have hh := finite_xor_43_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_43_2 : ∀ b : Fin 256,
    Scalar.laneIndex 43 2 (Projection.scatter [15,22,29,36,43,50,57] (BitVec.ofNat 8 b.val)) ^^^ laneXor 43 2 =
    Scalar.laneIndex 43 2 (~~~(Projection.scatter [15,22,29,36,43,50,57] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_43_2 (p : Word) :
    Scalar.laneIndex 43 2 p ^^^ laneXor 43 2 = Scalar.laneIndex 43 2 (~~~p) := by
  let cells : List Nat := [15,22,29,36,43,50,57]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 43 2) (by decide)
  have hh := finite_xor_43_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_43_3 : ∀ b : Fin 256,
    Scalar.laneIndex 43 3 (Projection.scatter [16,25,34,43,52,61] (BitVec.ofNat 8 b.val)) ^^^ laneXor 43 3 =
    Scalar.laneIndex 43 3 (~~~(Projection.scatter [16,25,34,43,52,61] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_43_3 (p : Word) :
    Scalar.laneIndex 43 3 p ^^^ laneXor 43 3 = Scalar.laneIndex 43 3 (~~~p) := by
  let cells : List Nat := [16,25,34,43,52,61]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 43 3) (by decide)
  have hh := finite_xor_43_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_44_0 : ∀ b : Fin 256,
    Scalar.laneIndex 44 0 (Projection.scatter [4,12,20,28,36,44,52,60] (BitVec.ofNat 8 b.val)) ^^^ laneXor 44 0 =
    Scalar.laneIndex 44 0 (~~~(Projection.scatter [4,12,20,28,36,44,52,60] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_44_0 (p : Word) :
    Scalar.laneIndex 44 0 p ^^^ laneXor 44 0 = Scalar.laneIndex 44 0 (~~~p) := by
  let cells : List Nat := [4,12,20,28,36,44,52,60]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 44 0) (by decide)
  have hh := finite_xor_44_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_44_1 : ∀ b : Fin 256,
    Scalar.laneIndex 44 1 (Projection.scatter [40,41,42,43,44,45,46,47] (BitVec.ofNat 8 b.val)) ^^^ laneXor 44 1 =
    Scalar.laneIndex 44 1 (~~~(Projection.scatter [40,41,42,43,44,45,46,47] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_44_1 (p : Word) :
    Scalar.laneIndex 44 1 p ^^^ laneXor 44 1 = Scalar.laneIndex 44 1 (~~~p) := by
  let cells : List Nat := [40,41,42,43,44,45,46,47]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 44 1) (by decide)
  have hh := finite_xor_44_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_44_2 : ∀ b : Fin 256,
    Scalar.laneIndex 44 2 (Projection.scatter [23,30,37,44,51,58] (BitVec.ofNat 8 b.val)) ^^^ laneXor 44 2 =
    Scalar.laneIndex 44 2 (~~~(Projection.scatter [23,30,37,44,51,58] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_44_2 (p : Word) :
    Scalar.laneIndex 44 2 p ^^^ laneXor 44 2 = Scalar.laneIndex 44 2 (~~~p) := by
  let cells : List Nat := [23,30,37,44,51,58]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 44 2) (by decide)
  have hh := finite_xor_44_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_44_3 : ∀ b : Fin 256,
    Scalar.laneIndex 44 3 (Projection.scatter [8,17,26,35,44,53,62] (BitVec.ofNat 8 b.val)) ^^^ laneXor 44 3 =
    Scalar.laneIndex 44 3 (~~~(Projection.scatter [8,17,26,35,44,53,62] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_44_3 (p : Word) :
    Scalar.laneIndex 44 3 p ^^^ laneXor 44 3 = Scalar.laneIndex 44 3 (~~~p) := by
  let cells : List Nat := [8,17,26,35,44,53,62]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 44 3) (by decide)
  have hh := finite_xor_44_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_45_0 : ∀ b : Fin 256,
    Scalar.laneIndex 45 0 (Projection.scatter [5,13,21,29,37,45,53,61] (BitVec.ofNat 8 b.val)) ^^^ laneXor 45 0 =
    Scalar.laneIndex 45 0 (~~~(Projection.scatter [5,13,21,29,37,45,53,61] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_45_0 (p : Word) :
    Scalar.laneIndex 45 0 p ^^^ laneXor 45 0 = Scalar.laneIndex 45 0 (~~~p) := by
  let cells : List Nat := [5,13,21,29,37,45,53,61]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 45 0) (by decide)
  have hh := finite_xor_45_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_45_1 : ∀ b : Fin 256,
    Scalar.laneIndex 45 1 (Projection.scatter [40,41,42,43,44,45,46,47] (BitVec.ofNat 8 b.val)) ^^^ laneXor 45 1 =
    Scalar.laneIndex 45 1 (~~~(Projection.scatter [40,41,42,43,44,45,46,47] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_45_1 (p : Word) :
    Scalar.laneIndex 45 1 p ^^^ laneXor 45 1 = Scalar.laneIndex 45 1 (~~~p) := by
  let cells : List Nat := [40,41,42,43,44,45,46,47]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 45 1) (by decide)
  have hh := finite_xor_45_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_45_2 : ∀ b : Fin 256,
    Scalar.laneIndex 45 2 (Projection.scatter [31,38,45,52,59] (BitVec.ofNat 8 b.val)) ^^^ laneXor 45 2 =
    Scalar.laneIndex 45 2 (~~~(Projection.scatter [31,38,45,52,59] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_45_2 (p : Word) :
    Scalar.laneIndex 45 2 p ^^^ laneXor 45 2 = Scalar.laneIndex 45 2 (~~~p) := by
  let cells : List Nat := [31,38,45,52,59]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 45 2) (by decide)
  have hh := finite_xor_45_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_45_3 : ∀ b : Fin 256,
    Scalar.laneIndex 45 3 (Projection.scatter [0,9,18,27,36,45,54,63] (BitVec.ofNat 8 b.val)) ^^^ laneXor 45 3 =
    Scalar.laneIndex 45 3 (~~~(Projection.scatter [0,9,18,27,36,45,54,63] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_45_3 (p : Word) :
    Scalar.laneIndex 45 3 p ^^^ laneXor 45 3 = Scalar.laneIndex 45 3 (~~~p) := by
  let cells : List Nat := [0,9,18,27,36,45,54,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 45 3) (by decide)
  have hh := finite_xor_45_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_46_0 : ∀ b : Fin 256,
    Scalar.laneIndex 46 0 (Projection.scatter [6,14,22,30,38,46,54,62] (BitVec.ofNat 8 b.val)) ^^^ laneXor 46 0 =
    Scalar.laneIndex 46 0 (~~~(Projection.scatter [6,14,22,30,38,46,54,62] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_46_0 (p : Word) :
    Scalar.laneIndex 46 0 p ^^^ laneXor 46 0 = Scalar.laneIndex 46 0 (~~~p) := by
  let cells : List Nat := [6,14,22,30,38,46,54,62]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 46 0) (by decide)
  have hh := finite_xor_46_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_46_1 : ∀ b : Fin 256,
    Scalar.laneIndex 46 1 (Projection.scatter [40,41,42,43,44,45,46,47] (BitVec.ofNat 8 b.val)) ^^^ laneXor 46 1 =
    Scalar.laneIndex 46 1 (~~~(Projection.scatter [40,41,42,43,44,45,46,47] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_46_1 (p : Word) :
    Scalar.laneIndex 46 1 p ^^^ laneXor 46 1 = Scalar.laneIndex 46 1 (~~~p) := by
  let cells : List Nat := [40,41,42,43,44,45,46,47]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 46 1) (by decide)
  have hh := finite_xor_46_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_46_2 : ∀ b : Fin 256,
    Scalar.laneIndex 46 2 (Projection.scatter [1,10,19,28,37,46,53,60] (BitVec.ofNat 8 b.val)) ^^^ laneXor 46 2 =
    Scalar.laneIndex 46 2 (~~~(Projection.scatter [1,10,19,28,37,46,53,60] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_46_2 (p : Word) :
    Scalar.laneIndex 46 2 p ^^^ laneXor 46 2 = Scalar.laneIndex 46 2 (~~~p) := by
  let cells : List Nat := [1,10,19,28,37,46,53,60]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 46 2) (by decide)
  have hh := finite_xor_46_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_47_0 : ∀ b : Fin 256,
    Scalar.laneIndex 47 0 (Projection.scatter [7,15,23,31,39,47,55,63] (BitVec.ofNat 8 b.val)) ^^^ laneXor 47 0 =
    Scalar.laneIndex 47 0 (~~~(Projection.scatter [7,15,23,31,39,47,55,63] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_47_0 (p : Word) :
    Scalar.laneIndex 47 0 p ^^^ laneXor 47 0 = Scalar.laneIndex 47 0 (~~~p) := by
  let cells : List Nat := [7,15,23,31,39,47,55,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 47 0) (by decide)
  have hh := finite_xor_47_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_47_1 : ∀ b : Fin 256,
    Scalar.laneIndex 47 1 (Projection.scatter [40,41,42,43,44,45,46,47] (BitVec.ofNat 8 b.val)) ^^^ laneXor 47 1 =
    Scalar.laneIndex 47 1 (~~~(Projection.scatter [40,41,42,43,44,45,46,47] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_47_1 (p : Word) :
    Scalar.laneIndex 47 1 p ^^^ laneXor 47 1 = Scalar.laneIndex 47 1 (~~~p) := by
  let cells : List Nat := [40,41,42,43,44,45,46,47]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 47 1) (by decide)
  have hh := finite_xor_47_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_47_2 : ∀ b : Fin 256,
    Scalar.laneIndex 47 2 (Projection.scatter [2,11,20,29,38,47,54,61] (BitVec.ofNat 8 b.val)) ^^^ laneXor 47 2 =
    Scalar.laneIndex 47 2 (~~~(Projection.scatter [2,11,20,29,38,47,54,61] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_47_2 (p : Word) :
    Scalar.laneIndex 47 2 p ^^^ laneXor 47 2 = Scalar.laneIndex 47 2 (~~~p) := by
  let cells : List Nat := [2,11,20,29,38,47,54,61]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 47 2) (by decide)
  have hh := finite_xor_47_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

end NeuralReversi.Solve1.Neon

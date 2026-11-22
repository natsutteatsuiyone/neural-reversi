import NeuralReversi.Solve1.NeonModel
import NeuralReversi.Solve1.NeonProofs.Part6
namespace NeuralReversi.Solve1.Neon
set_option maxRecDepth 100000
set_option maxHeartbeats 0
private theorem finite_xor_56_0 : ∀ b : Fin 256,
    Scalar.laneIndex 56 0 (Projection.scatter [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 b.val)) ^^^ laneXor 56 0 =
    Scalar.laneIndex 56 0 (~~~(Projection.scatter [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_56_0 (p : Word) :
    Scalar.laneIndex 56 0 p ^^^ laneXor 56 0 = Scalar.laneIndex 56 0 (~~~p) := by
  let cells : List Nat := [0,8,16,24,32,40,48,56]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 56 0) (by decide)
  have hh := finite_xor_56_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_56_1 : ∀ b : Fin 256,
    Scalar.laneIndex 56 1 (Projection.scatter [56,57,58,59,60,61,62,63] (BitVec.ofNat 8 b.val)) ^^^ laneXor 56 1 =
    Scalar.laneIndex 56 1 (~~~(Projection.scatter [56,57,58,59,60,61,62,63] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_56_1 (p : Word) :
    Scalar.laneIndex 56 1 p ^^^ laneXor 56 1 = Scalar.laneIndex 56 1 (~~~p) := by
  let cells : List Nat := [56,57,58,59,60,61,62,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 56 1) (by decide)
  have hh := finite_xor_56_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_56_2 : ∀ b : Fin 256,
    Scalar.laneIndex 56 2 (Projection.scatter [7,14,21,28,35,42,49,56] (BitVec.ofNat 8 b.val)) ^^^ laneXor 56 2 =
    Scalar.laneIndex 56 2 (~~~(Projection.scatter [7,14,21,28,35,42,49,56] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_56_2 (p : Word) :
    Scalar.laneIndex 56 2 p ^^^ laneXor 56 2 = Scalar.laneIndex 56 2 (~~~p) := by
  let cells : List Nat := [7,14,21,28,35,42,49,56]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 56 2) (by decide)
  have hh := finite_xor_56_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_57_0 : ∀ b : Fin 256,
    Scalar.laneIndex 57 0 (Projection.scatter [1,9,17,25,33,41,49,57] (BitVec.ofNat 8 b.val)) ^^^ laneXor 57 0 =
    Scalar.laneIndex 57 0 (~~~(Projection.scatter [1,9,17,25,33,41,49,57] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_57_0 (p : Word) :
    Scalar.laneIndex 57 0 p ^^^ laneXor 57 0 = Scalar.laneIndex 57 0 (~~~p) := by
  let cells : List Nat := [1,9,17,25,33,41,49,57]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 57 0) (by decide)
  have hh := finite_xor_57_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_57_1 : ∀ b : Fin 256,
    Scalar.laneIndex 57 1 (Projection.scatter [56,57,58,59,60,61,62,63] (BitVec.ofNat 8 b.val)) ^^^ laneXor 57 1 =
    Scalar.laneIndex 57 1 (~~~(Projection.scatter [56,57,58,59,60,61,62,63] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_57_1 (p : Word) :
    Scalar.laneIndex 57 1 p ^^^ laneXor 57 1 = Scalar.laneIndex 57 1 (~~~p) := by
  let cells : List Nat := [56,57,58,59,60,61,62,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 57 1) (by decide)
  have hh := finite_xor_57_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_57_2 : ∀ b : Fin 256,
    Scalar.laneIndex 57 2 (Projection.scatter [15,22,29,36,43,50,57] (BitVec.ofNat 8 b.val)) ^^^ laneXor 57 2 =
    Scalar.laneIndex 57 2 (~~~(Projection.scatter [15,22,29,36,43,50,57] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_57_2 (p : Word) :
    Scalar.laneIndex 57 2 p ^^^ laneXor 57 2 = Scalar.laneIndex 57 2 (~~~p) := by
  let cells : List Nat := [15,22,29,36,43,50,57]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 57 2) (by decide)
  have hh := finite_xor_57_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_58_0 : ∀ b : Fin 256,
    Scalar.laneIndex 58 0 (Projection.scatter [2,10,18,26,34,42,50,58] (BitVec.ofNat 8 b.val)) ^^^ laneXor 58 0 =
    Scalar.laneIndex 58 0 (~~~(Projection.scatter [2,10,18,26,34,42,50,58] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_58_0 (p : Word) :
    Scalar.laneIndex 58 0 p ^^^ laneXor 58 0 = Scalar.laneIndex 58 0 (~~~p) := by
  let cells : List Nat := [2,10,18,26,34,42,50,58]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 58 0) (by decide)
  have hh := finite_xor_58_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_58_1 : ∀ b : Fin 256,
    Scalar.laneIndex 58 1 (Projection.scatter [56,57,58,59,60,61,62,63] (BitVec.ofNat 8 b.val)) ^^^ laneXor 58 1 =
    Scalar.laneIndex 58 1 (~~~(Projection.scatter [56,57,58,59,60,61,62,63] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_58_1 (p : Word) :
    Scalar.laneIndex 58 1 p ^^^ laneXor 58 1 = Scalar.laneIndex 58 1 (~~~p) := by
  let cells : List Nat := [56,57,58,59,60,61,62,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 58 1) (by decide)
  have hh := finite_xor_58_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_58_2 : ∀ b : Fin 256,
    Scalar.laneIndex 58 2 (Projection.scatter [23,30,37,40,44,49,51,58] (BitVec.ofNat 8 b.val)) ^^^ laneXor 58 2 =
    Scalar.laneIndex 58 2 (~~~(Projection.scatter [23,30,37,40,44,49,51,58] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_58_2 (p : Word) :
    Scalar.laneIndex 58 2 p ^^^ laneXor 58 2 = Scalar.laneIndex 58 2 (~~~p) := by
  let cells : List Nat := [23,30,37,40,44,49,51,58]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 58 2) (by decide)
  have hh := finite_xor_58_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_59_0 : ∀ b : Fin 256,
    Scalar.laneIndex 59 0 (Projection.scatter [3,11,19,27,35,43,51,59] (BitVec.ofNat 8 b.val)) ^^^ laneXor 59 0 =
    Scalar.laneIndex 59 0 (~~~(Projection.scatter [3,11,19,27,35,43,51,59] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_59_0 (p : Word) :
    Scalar.laneIndex 59 0 p ^^^ laneXor 59 0 = Scalar.laneIndex 59 0 (~~~p) := by
  let cells : List Nat := [3,11,19,27,35,43,51,59]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 59 0) (by decide)
  have hh := finite_xor_59_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_59_1 : ∀ b : Fin 256,
    Scalar.laneIndex 59 1 (Projection.scatter [56,57,58,59,60,61,62,63] (BitVec.ofNat 8 b.val)) ^^^ laneXor 59 1 =
    Scalar.laneIndex 59 1 (~~~(Projection.scatter [56,57,58,59,60,61,62,63] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_59_1 (p : Word) :
    Scalar.laneIndex 59 1 p ^^^ laneXor 59 1 = Scalar.laneIndex 59 1 (~~~p) := by
  let cells : List Nat := [56,57,58,59,60,61,62,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 59 1) (by decide)
  have hh := finite_xor_59_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_59_2 : ∀ b : Fin 256,
    Scalar.laneIndex 59 2 (Projection.scatter [31,32,38,41,45,50,52,59] (BitVec.ofNat 8 b.val)) ^^^ laneXor 59 2 =
    Scalar.laneIndex 59 2 (~~~(Projection.scatter [31,32,38,41,45,50,52,59] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_59_2 (p : Word) :
    Scalar.laneIndex 59 2 p ^^^ laneXor 59 2 = Scalar.laneIndex 59 2 (~~~p) := by
  let cells : List Nat := [31,32,38,41,45,50,52,59]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 59 2) (by decide)
  have hh := finite_xor_59_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_60_0 : ∀ b : Fin 256,
    Scalar.laneIndex 60 0 (Projection.scatter [4,12,20,28,36,44,52,60] (BitVec.ofNat 8 b.val)) ^^^ laneXor 60 0 =
    Scalar.laneIndex 60 0 (~~~(Projection.scatter [4,12,20,28,36,44,52,60] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_60_0 (p : Word) :
    Scalar.laneIndex 60 0 p ^^^ laneXor 60 0 = Scalar.laneIndex 60 0 (~~~p) := by
  let cells : List Nat := [4,12,20,28,36,44,52,60]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 60 0) (by decide)
  have hh := finite_xor_60_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_60_1 : ∀ b : Fin 256,
    Scalar.laneIndex 60 1 (Projection.scatter [56,57,58,59,60,61,62,63] (BitVec.ofNat 8 b.val)) ^^^ laneXor 60 1 =
    Scalar.laneIndex 60 1 (~~~(Projection.scatter [56,57,58,59,60,61,62,63] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_60_1 (p : Word) :
    Scalar.laneIndex 60 1 p ^^^ laneXor 60 1 = Scalar.laneIndex 60 1 (~~~p) := by
  let cells : List Nat := [56,57,58,59,60,61,62,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 60 1) (by decide)
  have hh := finite_xor_60_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_60_2 : ∀ b : Fin 256,
    Scalar.laneIndex 60 2 (Projection.scatter [24,33,39,42,46,51,53,60] (BitVec.ofNat 8 b.val)) ^^^ laneXor 60 2 =
    Scalar.laneIndex 60 2 (~~~(Projection.scatter [24,33,39,42,46,51,53,60] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_60_2 (p : Word) :
    Scalar.laneIndex 60 2 p ^^^ laneXor 60 2 = Scalar.laneIndex 60 2 (~~~p) := by
  let cells : List Nat := [24,33,39,42,46,51,53,60]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 60 2) (by decide)
  have hh := finite_xor_60_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_61_0 : ∀ b : Fin 256,
    Scalar.laneIndex 61 0 (Projection.scatter [5,13,21,29,37,45,53,61] (BitVec.ofNat 8 b.val)) ^^^ laneXor 61 0 =
    Scalar.laneIndex 61 0 (~~~(Projection.scatter [5,13,21,29,37,45,53,61] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_61_0 (p : Word) :
    Scalar.laneIndex 61 0 p ^^^ laneXor 61 0 = Scalar.laneIndex 61 0 (~~~p) := by
  let cells : List Nat := [5,13,21,29,37,45,53,61]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 61 0) (by decide)
  have hh := finite_xor_61_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_61_1 : ∀ b : Fin 256,
    Scalar.laneIndex 61 1 (Projection.scatter [56,57,58,59,60,61,62,63] (BitVec.ofNat 8 b.val)) ^^^ laneXor 61 1 =
    Scalar.laneIndex 61 1 (~~~(Projection.scatter [56,57,58,59,60,61,62,63] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_61_1 (p : Word) :
    Scalar.laneIndex 61 1 p ^^^ laneXor 61 1 = Scalar.laneIndex 61 1 (~~~p) := by
  let cells : List Nat := [56,57,58,59,60,61,62,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 61 1) (by decide)
  have hh := finite_xor_61_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_61_2 : ∀ b : Fin 256,
    Scalar.laneIndex 61 2 (Projection.scatter [16,25,34,43,47,52,54,61] (BitVec.ofNat 8 b.val)) ^^^ laneXor 61 2 =
    Scalar.laneIndex 61 2 (~~~(Projection.scatter [16,25,34,43,47,52,54,61] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_61_2 (p : Word) :
    Scalar.laneIndex 61 2 p ^^^ laneXor 61 2 = Scalar.laneIndex 61 2 (~~~p) := by
  let cells : List Nat := [16,25,34,43,47,52,54,61]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 61 2) (by decide)
  have hh := finite_xor_61_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_62_0 : ∀ b : Fin 256,
    Scalar.laneIndex 62 0 (Projection.scatter [6,14,22,30,38,46,54,62] (BitVec.ofNat 8 b.val)) ^^^ laneXor 62 0 =
    Scalar.laneIndex 62 0 (~~~(Projection.scatter [6,14,22,30,38,46,54,62] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_62_0 (p : Word) :
    Scalar.laneIndex 62 0 p ^^^ laneXor 62 0 = Scalar.laneIndex 62 0 (~~~p) := by
  let cells : List Nat := [6,14,22,30,38,46,54,62]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 62 0) (by decide)
  have hh := finite_xor_62_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_62_1 : ∀ b : Fin 256,
    Scalar.laneIndex 62 1 (Projection.scatter [56,57,58,59,60,61,62,63] (BitVec.ofNat 8 b.val)) ^^^ laneXor 62 1 =
    Scalar.laneIndex 62 1 (~~~(Projection.scatter [56,57,58,59,60,61,62,63] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_62_1 (p : Word) :
    Scalar.laneIndex 62 1 p ^^^ laneXor 62 1 = Scalar.laneIndex 62 1 (~~~p) := by
  let cells : List Nat := [56,57,58,59,60,61,62,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 62 1) (by decide)
  have hh := finite_xor_62_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_62_2 : ∀ b : Fin 256,
    Scalar.laneIndex 62 2 (Projection.scatter [8,17,26,35,44,53,62] (BitVec.ofNat 8 b.val)) ^^^ laneXor 62 2 =
    Scalar.laneIndex 62 2 (~~~(Projection.scatter [8,17,26,35,44,53,62] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_62_2 (p : Word) :
    Scalar.laneIndex 62 2 p ^^^ laneXor 62 2 = Scalar.laneIndex 62 2 (~~~p) := by
  let cells : List Nat := [8,17,26,35,44,53,62]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 62 2) (by decide)
  have hh := finite_xor_62_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_63_0 : ∀ b : Fin 256,
    Scalar.laneIndex 63 0 (Projection.scatter [7,15,23,31,39,47,55,63] (BitVec.ofNat 8 b.val)) ^^^ laneXor 63 0 =
    Scalar.laneIndex 63 0 (~~~(Projection.scatter [7,15,23,31,39,47,55,63] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_63_0 (p : Word) :
    Scalar.laneIndex 63 0 p ^^^ laneXor 63 0 = Scalar.laneIndex 63 0 (~~~p) := by
  let cells : List Nat := [7,15,23,31,39,47,55,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 63 0) (by decide)
  have hh := finite_xor_63_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_63_1 : ∀ b : Fin 256,
    Scalar.laneIndex 63 1 (Projection.scatter [56,57,58,59,60,61,62,63] (BitVec.ofNat 8 b.val)) ^^^ laneXor 63 1 =
    Scalar.laneIndex 63 1 (~~~(Projection.scatter [56,57,58,59,60,61,62,63] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_63_1 (p : Word) :
    Scalar.laneIndex 63 1 p ^^^ laneXor 63 1 = Scalar.laneIndex 63 1 (~~~p) := by
  let cells : List Nat := [56,57,58,59,60,61,62,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 63 1) (by decide)
  have hh := finite_xor_63_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_63_2 : ∀ b : Fin 256,
    Scalar.laneIndex 63 2 (Projection.scatter [0,9,18,27,36,45,54,63] (BitVec.ofNat 8 b.val)) ^^^ laneXor 63 2 =
    Scalar.laneIndex 63 2 (~~~(Projection.scatter [0,9,18,27,36,45,54,63] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_63_2 (p : Word) :
    Scalar.laneIndex 63 2 p ^^^ laneXor 63 2 = Scalar.laneIndex 63 2 (~~~p) := by
  let cells : List Nat := [0,9,18,27,36,45,54,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 63 2) (by decide)
  have hh := finite_xor_63_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

end NeuralReversi.Solve1.Neon

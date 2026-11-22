import NeuralReversi.Solve1.NeonModel
import NeuralReversi.Solve1.NeonProofs.Part1
namespace NeuralReversi.Solve1.Neon
set_option maxRecDepth 100000
set_option maxHeartbeats 0
private theorem finite_xor_16_0 : ∀ b : Fin 256,
    Scalar.laneIndex 16 0 (Projection.scatter [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 b.val)) ^^^ laneXor 16 0 =
    Scalar.laneIndex 16 0 (~~~(Projection.scatter [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_16_0 (p : Word) :
    Scalar.laneIndex 16 0 p ^^^ laneXor 16 0 = Scalar.laneIndex 16 0 (~~~p) := by
  let cells : List Nat := [0,8,16,24,32,40,48,56]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 16 0) (by decide)
  have hh := finite_xor_16_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_16_1 : ∀ b : Fin 256,
    Scalar.laneIndex 16 1 (Projection.scatter [16,17,18,19,20,21,22,23] (BitVec.ofNat 8 b.val)) ^^^ laneXor 16 1 =
    Scalar.laneIndex 16 1 (~~~(Projection.scatter [16,17,18,19,20,21,22,23] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_16_1 (p : Word) :
    Scalar.laneIndex 16 1 p ^^^ laneXor 16 1 = Scalar.laneIndex 16 1 (~~~p) := by
  let cells : List Nat := [16,17,18,19,20,21,22,23]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 16 1) (by decide)
  have hh := finite_xor_16_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_16_2 : ∀ b : Fin 256,
    Scalar.laneIndex 16 2 (Projection.scatter [2,9,16,25,34,43,52,61] (BitVec.ofNat 8 b.val)) ^^^ laneXor 16 2 =
    Scalar.laneIndex 16 2 (~~~(Projection.scatter [2,9,16,25,34,43,52,61] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_16_2 (p : Word) :
    Scalar.laneIndex 16 2 p ^^^ laneXor 16 2 = Scalar.laneIndex 16 2 (~~~p) := by
  let cells : List Nat := [2,9,16,25,34,43,52,61]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 16 2) (by decide)
  have hh := finite_xor_16_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_17_0 : ∀ b : Fin 256,
    Scalar.laneIndex 17 0 (Projection.scatter [1,9,17,25,33,41,49,57] (BitVec.ofNat 8 b.val)) ^^^ laneXor 17 0 =
    Scalar.laneIndex 17 0 (~~~(Projection.scatter [1,9,17,25,33,41,49,57] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_17_0 (p : Word) :
    Scalar.laneIndex 17 0 p ^^^ laneXor 17 0 = Scalar.laneIndex 17 0 (~~~p) := by
  let cells : List Nat := [1,9,17,25,33,41,49,57]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 17 0) (by decide)
  have hh := finite_xor_17_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_17_1 : ∀ b : Fin 256,
    Scalar.laneIndex 17 1 (Projection.scatter [16,17,18,19,20,21,22,23] (BitVec.ofNat 8 b.val)) ^^^ laneXor 17 1 =
    Scalar.laneIndex 17 1 (~~~(Projection.scatter [16,17,18,19,20,21,22,23] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_17_1 (p : Word) :
    Scalar.laneIndex 17 1 p ^^^ laneXor 17 1 = Scalar.laneIndex 17 1 (~~~p) := by
  let cells : List Nat := [16,17,18,19,20,21,22,23]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 17 1) (by decide)
  have hh := finite_xor_17_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_17_2 : ∀ b : Fin 256,
    Scalar.laneIndex 17 2 (Projection.scatter [3,10,17,26,35,44,53,62] (BitVec.ofNat 8 b.val)) ^^^ laneXor 17 2 =
    Scalar.laneIndex 17 2 (~~~(Projection.scatter [3,10,17,26,35,44,53,62] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_17_2 (p : Word) :
    Scalar.laneIndex 17 2 p ^^^ laneXor 17 2 = Scalar.laneIndex 17 2 (~~~p) := by
  let cells : List Nat := [3,10,17,26,35,44,53,62]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 17 2) (by decide)
  have hh := finite_xor_17_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_18_0 : ∀ b : Fin 256,
    Scalar.laneIndex 18 0 (Projection.scatter [2,10,18,26,34,42,50,58] (BitVec.ofNat 8 b.val)) ^^^ laneXor 18 0 =
    Scalar.laneIndex 18 0 (~~~(Projection.scatter [2,10,18,26,34,42,50,58] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_18_0 (p : Word) :
    Scalar.laneIndex 18 0 p ^^^ laneXor 18 0 = Scalar.laneIndex 18 0 (~~~p) := by
  let cells : List Nat := [2,10,18,26,34,42,50,58]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 18 0) (by decide)
  have hh := finite_xor_18_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_18_1 : ∀ b : Fin 256,
    Scalar.laneIndex 18 1 (Projection.scatter [16,17,18,19,20,21,22,23] (BitVec.ofNat 8 b.val)) ^^^ laneXor 18 1 =
    Scalar.laneIndex 18 1 (~~~(Projection.scatter [16,17,18,19,20,21,22,23] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_18_1 (p : Word) :
    Scalar.laneIndex 18 1 p ^^^ laneXor 18 1 = Scalar.laneIndex 18 1 (~~~p) := by
  let cells : List Nat := [16,17,18,19,20,21,22,23]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 18 1) (by decide)
  have hh := finite_xor_18_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_18_2 : ∀ b : Fin 256,
    Scalar.laneIndex 18 2 (Projection.scatter [4,11,18,25,32] (BitVec.ofNat 8 b.val)) ^^^ laneXor 18 2 =
    Scalar.laneIndex 18 2 (~~~(Projection.scatter [4,11,18,25,32] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_18_2 (p : Word) :
    Scalar.laneIndex 18 2 p ^^^ laneXor 18 2 = Scalar.laneIndex 18 2 (~~~p) := by
  let cells : List Nat := [4,11,18,25,32]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 18 2) (by decide)
  have hh := finite_xor_18_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_18_3 : ∀ b : Fin 256,
    Scalar.laneIndex 18 3 (Projection.scatter [0,9,18,27,36,45,54,63] (BitVec.ofNat 8 b.val)) ^^^ laneXor 18 3 =
    Scalar.laneIndex 18 3 (~~~(Projection.scatter [0,9,18,27,36,45,54,63] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_18_3 (p : Word) :
    Scalar.laneIndex 18 3 p ^^^ laneXor 18 3 = Scalar.laneIndex 18 3 (~~~p) := by
  let cells : List Nat := [0,9,18,27,36,45,54,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 18 3) (by decide)
  have hh := finite_xor_18_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_19_0 : ∀ b : Fin 256,
    Scalar.laneIndex 19 0 (Projection.scatter [3,11,19,27,35,43,51,59] (BitVec.ofNat 8 b.val)) ^^^ laneXor 19 0 =
    Scalar.laneIndex 19 0 (~~~(Projection.scatter [3,11,19,27,35,43,51,59] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_19_0 (p : Word) :
    Scalar.laneIndex 19 0 p ^^^ laneXor 19 0 = Scalar.laneIndex 19 0 (~~~p) := by
  let cells : List Nat := [3,11,19,27,35,43,51,59]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 19 0) (by decide)
  have hh := finite_xor_19_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_19_1 : ∀ b : Fin 256,
    Scalar.laneIndex 19 1 (Projection.scatter [16,17,18,19,20,21,22,23] (BitVec.ofNat 8 b.val)) ^^^ laneXor 19 1 =
    Scalar.laneIndex 19 1 (~~~(Projection.scatter [16,17,18,19,20,21,22,23] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_19_1 (p : Word) :
    Scalar.laneIndex 19 1 p ^^^ laneXor 19 1 = Scalar.laneIndex 19 1 (~~~p) := by
  let cells : List Nat := [16,17,18,19,20,21,22,23]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 19 1) (by decide)
  have hh := finite_xor_19_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_19_2 : ∀ b : Fin 256,
    Scalar.laneIndex 19 2 (Projection.scatter [5,12,19,26,33,40] (BitVec.ofNat 8 b.val)) ^^^ laneXor 19 2 =
    Scalar.laneIndex 19 2 (~~~(Projection.scatter [5,12,19,26,33,40] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_19_2 (p : Word) :
    Scalar.laneIndex 19 2 p ^^^ laneXor 19 2 = Scalar.laneIndex 19 2 (~~~p) := by
  let cells : List Nat := [5,12,19,26,33,40]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 19 2) (by decide)
  have hh := finite_xor_19_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_19_3 : ∀ b : Fin 256,
    Scalar.laneIndex 19 3 (Projection.scatter [1,10,19,28,37,46,55] (BitVec.ofNat 8 b.val)) ^^^ laneXor 19 3 =
    Scalar.laneIndex 19 3 (~~~(Projection.scatter [1,10,19,28,37,46,55] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_19_3 (p : Word) :
    Scalar.laneIndex 19 3 p ^^^ laneXor 19 3 = Scalar.laneIndex 19 3 (~~~p) := by
  let cells : List Nat := [1,10,19,28,37,46,55]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 19 3) (by decide)
  have hh := finite_xor_19_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_20_0 : ∀ b : Fin 256,
    Scalar.laneIndex 20 0 (Projection.scatter [4,12,20,28,36,44,52,60] (BitVec.ofNat 8 b.val)) ^^^ laneXor 20 0 =
    Scalar.laneIndex 20 0 (~~~(Projection.scatter [4,12,20,28,36,44,52,60] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_20_0 (p : Word) :
    Scalar.laneIndex 20 0 p ^^^ laneXor 20 0 = Scalar.laneIndex 20 0 (~~~p) := by
  let cells : List Nat := [4,12,20,28,36,44,52,60]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 20 0) (by decide)
  have hh := finite_xor_20_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_20_1 : ∀ b : Fin 256,
    Scalar.laneIndex 20 1 (Projection.scatter [16,17,18,19,20,21,22,23] (BitVec.ofNat 8 b.val)) ^^^ laneXor 20 1 =
    Scalar.laneIndex 20 1 (~~~(Projection.scatter [16,17,18,19,20,21,22,23] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_20_1 (p : Word) :
    Scalar.laneIndex 20 1 p ^^^ laneXor 20 1 = Scalar.laneIndex 20 1 (~~~p) := by
  let cells : List Nat := [16,17,18,19,20,21,22,23]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 20 1) (by decide)
  have hh := finite_xor_20_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_20_2 : ∀ b : Fin 256,
    Scalar.laneIndex 20 2 (Projection.scatter [6,13,20,27,34,41,48] (BitVec.ofNat 8 b.val)) ^^^ laneXor 20 2 =
    Scalar.laneIndex 20 2 (~~~(Projection.scatter [6,13,20,27,34,41,48] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_20_2 (p : Word) :
    Scalar.laneIndex 20 2 p ^^^ laneXor 20 2 = Scalar.laneIndex 20 2 (~~~p) := by
  let cells : List Nat := [6,13,20,27,34,41,48]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 20 2) (by decide)
  have hh := finite_xor_20_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_20_3 : ∀ b : Fin 256,
    Scalar.laneIndex 20 3 (Projection.scatter [2,11,20,29,38,47] (BitVec.ofNat 8 b.val)) ^^^ laneXor 20 3 =
    Scalar.laneIndex 20 3 (~~~(Projection.scatter [2,11,20,29,38,47] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_20_3 (p : Word) :
    Scalar.laneIndex 20 3 p ^^^ laneXor 20 3 = Scalar.laneIndex 20 3 (~~~p) := by
  let cells : List Nat := [2,11,20,29,38,47]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 20 3) (by decide)
  have hh := finite_xor_20_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_21_0 : ∀ b : Fin 256,
    Scalar.laneIndex 21 0 (Projection.scatter [5,13,21,29,37,45,53,61] (BitVec.ofNat 8 b.val)) ^^^ laneXor 21 0 =
    Scalar.laneIndex 21 0 (~~~(Projection.scatter [5,13,21,29,37,45,53,61] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_21_0 (p : Word) :
    Scalar.laneIndex 21 0 p ^^^ laneXor 21 0 = Scalar.laneIndex 21 0 (~~~p) := by
  let cells : List Nat := [5,13,21,29,37,45,53,61]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 21 0) (by decide)
  have hh := finite_xor_21_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_21_1 : ∀ b : Fin 256,
    Scalar.laneIndex 21 1 (Projection.scatter [16,17,18,19,20,21,22,23] (BitVec.ofNat 8 b.val)) ^^^ laneXor 21 1 =
    Scalar.laneIndex 21 1 (~~~(Projection.scatter [16,17,18,19,20,21,22,23] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_21_1 (p : Word) :
    Scalar.laneIndex 21 1 p ^^^ laneXor 21 1 = Scalar.laneIndex 21 1 (~~~p) := by
  let cells : List Nat := [16,17,18,19,20,21,22,23]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 21 1) (by decide)
  have hh := finite_xor_21_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_21_2 : ∀ b : Fin 256,
    Scalar.laneIndex 21 2 (Projection.scatter [7,14,21,28,35,42,49,56] (BitVec.ofNat 8 b.val)) ^^^ laneXor 21 2 =
    Scalar.laneIndex 21 2 (~~~(Projection.scatter [7,14,21,28,35,42,49,56] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_21_2 (p : Word) :
    Scalar.laneIndex 21 2 p ^^^ laneXor 21 2 = Scalar.laneIndex 21 2 (~~~p) := by
  let cells : List Nat := [7,14,21,28,35,42,49,56]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 21 2) (by decide)
  have hh := finite_xor_21_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_21_3 : ∀ b : Fin 256,
    Scalar.laneIndex 21 3 (Projection.scatter [3,12,21,30,39] (BitVec.ofNat 8 b.val)) ^^^ laneXor 21 3 =
    Scalar.laneIndex 21 3 (~~~(Projection.scatter [3,12,21,30,39] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_21_3 (p : Word) :
    Scalar.laneIndex 21 3 p ^^^ laneXor 21 3 = Scalar.laneIndex 21 3 (~~~p) := by
  let cells : List Nat := [3,12,21,30,39]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 21 3) (by decide)
  have hh := finite_xor_21_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_22_0 : ∀ b : Fin 256,
    Scalar.laneIndex 22 0 (Projection.scatter [6,14,22,30,38,46,54,62] (BitVec.ofNat 8 b.val)) ^^^ laneXor 22 0 =
    Scalar.laneIndex 22 0 (~~~(Projection.scatter [6,14,22,30,38,46,54,62] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_22_0 (p : Word) :
    Scalar.laneIndex 22 0 p ^^^ laneXor 22 0 = Scalar.laneIndex 22 0 (~~~p) := by
  let cells : List Nat := [6,14,22,30,38,46,54,62]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 22 0) (by decide)
  have hh := finite_xor_22_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_22_1 : ∀ b : Fin 256,
    Scalar.laneIndex 22 1 (Projection.scatter [16,17,18,19,20,21,22,23] (BitVec.ofNat 8 b.val)) ^^^ laneXor 22 1 =
    Scalar.laneIndex 22 1 (~~~(Projection.scatter [16,17,18,19,20,21,22,23] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_22_1 (p : Word) :
    Scalar.laneIndex 22 1 p ^^^ laneXor 22 1 = Scalar.laneIndex 22 1 (~~~p) := by
  let cells : List Nat := [16,17,18,19,20,21,22,23]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 22 1) (by decide)
  have hh := finite_xor_22_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_22_2 : ∀ b : Fin 256,
    Scalar.laneIndex 22 2 (Projection.scatter [4,13,22,29,36,43,50,57] (BitVec.ofNat 8 b.val)) ^^^ laneXor 22 2 =
    Scalar.laneIndex 22 2 (~~~(Projection.scatter [4,13,22,29,36,43,50,57] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_22_2 (p : Word) :
    Scalar.laneIndex 22 2 p ^^^ laneXor 22 2 = Scalar.laneIndex 22 2 (~~~p) := by
  let cells : List Nat := [4,13,22,29,36,43,50,57]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 22 2) (by decide)
  have hh := finite_xor_22_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_23_0 : ∀ b : Fin 256,
    Scalar.laneIndex 23 0 (Projection.scatter [7,15,23,31,39,47,55,63] (BitVec.ofNat 8 b.val)) ^^^ laneXor 23 0 =
    Scalar.laneIndex 23 0 (~~~(Projection.scatter [7,15,23,31,39,47,55,63] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_23_0 (p : Word) :
    Scalar.laneIndex 23 0 p ^^^ laneXor 23 0 = Scalar.laneIndex 23 0 (~~~p) := by
  let cells : List Nat := [7,15,23,31,39,47,55,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 23 0) (by decide)
  have hh := finite_xor_23_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_23_1 : ∀ b : Fin 256,
    Scalar.laneIndex 23 1 (Projection.scatter [16,17,18,19,20,21,22,23] (BitVec.ofNat 8 b.val)) ^^^ laneXor 23 1 =
    Scalar.laneIndex 23 1 (~~~(Projection.scatter [16,17,18,19,20,21,22,23] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_23_1 (p : Word) :
    Scalar.laneIndex 23 1 p ^^^ laneXor 23 1 = Scalar.laneIndex 23 1 (~~~p) := by
  let cells : List Nat := [16,17,18,19,20,21,22,23]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 23 1) (by decide)
  have hh := finite_xor_23_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_23_2 : ∀ b : Fin 256,
    Scalar.laneIndex 23 2 (Projection.scatter [5,14,23,30,37,44,51,58] (BitVec.ofNat 8 b.val)) ^^^ laneXor 23 2 =
    Scalar.laneIndex 23 2 (~~~(Projection.scatter [5,14,23,30,37,44,51,58] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_23_2 (p : Word) :
    Scalar.laneIndex 23 2 p ^^^ laneXor 23 2 = Scalar.laneIndex 23 2 (~~~p) := by
  let cells : List Nat := [5,14,23,30,37,44,51,58]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 23 2) (by decide)
  have hh := finite_xor_23_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

end NeuralReversi.Solve1.Neon

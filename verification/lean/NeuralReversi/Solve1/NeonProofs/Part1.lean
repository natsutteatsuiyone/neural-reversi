import NeuralReversi.Solve1.NeonModel
import NeuralReversi.Solve1.NeonProofs.Part0
namespace NeuralReversi.Solve1.Neon
set_option maxRecDepth 100000
set_option maxHeartbeats 0
private theorem finite_xor_8_0 : ∀ b : Fin 256,
    Scalar.laneIndex 8 0 (Projection.scatter [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 b.val)) ^^^ laneXor 8 0 =
    Scalar.laneIndex 8 0 (~~~(Projection.scatter [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_8_0 (p : Word) :
    Scalar.laneIndex 8 0 p ^^^ laneXor 8 0 = Scalar.laneIndex 8 0 (~~~p) := by
  let cells : List Nat := [0,8,16,24,32,40,48,56]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 8 0) (by decide)
  have hh := finite_xor_8_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_8_1 : ∀ b : Fin 256,
    Scalar.laneIndex 8 1 (Projection.scatter [8,9,10,11,12,13,14,15] (BitVec.ofNat 8 b.val)) ^^^ laneXor 8 1 =
    Scalar.laneIndex 8 1 (~~~(Projection.scatter [8,9,10,11,12,13,14,15] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_8_1 (p : Word) :
    Scalar.laneIndex 8 1 p ^^^ laneXor 8 1 = Scalar.laneIndex 8 1 (~~~p) := by
  let cells : List Nat := [8,9,10,11,12,13,14,15]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 8 1) (by decide)
  have hh := finite_xor_8_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_8_2 : ∀ b : Fin 256,
    Scalar.laneIndex 8 2 (Projection.scatter [8,17,26,35,44,53,62] (BitVec.ofNat 8 b.val)) ^^^ laneXor 8 2 =
    Scalar.laneIndex 8 2 (~~~(Projection.scatter [8,17,26,35,44,53,62] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_8_2 (p : Word) :
    Scalar.laneIndex 8 2 p ^^^ laneXor 8 2 = Scalar.laneIndex 8 2 (~~~p) := by
  let cells : List Nat := [8,17,26,35,44,53,62]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 8 2) (by decide)
  have hh := finite_xor_8_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_9_0 : ∀ b : Fin 256,
    Scalar.laneIndex 9 0 (Projection.scatter [1,9,17,25,33,41,49,57] (BitVec.ofNat 8 b.val)) ^^^ laneXor 9 0 =
    Scalar.laneIndex 9 0 (~~~(Projection.scatter [1,9,17,25,33,41,49,57] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_9_0 (p : Word) :
    Scalar.laneIndex 9 0 p ^^^ laneXor 9 0 = Scalar.laneIndex 9 0 (~~~p) := by
  let cells : List Nat := [1,9,17,25,33,41,49,57]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 9 0) (by decide)
  have hh := finite_xor_9_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_9_1 : ∀ b : Fin 256,
    Scalar.laneIndex 9 1 (Projection.scatter [8,9,10,11,12,13,14,15] (BitVec.ofNat 8 b.val)) ^^^ laneXor 9 1 =
    Scalar.laneIndex 9 1 (~~~(Projection.scatter [8,9,10,11,12,13,14,15] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_9_1 (p : Word) :
    Scalar.laneIndex 9 1 p ^^^ laneXor 9 1 = Scalar.laneIndex 9 1 (~~~p) := by
  let cells : List Nat := [8,9,10,11,12,13,14,15]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 9 1) (by decide)
  have hh := finite_xor_9_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_9_2 : ∀ b : Fin 256,
    Scalar.laneIndex 9 2 (Projection.scatter [0,9,18,27,36,45,54,63] (BitVec.ofNat 8 b.val)) ^^^ laneXor 9 2 =
    Scalar.laneIndex 9 2 (~~~(Projection.scatter [0,9,18,27,36,45,54,63] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_9_2 (p : Word) :
    Scalar.laneIndex 9 2 p ^^^ laneXor 9 2 = Scalar.laneIndex 9 2 (~~~p) := by
  let cells : List Nat := [0,9,18,27,36,45,54,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 9 2) (by decide)
  have hh := finite_xor_9_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_10_0 : ∀ b : Fin 256,
    Scalar.laneIndex 10 0 (Projection.scatter [2,10,18,26,34,42,50,58] (BitVec.ofNat 8 b.val)) ^^^ laneXor 10 0 =
    Scalar.laneIndex 10 0 (~~~(Projection.scatter [2,10,18,26,34,42,50,58] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_10_0 (p : Word) :
    Scalar.laneIndex 10 0 p ^^^ laneXor 10 0 = Scalar.laneIndex 10 0 (~~~p) := by
  let cells : List Nat := [2,10,18,26,34,42,50,58]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 10 0) (by decide)
  have hh := finite_xor_10_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_10_1 : ∀ b : Fin 256,
    Scalar.laneIndex 10 1 (Projection.scatter [8,9,10,11,12,13,14,15] (BitVec.ofNat 8 b.val)) ^^^ laneXor 10 1 =
    Scalar.laneIndex 10 1 (~~~(Projection.scatter [8,9,10,11,12,13,14,15] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_10_1 (p : Word) :
    Scalar.laneIndex 10 1 p ^^^ laneXor 10 1 = Scalar.laneIndex 10 1 (~~~p) := by
  let cells : List Nat := [8,9,10,11,12,13,14,15]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 10 1) (by decide)
  have hh := finite_xor_10_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_10_2 : ∀ b : Fin 256,
    Scalar.laneIndex 10 2 (Projection.scatter [10,17,19,24,28,37,46,55] (BitVec.ofNat 8 b.val)) ^^^ laneXor 10 2 =
    Scalar.laneIndex 10 2 (~~~(Projection.scatter [10,17,19,24,28,37,46,55] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_10_2 (p : Word) :
    Scalar.laneIndex 10 2 p ^^^ laneXor 10 2 = Scalar.laneIndex 10 2 (~~~p) := by
  let cells : List Nat := [10,17,19,24,28,37,46,55]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 10 2) (by decide)
  have hh := finite_xor_10_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_11_0 : ∀ b : Fin 256,
    Scalar.laneIndex 11 0 (Projection.scatter [3,11,19,27,35,43,51,59] (BitVec.ofNat 8 b.val)) ^^^ laneXor 11 0 =
    Scalar.laneIndex 11 0 (~~~(Projection.scatter [3,11,19,27,35,43,51,59] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_11_0 (p : Word) :
    Scalar.laneIndex 11 0 p ^^^ laneXor 11 0 = Scalar.laneIndex 11 0 (~~~p) := by
  let cells : List Nat := [3,11,19,27,35,43,51,59]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 11 0) (by decide)
  have hh := finite_xor_11_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_11_1 : ∀ b : Fin 256,
    Scalar.laneIndex 11 1 (Projection.scatter [8,9,10,11,12,13,14,15] (BitVec.ofNat 8 b.val)) ^^^ laneXor 11 1 =
    Scalar.laneIndex 11 1 (~~~(Projection.scatter [8,9,10,11,12,13,14,15] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_11_1 (p : Word) :
    Scalar.laneIndex 11 1 p ^^^ laneXor 11 1 = Scalar.laneIndex 11 1 (~~~p) := by
  let cells : List Nat := [8,9,10,11,12,13,14,15]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 11 1) (by decide)
  have hh := finite_xor_11_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_11_2 : ∀ b : Fin 256,
    Scalar.laneIndex 11 2 (Projection.scatter [11,18,20,25,29,32,38,47] (BitVec.ofNat 8 b.val)) ^^^ laneXor 11 2 =
    Scalar.laneIndex 11 2 (~~~(Projection.scatter [11,18,20,25,29,32,38,47] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_11_2 (p : Word) :
    Scalar.laneIndex 11 2 p ^^^ laneXor 11 2 = Scalar.laneIndex 11 2 (~~~p) := by
  let cells : List Nat := [11,18,20,25,29,32,38,47]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 11 2) (by decide)
  have hh := finite_xor_11_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_12_0 : ∀ b : Fin 256,
    Scalar.laneIndex 12 0 (Projection.scatter [4,12,20,28,36,44,52,60] (BitVec.ofNat 8 b.val)) ^^^ laneXor 12 0 =
    Scalar.laneIndex 12 0 (~~~(Projection.scatter [4,12,20,28,36,44,52,60] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_12_0 (p : Word) :
    Scalar.laneIndex 12 0 p ^^^ laneXor 12 0 = Scalar.laneIndex 12 0 (~~~p) := by
  let cells : List Nat := [4,12,20,28,36,44,52,60]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 12 0) (by decide)
  have hh := finite_xor_12_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_12_1 : ∀ b : Fin 256,
    Scalar.laneIndex 12 1 (Projection.scatter [8,9,10,11,12,13,14,15] (BitVec.ofNat 8 b.val)) ^^^ laneXor 12 1 =
    Scalar.laneIndex 12 1 (~~~(Projection.scatter [8,9,10,11,12,13,14,15] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_12_1 (p : Word) :
    Scalar.laneIndex 12 1 p ^^^ laneXor 12 1 = Scalar.laneIndex 12 1 (~~~p) := by
  let cells : List Nat := [8,9,10,11,12,13,14,15]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 12 1) (by decide)
  have hh := finite_xor_12_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_12_2 : ∀ b : Fin 256,
    Scalar.laneIndex 12 2 (Projection.scatter [12,19,21,26,30,33,39,40] (BitVec.ofNat 8 b.val)) ^^^ laneXor 12 2 =
    Scalar.laneIndex 12 2 (~~~(Projection.scatter [12,19,21,26,30,33,39,40] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_12_2 (p : Word) :
    Scalar.laneIndex 12 2 p ^^^ laneXor 12 2 = Scalar.laneIndex 12 2 (~~~p) := by
  let cells : List Nat := [12,19,21,26,30,33,39,40]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 12 2) (by decide)
  have hh := finite_xor_12_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_13_0 : ∀ b : Fin 256,
    Scalar.laneIndex 13 0 (Projection.scatter [5,13,21,29,37,45,53,61] (BitVec.ofNat 8 b.val)) ^^^ laneXor 13 0 =
    Scalar.laneIndex 13 0 (~~~(Projection.scatter [5,13,21,29,37,45,53,61] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_13_0 (p : Word) :
    Scalar.laneIndex 13 0 p ^^^ laneXor 13 0 = Scalar.laneIndex 13 0 (~~~p) := by
  let cells : List Nat := [5,13,21,29,37,45,53,61]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 13 0) (by decide)
  have hh := finite_xor_13_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_13_1 : ∀ b : Fin 256,
    Scalar.laneIndex 13 1 (Projection.scatter [8,9,10,11,12,13,14,15] (BitVec.ofNat 8 b.val)) ^^^ laneXor 13 1 =
    Scalar.laneIndex 13 1 (~~~(Projection.scatter [8,9,10,11,12,13,14,15] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_13_1 (p : Word) :
    Scalar.laneIndex 13 1 p ^^^ laneXor 13 1 = Scalar.laneIndex 13 1 (~~~p) := by
  let cells : List Nat := [8,9,10,11,12,13,14,15]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 13 1) (by decide)
  have hh := finite_xor_13_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_13_2 : ∀ b : Fin 256,
    Scalar.laneIndex 13 2 (Projection.scatter [13,20,22,27,31,34,41,48] (BitVec.ofNat 8 b.val)) ^^^ laneXor 13 2 =
    Scalar.laneIndex 13 2 (~~~(Projection.scatter [13,20,22,27,31,34,41,48] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_13_2 (p : Word) :
    Scalar.laneIndex 13 2 p ^^^ laneXor 13 2 = Scalar.laneIndex 13 2 (~~~p) := by
  let cells : List Nat := [13,20,22,27,31,34,41,48]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 13 2) (by decide)
  have hh := finite_xor_13_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_14_0 : ∀ b : Fin 256,
    Scalar.laneIndex 14 0 (Projection.scatter [6,14,22,30,38,46,54,62] (BitVec.ofNat 8 b.val)) ^^^ laneXor 14 0 =
    Scalar.laneIndex 14 0 (~~~(Projection.scatter [6,14,22,30,38,46,54,62] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_14_0 (p : Word) :
    Scalar.laneIndex 14 0 p ^^^ laneXor 14 0 = Scalar.laneIndex 14 0 (~~~p) := by
  let cells : List Nat := [6,14,22,30,38,46,54,62]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 14 0) (by decide)
  have hh := finite_xor_14_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_14_1 : ∀ b : Fin 256,
    Scalar.laneIndex 14 1 (Projection.scatter [8,9,10,11,12,13,14,15] (BitVec.ofNat 8 b.val)) ^^^ laneXor 14 1 =
    Scalar.laneIndex 14 1 (~~~(Projection.scatter [8,9,10,11,12,13,14,15] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_14_1 (p : Word) :
    Scalar.laneIndex 14 1 p ^^^ laneXor 14 1 = Scalar.laneIndex 14 1 (~~~p) := by
  let cells : List Nat := [8,9,10,11,12,13,14,15]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 14 1) (by decide)
  have hh := finite_xor_14_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_14_2 : ∀ b : Fin 256,
    Scalar.laneIndex 14 2 (Projection.scatter [7,14,21,28,35,42,49,56] (BitVec.ofNat 8 b.val)) ^^^ laneXor 14 2 =
    Scalar.laneIndex 14 2 (~~~(Projection.scatter [7,14,21,28,35,42,49,56] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_14_2 (p : Word) :
    Scalar.laneIndex 14 2 p ^^^ laneXor 14 2 = Scalar.laneIndex 14 2 (~~~p) := by
  let cells : List Nat := [7,14,21,28,35,42,49,56]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 14 2) (by decide)
  have hh := finite_xor_14_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_15_0 : ∀ b : Fin 256,
    Scalar.laneIndex 15 0 (Projection.scatter [7,15,23,31,39,47,55,63] (BitVec.ofNat 8 b.val)) ^^^ laneXor 15 0 =
    Scalar.laneIndex 15 0 (~~~(Projection.scatter [7,15,23,31,39,47,55,63] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_15_0 (p : Word) :
    Scalar.laneIndex 15 0 p ^^^ laneXor 15 0 = Scalar.laneIndex 15 0 (~~~p) := by
  let cells : List Nat := [7,15,23,31,39,47,55,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 15 0) (by decide)
  have hh := finite_xor_15_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_15_1 : ∀ b : Fin 256,
    Scalar.laneIndex 15 1 (Projection.scatter [8,9,10,11,12,13,14,15] (BitVec.ofNat 8 b.val)) ^^^ laneXor 15 1 =
    Scalar.laneIndex 15 1 (~~~(Projection.scatter [8,9,10,11,12,13,14,15] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_15_1 (p : Word) :
    Scalar.laneIndex 15 1 p ^^^ laneXor 15 1 = Scalar.laneIndex 15 1 (~~~p) := by
  let cells : List Nat := [8,9,10,11,12,13,14,15]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 15 1) (by decide)
  have hh := finite_xor_15_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_15_2 : ∀ b : Fin 256,
    Scalar.laneIndex 15 2 (Projection.scatter [15,22,29,36,43,50,57] (BitVec.ofNat 8 b.val)) ^^^ laneXor 15 2 =
    Scalar.laneIndex 15 2 (~~~(Projection.scatter [15,22,29,36,43,50,57] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_15_2 (p : Word) :
    Scalar.laneIndex 15 2 p ^^^ laneXor 15 2 = Scalar.laneIndex 15 2 (~~~p) := by
  let cells : List Nat := [15,22,29,36,43,50,57]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 15 2) (by decide)
  have hh := finite_xor_15_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

end NeuralReversi.Solve1.Neon

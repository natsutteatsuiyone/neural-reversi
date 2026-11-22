import NeuralReversi.Solve1.NeonModel
import NeuralReversi.Solve1.NeonProofs.Part2
namespace NeuralReversi.Solve1.Neon
set_option maxRecDepth 100000
set_option maxHeartbeats 0
private theorem finite_xor_24_0 : ∀ b : Fin 256,
    Scalar.laneIndex 24 0 (Projection.scatter [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 b.val)) ^^^ laneXor 24 0 =
    Scalar.laneIndex 24 0 (~~~(Projection.scatter [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_24_0 (p : Word) :
    Scalar.laneIndex 24 0 p ^^^ laneXor 24 0 = Scalar.laneIndex 24 0 (~~~p) := by
  let cells : List Nat := [0,8,16,24,32,40,48,56]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 24 0) (by decide)
  have hh := finite_xor_24_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_24_1 : ∀ b : Fin 256,
    Scalar.laneIndex 24 1 (Projection.scatter [24,25,26,27,28,29,30,31] (BitVec.ofNat 8 b.val)) ^^^ laneXor 24 1 =
    Scalar.laneIndex 24 1 (~~~(Projection.scatter [24,25,26,27,28,29,30,31] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_24_1 (p : Word) :
    Scalar.laneIndex 24 1 p ^^^ laneXor 24 1 = Scalar.laneIndex 24 1 (~~~p) := by
  let cells : List Nat := [24,25,26,27,28,29,30,31]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 24 1) (by decide)
  have hh := finite_xor_24_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_24_2 : ∀ b : Fin 256,
    Scalar.laneIndex 24 2 (Projection.scatter [3,10,17,24,33,42,51,60] (BitVec.ofNat 8 b.val)) ^^^ laneXor 24 2 =
    Scalar.laneIndex 24 2 (~~~(Projection.scatter [3,10,17,24,33,42,51,60] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_24_2 (p : Word) :
    Scalar.laneIndex 24 2 p ^^^ laneXor 24 2 = Scalar.laneIndex 24 2 (~~~p) := by
  let cells : List Nat := [3,10,17,24,33,42,51,60]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 24 2) (by decide)
  have hh := finite_xor_24_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_25_0 : ∀ b : Fin 256,
    Scalar.laneIndex 25 0 (Projection.scatter [1,9,17,25,33,41,49,57] (BitVec.ofNat 8 b.val)) ^^^ laneXor 25 0 =
    Scalar.laneIndex 25 0 (~~~(Projection.scatter [1,9,17,25,33,41,49,57] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_25_0 (p : Word) :
    Scalar.laneIndex 25 0 p ^^^ laneXor 25 0 = Scalar.laneIndex 25 0 (~~~p) := by
  let cells : List Nat := [1,9,17,25,33,41,49,57]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 25 0) (by decide)
  have hh := finite_xor_25_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_25_1 : ∀ b : Fin 256,
    Scalar.laneIndex 25 1 (Projection.scatter [24,25,26,27,28,29,30,31] (BitVec.ofNat 8 b.val)) ^^^ laneXor 25 1 =
    Scalar.laneIndex 25 1 (~~~(Projection.scatter [24,25,26,27,28,29,30,31] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_25_1 (p : Word) :
    Scalar.laneIndex 25 1 p ^^^ laneXor 25 1 = Scalar.laneIndex 25 1 (~~~p) := by
  let cells : List Nat := [24,25,26,27,28,29,30,31]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 25 1) (by decide)
  have hh := finite_xor_25_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_25_2 : ∀ b : Fin 256,
    Scalar.laneIndex 25 2 (Projection.scatter [4,11,18,25,34,43,52,61] (BitVec.ofNat 8 b.val)) ^^^ laneXor 25 2 =
    Scalar.laneIndex 25 2 (~~~(Projection.scatter [4,11,18,25,34,43,52,61] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_25_2 (p : Word) :
    Scalar.laneIndex 25 2 p ^^^ laneXor 25 2 = Scalar.laneIndex 25 2 (~~~p) := by
  let cells : List Nat := [4,11,18,25,34,43,52,61]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 25 2) (by decide)
  have hh := finite_xor_25_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_26_0 : ∀ b : Fin 256,
    Scalar.laneIndex 26 0 (Projection.scatter [2,10,18,26,34,42,50,58] (BitVec.ofNat 8 b.val)) ^^^ laneXor 26 0 =
    Scalar.laneIndex 26 0 (~~~(Projection.scatter [2,10,18,26,34,42,50,58] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_26_0 (p : Word) :
    Scalar.laneIndex 26 0 p ^^^ laneXor 26 0 = Scalar.laneIndex 26 0 (~~~p) := by
  let cells : List Nat := [2,10,18,26,34,42,50,58]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 26 0) (by decide)
  have hh := finite_xor_26_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_26_1 : ∀ b : Fin 256,
    Scalar.laneIndex 26 1 (Projection.scatter [24,25,26,27,28,29,30,31] (BitVec.ofNat 8 b.val)) ^^^ laneXor 26 1 =
    Scalar.laneIndex 26 1 (~~~(Projection.scatter [24,25,26,27,28,29,30,31] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_26_1 (p : Word) :
    Scalar.laneIndex 26 1 p ^^^ laneXor 26 1 = Scalar.laneIndex 26 1 (~~~p) := by
  let cells : List Nat := [24,25,26,27,28,29,30,31]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 26 1) (by decide)
  have hh := finite_xor_26_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_26_2 : ∀ b : Fin 256,
    Scalar.laneIndex 26 2 (Projection.scatter [5,12,19,26,33,40] (BitVec.ofNat 8 b.val)) ^^^ laneXor 26 2 =
    Scalar.laneIndex 26 2 (~~~(Projection.scatter [5,12,19,26,33,40] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_26_2 (p : Word) :
    Scalar.laneIndex 26 2 p ^^^ laneXor 26 2 = Scalar.laneIndex 26 2 (~~~p) := by
  let cells : List Nat := [5,12,19,26,33,40]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 26 2) (by decide)
  have hh := finite_xor_26_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_26_3 : ∀ b : Fin 256,
    Scalar.laneIndex 26 3 (Projection.scatter [8,17,26,35,44,53,62] (BitVec.ofNat 8 b.val)) ^^^ laneXor 26 3 =
    Scalar.laneIndex 26 3 (~~~(Projection.scatter [8,17,26,35,44,53,62] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_26_3 (p : Word) :
    Scalar.laneIndex 26 3 p ^^^ laneXor 26 3 = Scalar.laneIndex 26 3 (~~~p) := by
  let cells : List Nat := [8,17,26,35,44,53,62]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 26 3) (by decide)
  have hh := finite_xor_26_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_27_0 : ∀ b : Fin 256,
    Scalar.laneIndex 27 0 (Projection.scatter [3,11,19,27,35,43,51,59] (BitVec.ofNat 8 b.val)) ^^^ laneXor 27 0 =
    Scalar.laneIndex 27 0 (~~~(Projection.scatter [3,11,19,27,35,43,51,59] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_27_0 (p : Word) :
    Scalar.laneIndex 27 0 p ^^^ laneXor 27 0 = Scalar.laneIndex 27 0 (~~~p) := by
  let cells : List Nat := [3,11,19,27,35,43,51,59]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 27 0) (by decide)
  have hh := finite_xor_27_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_27_1 : ∀ b : Fin 256,
    Scalar.laneIndex 27 1 (Projection.scatter [24,25,26,27,28,29,30,31] (BitVec.ofNat 8 b.val)) ^^^ laneXor 27 1 =
    Scalar.laneIndex 27 1 (~~~(Projection.scatter [24,25,26,27,28,29,30,31] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_27_1 (p : Word) :
    Scalar.laneIndex 27 1 p ^^^ laneXor 27 1 = Scalar.laneIndex 27 1 (~~~p) := by
  let cells : List Nat := [24,25,26,27,28,29,30,31]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 27 1) (by decide)
  have hh := finite_xor_27_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_27_2 : ∀ b : Fin 256,
    Scalar.laneIndex 27 2 (Projection.scatter [6,13,20,27,34,41,48] (BitVec.ofNat 8 b.val)) ^^^ laneXor 27 2 =
    Scalar.laneIndex 27 2 (~~~(Projection.scatter [6,13,20,27,34,41,48] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_27_2 (p : Word) :
    Scalar.laneIndex 27 2 p ^^^ laneXor 27 2 = Scalar.laneIndex 27 2 (~~~p) := by
  let cells : List Nat := [6,13,20,27,34,41,48]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 27 2) (by decide)
  have hh := finite_xor_27_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_27_3 : ∀ b : Fin 256,
    Scalar.laneIndex 27 3 (Projection.scatter [0,9,18,27,36,45,54,63] (BitVec.ofNat 8 b.val)) ^^^ laneXor 27 3 =
    Scalar.laneIndex 27 3 (~~~(Projection.scatter [0,9,18,27,36,45,54,63] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_27_3 (p : Word) :
    Scalar.laneIndex 27 3 p ^^^ laneXor 27 3 = Scalar.laneIndex 27 3 (~~~p) := by
  let cells : List Nat := [0,9,18,27,36,45,54,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 27 3) (by decide)
  have hh := finite_xor_27_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_28_0 : ∀ b : Fin 256,
    Scalar.laneIndex 28 0 (Projection.scatter [4,12,20,28,36,44,52,60] (BitVec.ofNat 8 b.val)) ^^^ laneXor 28 0 =
    Scalar.laneIndex 28 0 (~~~(Projection.scatter [4,12,20,28,36,44,52,60] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_28_0 (p : Word) :
    Scalar.laneIndex 28 0 p ^^^ laneXor 28 0 = Scalar.laneIndex 28 0 (~~~p) := by
  let cells : List Nat := [4,12,20,28,36,44,52,60]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 28 0) (by decide)
  have hh := finite_xor_28_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_28_1 : ∀ b : Fin 256,
    Scalar.laneIndex 28 1 (Projection.scatter [24,25,26,27,28,29,30,31] (BitVec.ofNat 8 b.val)) ^^^ laneXor 28 1 =
    Scalar.laneIndex 28 1 (~~~(Projection.scatter [24,25,26,27,28,29,30,31] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_28_1 (p : Word) :
    Scalar.laneIndex 28 1 p ^^^ laneXor 28 1 = Scalar.laneIndex 28 1 (~~~p) := by
  let cells : List Nat := [24,25,26,27,28,29,30,31]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 28 1) (by decide)
  have hh := finite_xor_28_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_28_2 : ∀ b : Fin 256,
    Scalar.laneIndex 28 2 (Projection.scatter [7,14,21,28,35,42,49,56] (BitVec.ofNat 8 b.val)) ^^^ laneXor 28 2 =
    Scalar.laneIndex 28 2 (~~~(Projection.scatter [7,14,21,28,35,42,49,56] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_28_2 (p : Word) :
    Scalar.laneIndex 28 2 p ^^^ laneXor 28 2 = Scalar.laneIndex 28 2 (~~~p) := by
  let cells : List Nat := [7,14,21,28,35,42,49,56]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 28 2) (by decide)
  have hh := finite_xor_28_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_28_3 : ∀ b : Fin 256,
    Scalar.laneIndex 28 3 (Projection.scatter [1,10,19,28,37,46,55] (BitVec.ofNat 8 b.val)) ^^^ laneXor 28 3 =
    Scalar.laneIndex 28 3 (~~~(Projection.scatter [1,10,19,28,37,46,55] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_28_3 (p : Word) :
    Scalar.laneIndex 28 3 p ^^^ laneXor 28 3 = Scalar.laneIndex 28 3 (~~~p) := by
  let cells : List Nat := [1,10,19,28,37,46,55]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 28 3) (by decide)
  have hh := finite_xor_28_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_29_0 : ∀ b : Fin 256,
    Scalar.laneIndex 29 0 (Projection.scatter [5,13,21,29,37,45,53,61] (BitVec.ofNat 8 b.val)) ^^^ laneXor 29 0 =
    Scalar.laneIndex 29 0 (~~~(Projection.scatter [5,13,21,29,37,45,53,61] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_29_0 (p : Word) :
    Scalar.laneIndex 29 0 p ^^^ laneXor 29 0 = Scalar.laneIndex 29 0 (~~~p) := by
  let cells : List Nat := [5,13,21,29,37,45,53,61]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 29 0) (by decide)
  have hh := finite_xor_29_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_29_1 : ∀ b : Fin 256,
    Scalar.laneIndex 29 1 (Projection.scatter [24,25,26,27,28,29,30,31] (BitVec.ofNat 8 b.val)) ^^^ laneXor 29 1 =
    Scalar.laneIndex 29 1 (~~~(Projection.scatter [24,25,26,27,28,29,30,31] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_29_1 (p : Word) :
    Scalar.laneIndex 29 1 p ^^^ laneXor 29 1 = Scalar.laneIndex 29 1 (~~~p) := by
  let cells : List Nat := [24,25,26,27,28,29,30,31]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 29 1) (by decide)
  have hh := finite_xor_29_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_29_2 : ∀ b : Fin 256,
    Scalar.laneIndex 29 2 (Projection.scatter [15,22,29,36,43,50,57] (BitVec.ofNat 8 b.val)) ^^^ laneXor 29 2 =
    Scalar.laneIndex 29 2 (~~~(Projection.scatter [15,22,29,36,43,50,57] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_29_2 (p : Word) :
    Scalar.laneIndex 29 2 p ^^^ laneXor 29 2 = Scalar.laneIndex 29 2 (~~~p) := by
  let cells : List Nat := [15,22,29,36,43,50,57]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 29 2) (by decide)
  have hh := finite_xor_29_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_29_3 : ∀ b : Fin 256,
    Scalar.laneIndex 29 3 (Projection.scatter [2,11,20,29,38,47] (BitVec.ofNat 8 b.val)) ^^^ laneXor 29 3 =
    Scalar.laneIndex 29 3 (~~~(Projection.scatter [2,11,20,29,38,47] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_29_3 (p : Word) :
    Scalar.laneIndex 29 3 p ^^^ laneXor 29 3 = Scalar.laneIndex 29 3 (~~~p) := by
  let cells : List Nat := [2,11,20,29,38,47]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 29 3) (by decide)
  have hh := finite_xor_29_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_30_0 : ∀ b : Fin 256,
    Scalar.laneIndex 30 0 (Projection.scatter [6,14,22,30,38,46,54,62] (BitVec.ofNat 8 b.val)) ^^^ laneXor 30 0 =
    Scalar.laneIndex 30 0 (~~~(Projection.scatter [6,14,22,30,38,46,54,62] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_30_0 (p : Word) :
    Scalar.laneIndex 30 0 p ^^^ laneXor 30 0 = Scalar.laneIndex 30 0 (~~~p) := by
  let cells : List Nat := [6,14,22,30,38,46,54,62]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 30 0) (by decide)
  have hh := finite_xor_30_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_30_1 : ∀ b : Fin 256,
    Scalar.laneIndex 30 1 (Projection.scatter [24,25,26,27,28,29,30,31] (BitVec.ofNat 8 b.val)) ^^^ laneXor 30 1 =
    Scalar.laneIndex 30 1 (~~~(Projection.scatter [24,25,26,27,28,29,30,31] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_30_1 (p : Word) :
    Scalar.laneIndex 30 1 p ^^^ laneXor 30 1 = Scalar.laneIndex 30 1 (~~~p) := by
  let cells : List Nat := [24,25,26,27,28,29,30,31]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 30 1) (by decide)
  have hh := finite_xor_30_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_30_2 : ∀ b : Fin 256,
    Scalar.laneIndex 30 2 (Projection.scatter [3,12,21,30,37,44,51,58] (BitVec.ofNat 8 b.val)) ^^^ laneXor 30 2 =
    Scalar.laneIndex 30 2 (~~~(Projection.scatter [3,12,21,30,37,44,51,58] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_30_2 (p : Word) :
    Scalar.laneIndex 30 2 p ^^^ laneXor 30 2 = Scalar.laneIndex 30 2 (~~~p) := by
  let cells : List Nat := [3,12,21,30,37,44,51,58]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 30 2) (by decide)
  have hh := finite_xor_30_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_31_0 : ∀ b : Fin 256,
    Scalar.laneIndex 31 0 (Projection.scatter [7,15,23,31,39,47,55,63] (BitVec.ofNat 8 b.val)) ^^^ laneXor 31 0 =
    Scalar.laneIndex 31 0 (~~~(Projection.scatter [7,15,23,31,39,47,55,63] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_31_0 (p : Word) :
    Scalar.laneIndex 31 0 p ^^^ laneXor 31 0 = Scalar.laneIndex 31 0 (~~~p) := by
  let cells : List Nat := [7,15,23,31,39,47,55,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 31 0) (by decide)
  have hh := finite_xor_31_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_31_1 : ∀ b : Fin 256,
    Scalar.laneIndex 31 1 (Projection.scatter [24,25,26,27,28,29,30,31] (BitVec.ofNat 8 b.val)) ^^^ laneXor 31 1 =
    Scalar.laneIndex 31 1 (~~~(Projection.scatter [24,25,26,27,28,29,30,31] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_31_1 (p : Word) :
    Scalar.laneIndex 31 1 p ^^^ laneXor 31 1 = Scalar.laneIndex 31 1 (~~~p) := by
  let cells : List Nat := [24,25,26,27,28,29,30,31]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 31 1) (by decide)
  have hh := finite_xor_31_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_31_2 : ∀ b : Fin 256,
    Scalar.laneIndex 31 2 (Projection.scatter [4,13,22,31,38,45,52,59] (BitVec.ofNat 8 b.val)) ^^^ laneXor 31 2 =
    Scalar.laneIndex 31 2 (~~~(Projection.scatter [4,13,22,31,38,45,52,59] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_31_2 (p : Word) :
    Scalar.laneIndex 31 2 p ^^^ laneXor 31 2 = Scalar.laneIndex 31 2 (~~~p) := by
  let cells : List Nat := [4,13,22,31,38,45,52,59]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 31 2) (by decide)
  have hh := finite_xor_31_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

end NeuralReversi.Solve1.Neon

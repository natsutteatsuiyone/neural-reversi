import NeuralReversi.Solve1.NeonModel
import NeuralReversi.Solve1.NeonProofs.Part3
namespace NeuralReversi.Solve1.Neon
set_option maxRecDepth 100000
set_option maxHeartbeats 0
private theorem finite_xor_32_0 : ∀ b : Fin 256,
    Scalar.laneIndex 32 0 (Projection.scatter [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 b.val)) ^^^ laneXor 32 0 =
    Scalar.laneIndex 32 0 (~~~(Projection.scatter [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_32_0 (p : Word) :
    Scalar.laneIndex 32 0 p ^^^ laneXor 32 0 = Scalar.laneIndex 32 0 (~~~p) := by
  let cells : List Nat := [0,8,16,24,32,40,48,56]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 32 0) (by decide)
  have hh := finite_xor_32_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_32_1 : ∀ b : Fin 256,
    Scalar.laneIndex 32 1 (Projection.scatter [32,33,34,35,36,37,38,39] (BitVec.ofNat 8 b.val)) ^^^ laneXor 32 1 =
    Scalar.laneIndex 32 1 (~~~(Projection.scatter [32,33,34,35,36,37,38,39] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_32_1 (p : Word) :
    Scalar.laneIndex 32 1 p ^^^ laneXor 32 1 = Scalar.laneIndex 32 1 (~~~p) := by
  let cells : List Nat := [32,33,34,35,36,37,38,39]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 32 1) (by decide)
  have hh := finite_xor_32_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_32_2 : ∀ b : Fin 256,
    Scalar.laneIndex 32 2 (Projection.scatter [4,11,18,25,32,41,50,59] (BitVec.ofNat 8 b.val)) ^^^ laneXor 32 2 =
    Scalar.laneIndex 32 2 (~~~(Projection.scatter [4,11,18,25,32,41,50,59] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_32_2 (p : Word) :
    Scalar.laneIndex 32 2 p ^^^ laneXor 32 2 = Scalar.laneIndex 32 2 (~~~p) := by
  let cells : List Nat := [4,11,18,25,32,41,50,59]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 32 2) (by decide)
  have hh := finite_xor_32_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_33_0 : ∀ b : Fin 256,
    Scalar.laneIndex 33 0 (Projection.scatter [1,9,17,25,33,41,49,57] (BitVec.ofNat 8 b.val)) ^^^ laneXor 33 0 =
    Scalar.laneIndex 33 0 (~~~(Projection.scatter [1,9,17,25,33,41,49,57] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_33_0 (p : Word) :
    Scalar.laneIndex 33 0 p ^^^ laneXor 33 0 = Scalar.laneIndex 33 0 (~~~p) := by
  let cells : List Nat := [1,9,17,25,33,41,49,57]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 33 0) (by decide)
  have hh := finite_xor_33_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_33_1 : ∀ b : Fin 256,
    Scalar.laneIndex 33 1 (Projection.scatter [32,33,34,35,36,37,38,39] (BitVec.ofNat 8 b.val)) ^^^ laneXor 33 1 =
    Scalar.laneIndex 33 1 (~~~(Projection.scatter [32,33,34,35,36,37,38,39] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_33_1 (p : Word) :
    Scalar.laneIndex 33 1 p ^^^ laneXor 33 1 = Scalar.laneIndex 33 1 (~~~p) := by
  let cells : List Nat := [32,33,34,35,36,37,38,39]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 33 1) (by decide)
  have hh := finite_xor_33_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_33_2 : ∀ b : Fin 256,
    Scalar.laneIndex 33 2 (Projection.scatter [5,12,19,26,33,42,51,60] (BitVec.ofNat 8 b.val)) ^^^ laneXor 33 2 =
    Scalar.laneIndex 33 2 (~~~(Projection.scatter [5,12,19,26,33,42,51,60] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_33_2 (p : Word) :
    Scalar.laneIndex 33 2 p ^^^ laneXor 33 2 = Scalar.laneIndex 33 2 (~~~p) := by
  let cells : List Nat := [5,12,19,26,33,42,51,60]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 33 2) (by decide)
  have hh := finite_xor_33_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_34_0 : ∀ b : Fin 256,
    Scalar.laneIndex 34 0 (Projection.scatter [2,10,18,26,34,42,50,58] (BitVec.ofNat 8 b.val)) ^^^ laneXor 34 0 =
    Scalar.laneIndex 34 0 (~~~(Projection.scatter [2,10,18,26,34,42,50,58] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_34_0 (p : Word) :
    Scalar.laneIndex 34 0 p ^^^ laneXor 34 0 = Scalar.laneIndex 34 0 (~~~p) := by
  let cells : List Nat := [2,10,18,26,34,42,50,58]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 34 0) (by decide)
  have hh := finite_xor_34_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_34_1 : ∀ b : Fin 256,
    Scalar.laneIndex 34 1 (Projection.scatter [32,33,34,35,36,37,38,39] (BitVec.ofNat 8 b.val)) ^^^ laneXor 34 1 =
    Scalar.laneIndex 34 1 (~~~(Projection.scatter [32,33,34,35,36,37,38,39] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_34_1 (p : Word) :
    Scalar.laneIndex 34 1 p ^^^ laneXor 34 1 = Scalar.laneIndex 34 1 (~~~p) := by
  let cells : List Nat := [32,33,34,35,36,37,38,39]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 34 1) (by decide)
  have hh := finite_xor_34_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_34_2 : ∀ b : Fin 256,
    Scalar.laneIndex 34 2 (Projection.scatter [6,13,20,27,34,41,48] (BitVec.ofNat 8 b.val)) ^^^ laneXor 34 2 =
    Scalar.laneIndex 34 2 (~~~(Projection.scatter [6,13,20,27,34,41,48] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_34_2 (p : Word) :
    Scalar.laneIndex 34 2 p ^^^ laneXor 34 2 = Scalar.laneIndex 34 2 (~~~p) := by
  let cells : List Nat := [6,13,20,27,34,41,48]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 34 2) (by decide)
  have hh := finite_xor_34_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_34_3 : ∀ b : Fin 256,
    Scalar.laneIndex 34 3 (Projection.scatter [16,25,34,43,52,61] (BitVec.ofNat 8 b.val)) ^^^ laneXor 34 3 =
    Scalar.laneIndex 34 3 (~~~(Projection.scatter [16,25,34,43,52,61] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_34_3 (p : Word) :
    Scalar.laneIndex 34 3 p ^^^ laneXor 34 3 = Scalar.laneIndex 34 3 (~~~p) := by
  let cells : List Nat := [16,25,34,43,52,61]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 34 3) (by decide)
  have hh := finite_xor_34_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_35_0 : ∀ b : Fin 256,
    Scalar.laneIndex 35 0 (Projection.scatter [3,11,19,27,35,43,51,59] (BitVec.ofNat 8 b.val)) ^^^ laneXor 35 0 =
    Scalar.laneIndex 35 0 (~~~(Projection.scatter [3,11,19,27,35,43,51,59] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_35_0 (p : Word) :
    Scalar.laneIndex 35 0 p ^^^ laneXor 35 0 = Scalar.laneIndex 35 0 (~~~p) := by
  let cells : List Nat := [3,11,19,27,35,43,51,59]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 35 0) (by decide)
  have hh := finite_xor_35_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_35_1 : ∀ b : Fin 256,
    Scalar.laneIndex 35 1 (Projection.scatter [32,33,34,35,36,37,38,39] (BitVec.ofNat 8 b.val)) ^^^ laneXor 35 1 =
    Scalar.laneIndex 35 1 (~~~(Projection.scatter [32,33,34,35,36,37,38,39] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_35_1 (p : Word) :
    Scalar.laneIndex 35 1 p ^^^ laneXor 35 1 = Scalar.laneIndex 35 1 (~~~p) := by
  let cells : List Nat := [32,33,34,35,36,37,38,39]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 35 1) (by decide)
  have hh := finite_xor_35_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_35_2 : ∀ b : Fin 256,
    Scalar.laneIndex 35 2 (Projection.scatter [7,14,21,28,35,42,49,56] (BitVec.ofNat 8 b.val)) ^^^ laneXor 35 2 =
    Scalar.laneIndex 35 2 (~~~(Projection.scatter [7,14,21,28,35,42,49,56] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_35_2 (p : Word) :
    Scalar.laneIndex 35 2 p ^^^ laneXor 35 2 = Scalar.laneIndex 35 2 (~~~p) := by
  let cells : List Nat := [7,14,21,28,35,42,49,56]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 35 2) (by decide)
  have hh := finite_xor_35_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_35_3 : ∀ b : Fin 256,
    Scalar.laneIndex 35 3 (Projection.scatter [8,17,26,35,44,53,62] (BitVec.ofNat 8 b.val)) ^^^ laneXor 35 3 =
    Scalar.laneIndex 35 3 (~~~(Projection.scatter [8,17,26,35,44,53,62] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_35_3 (p : Word) :
    Scalar.laneIndex 35 3 p ^^^ laneXor 35 3 = Scalar.laneIndex 35 3 (~~~p) := by
  let cells : List Nat := [8,17,26,35,44,53,62]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 35 3) (by decide)
  have hh := finite_xor_35_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_36_0 : ∀ b : Fin 256,
    Scalar.laneIndex 36 0 (Projection.scatter [4,12,20,28,36,44,52,60] (BitVec.ofNat 8 b.val)) ^^^ laneXor 36 0 =
    Scalar.laneIndex 36 0 (~~~(Projection.scatter [4,12,20,28,36,44,52,60] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_36_0 (p : Word) :
    Scalar.laneIndex 36 0 p ^^^ laneXor 36 0 = Scalar.laneIndex 36 0 (~~~p) := by
  let cells : List Nat := [4,12,20,28,36,44,52,60]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 36 0) (by decide)
  have hh := finite_xor_36_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_36_1 : ∀ b : Fin 256,
    Scalar.laneIndex 36 1 (Projection.scatter [32,33,34,35,36,37,38,39] (BitVec.ofNat 8 b.val)) ^^^ laneXor 36 1 =
    Scalar.laneIndex 36 1 (~~~(Projection.scatter [32,33,34,35,36,37,38,39] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_36_1 (p : Word) :
    Scalar.laneIndex 36 1 p ^^^ laneXor 36 1 = Scalar.laneIndex 36 1 (~~~p) := by
  let cells : List Nat := [32,33,34,35,36,37,38,39]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 36 1) (by decide)
  have hh := finite_xor_36_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_36_2 : ∀ b : Fin 256,
    Scalar.laneIndex 36 2 (Projection.scatter [15,22,29,36,43,50,57] (BitVec.ofNat 8 b.val)) ^^^ laneXor 36 2 =
    Scalar.laneIndex 36 2 (~~~(Projection.scatter [15,22,29,36,43,50,57] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_36_2 (p : Word) :
    Scalar.laneIndex 36 2 p ^^^ laneXor 36 2 = Scalar.laneIndex 36 2 (~~~p) := by
  let cells : List Nat := [15,22,29,36,43,50,57]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 36 2) (by decide)
  have hh := finite_xor_36_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_36_3 : ∀ b : Fin 256,
    Scalar.laneIndex 36 3 (Projection.scatter [0,9,18,27,36,45,54,63] (BitVec.ofNat 8 b.val)) ^^^ laneXor 36 3 =
    Scalar.laneIndex 36 3 (~~~(Projection.scatter [0,9,18,27,36,45,54,63] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_36_3 (p : Word) :
    Scalar.laneIndex 36 3 p ^^^ laneXor 36 3 = Scalar.laneIndex 36 3 (~~~p) := by
  let cells : List Nat := [0,9,18,27,36,45,54,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 36 3) (by decide)
  have hh := finite_xor_36_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_37_0 : ∀ b : Fin 256,
    Scalar.laneIndex 37 0 (Projection.scatter [5,13,21,29,37,45,53,61] (BitVec.ofNat 8 b.val)) ^^^ laneXor 37 0 =
    Scalar.laneIndex 37 0 (~~~(Projection.scatter [5,13,21,29,37,45,53,61] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_37_0 (p : Word) :
    Scalar.laneIndex 37 0 p ^^^ laneXor 37 0 = Scalar.laneIndex 37 0 (~~~p) := by
  let cells : List Nat := [5,13,21,29,37,45,53,61]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 37 0) (by decide)
  have hh := finite_xor_37_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_37_1 : ∀ b : Fin 256,
    Scalar.laneIndex 37 1 (Projection.scatter [32,33,34,35,36,37,38,39] (BitVec.ofNat 8 b.val)) ^^^ laneXor 37 1 =
    Scalar.laneIndex 37 1 (~~~(Projection.scatter [32,33,34,35,36,37,38,39] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_37_1 (p : Word) :
    Scalar.laneIndex 37 1 p ^^^ laneXor 37 1 = Scalar.laneIndex 37 1 (~~~p) := by
  let cells : List Nat := [32,33,34,35,36,37,38,39]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 37 1) (by decide)
  have hh := finite_xor_37_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_37_2 : ∀ b : Fin 256,
    Scalar.laneIndex 37 2 (Projection.scatter [23,30,37,44,51,58] (BitVec.ofNat 8 b.val)) ^^^ laneXor 37 2 =
    Scalar.laneIndex 37 2 (~~~(Projection.scatter [23,30,37,44,51,58] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_37_2 (p : Word) :
    Scalar.laneIndex 37 2 p ^^^ laneXor 37 2 = Scalar.laneIndex 37 2 (~~~p) := by
  let cells : List Nat := [23,30,37,44,51,58]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 37 2) (by decide)
  have hh := finite_xor_37_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_37_3 : ∀ b : Fin 256,
    Scalar.laneIndex 37 3 (Projection.scatter [1,10,19,28,37,46,55] (BitVec.ofNat 8 b.val)) ^^^ laneXor 37 3 =
    Scalar.laneIndex 37 3 (~~~(Projection.scatter [1,10,19,28,37,46,55] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_37_3 (p : Word) :
    Scalar.laneIndex 37 3 p ^^^ laneXor 37 3 = Scalar.laneIndex 37 3 (~~~p) := by
  let cells : List Nat := [1,10,19,28,37,46,55]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 37 3) (by decide)
  have hh := finite_xor_37_3 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_38_0 : ∀ b : Fin 256,
    Scalar.laneIndex 38 0 (Projection.scatter [6,14,22,30,38,46,54,62] (BitVec.ofNat 8 b.val)) ^^^ laneXor 38 0 =
    Scalar.laneIndex 38 0 (~~~(Projection.scatter [6,14,22,30,38,46,54,62] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_38_0 (p : Word) :
    Scalar.laneIndex 38 0 p ^^^ laneXor 38 0 = Scalar.laneIndex 38 0 (~~~p) := by
  let cells : List Nat := [6,14,22,30,38,46,54,62]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 38 0) (by decide)
  have hh := finite_xor_38_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_38_1 : ∀ b : Fin 256,
    Scalar.laneIndex 38 1 (Projection.scatter [32,33,34,35,36,37,38,39] (BitVec.ofNat 8 b.val)) ^^^ laneXor 38 1 =
    Scalar.laneIndex 38 1 (~~~(Projection.scatter [32,33,34,35,36,37,38,39] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_38_1 (p : Word) :
    Scalar.laneIndex 38 1 p ^^^ laneXor 38 1 = Scalar.laneIndex 38 1 (~~~p) := by
  let cells : List Nat := [32,33,34,35,36,37,38,39]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 38 1) (by decide)
  have hh := finite_xor_38_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_38_2 : ∀ b : Fin 256,
    Scalar.laneIndex 38 2 (Projection.scatter [2,11,20,29,38,45,52,59] (BitVec.ofNat 8 b.val)) ^^^ laneXor 38 2 =
    Scalar.laneIndex 38 2 (~~~(Projection.scatter [2,11,20,29,38,45,52,59] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_38_2 (p : Word) :
    Scalar.laneIndex 38 2 p ^^^ laneXor 38 2 = Scalar.laneIndex 38 2 (~~~p) := by
  let cells : List Nat := [2,11,20,29,38,45,52,59]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 38 2) (by decide)
  have hh := finite_xor_38_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_39_0 : ∀ b : Fin 256,
    Scalar.laneIndex 39 0 (Projection.scatter [7,15,23,31,39,47,55,63] (BitVec.ofNat 8 b.val)) ^^^ laneXor 39 0 =
    Scalar.laneIndex 39 0 (~~~(Projection.scatter [7,15,23,31,39,47,55,63] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_39_0 (p : Word) :
    Scalar.laneIndex 39 0 p ^^^ laneXor 39 0 = Scalar.laneIndex 39 0 (~~~p) := by
  let cells : List Nat := [7,15,23,31,39,47,55,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 39 0) (by decide)
  have hh := finite_xor_39_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_39_1 : ∀ b : Fin 256,
    Scalar.laneIndex 39 1 (Projection.scatter [32,33,34,35,36,37,38,39] (BitVec.ofNat 8 b.val)) ^^^ laneXor 39 1 =
    Scalar.laneIndex 39 1 (~~~(Projection.scatter [32,33,34,35,36,37,38,39] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_39_1 (p : Word) :
    Scalar.laneIndex 39 1 p ^^^ laneXor 39 1 = Scalar.laneIndex 39 1 (~~~p) := by
  let cells : List Nat := [32,33,34,35,36,37,38,39]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 39 1) (by decide)
  have hh := finite_xor_39_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_39_2 : ∀ b : Fin 256,
    Scalar.laneIndex 39 2 (Projection.scatter [3,12,21,30,39,46,53,60] (BitVec.ofNat 8 b.val)) ^^^ laneXor 39 2 =
    Scalar.laneIndex 39 2 (~~~(Projection.scatter [3,12,21,30,39,46,53,60] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_39_2 (p : Word) :
    Scalar.laneIndex 39 2 p ^^^ laneXor 39 2 = Scalar.laneIndex 39 2 (~~~p) := by
  let cells : List Nat := [3,12,21,30,39,46,53,60]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 39 2) (by decide)
  have hh := finite_xor_39_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

end NeuralReversi.Solve1.Neon

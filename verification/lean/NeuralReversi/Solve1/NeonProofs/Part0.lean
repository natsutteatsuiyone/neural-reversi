import NeuralReversi.Solve1.NeonModel
namespace NeuralReversi.Solve1.Neon
set_option maxRecDepth 100000
set_option maxHeartbeats 0
private theorem finite_xor_0_0 : ∀ b : Fin 256,
    Scalar.laneIndex 0 0 (Projection.scatter [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 b.val)) ^^^ laneXor 0 0 =
    Scalar.laneIndex 0 0 (~~~(Projection.scatter [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_0_0 (p : Word) :
    Scalar.laneIndex 0 0 p ^^^ laneXor 0 0 = Scalar.laneIndex 0 0 (~~~p) := by
  let cells : List Nat := [0,8,16,24,32,40,48,56]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 0 0) (by decide)
  have hh := finite_xor_0_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_0_1 : ∀ b : Fin 256,
    Scalar.laneIndex 0 1 (Projection.scatter [0,1,2,3,4,5,6,7] (BitVec.ofNat 8 b.val)) ^^^ laneXor 0 1 =
    Scalar.laneIndex 0 1 (~~~(Projection.scatter [0,1,2,3,4,5,6,7] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_0_1 (p : Word) :
    Scalar.laneIndex 0 1 p ^^^ laneXor 0 1 = Scalar.laneIndex 0 1 (~~~p) := by
  let cells : List Nat := [0,1,2,3,4,5,6,7]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 0 1) (by decide)
  have hh := finite_xor_0_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_0_2 : ∀ b : Fin 256,
    Scalar.laneIndex 0 2 (Projection.scatter [0,9,18,27,36,45,54,63] (BitVec.ofNat 8 b.val)) ^^^ laneXor 0 2 =
    Scalar.laneIndex 0 2 (~~~(Projection.scatter [0,9,18,27,36,45,54,63] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_0_2 (p : Word) :
    Scalar.laneIndex 0 2 p ^^^ laneXor 0 2 = Scalar.laneIndex 0 2 (~~~p) := by
  let cells : List Nat := [0,9,18,27,36,45,54,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 0 2) (by decide)
  have hh := finite_xor_0_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_1_0 : ∀ b : Fin 256,
    Scalar.laneIndex 1 0 (Projection.scatter [1,9,17,25,33,41,49,57] (BitVec.ofNat 8 b.val)) ^^^ laneXor 1 0 =
    Scalar.laneIndex 1 0 (~~~(Projection.scatter [1,9,17,25,33,41,49,57] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_1_0 (p : Word) :
    Scalar.laneIndex 1 0 p ^^^ laneXor 1 0 = Scalar.laneIndex 1 0 (~~~p) := by
  let cells : List Nat := [1,9,17,25,33,41,49,57]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 1 0) (by decide)
  have hh := finite_xor_1_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_1_1 : ∀ b : Fin 256,
    Scalar.laneIndex 1 1 (Projection.scatter [0,1,2,3,4,5,6,7] (BitVec.ofNat 8 b.val)) ^^^ laneXor 1 1 =
    Scalar.laneIndex 1 1 (~~~(Projection.scatter [0,1,2,3,4,5,6,7] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_1_1 (p : Word) :
    Scalar.laneIndex 1 1 p ^^^ laneXor 1 1 = Scalar.laneIndex 1 1 (~~~p) := by
  let cells : List Nat := [0,1,2,3,4,5,6,7]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 1 1) (by decide)
  have hh := finite_xor_1_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_1_2 : ∀ b : Fin 256,
    Scalar.laneIndex 1 2 (Projection.scatter [1,10,19,28,37,46,55] (BitVec.ofNat 8 b.val)) ^^^ laneXor 1 2 =
    Scalar.laneIndex 1 2 (~~~(Projection.scatter [1,10,19,28,37,46,55] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_1_2 (p : Word) :
    Scalar.laneIndex 1 2 p ^^^ laneXor 1 2 = Scalar.laneIndex 1 2 (~~~p) := by
  let cells : List Nat := [1,10,19,28,37,46,55]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 1 2) (by decide)
  have hh := finite_xor_1_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_2_0 : ∀ b : Fin 256,
    Scalar.laneIndex 2 0 (Projection.scatter [2,10,18,26,34,42,50,58] (BitVec.ofNat 8 b.val)) ^^^ laneXor 2 0 =
    Scalar.laneIndex 2 0 (~~~(Projection.scatter [2,10,18,26,34,42,50,58] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_2_0 (p : Word) :
    Scalar.laneIndex 2 0 p ^^^ laneXor 2 0 = Scalar.laneIndex 2 0 (~~~p) := by
  let cells : List Nat := [2,10,18,26,34,42,50,58]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 2 0) (by decide)
  have hh := finite_xor_2_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_2_1 : ∀ b : Fin 256,
    Scalar.laneIndex 2 1 (Projection.scatter [0,1,2,3,4,5,6,7] (BitVec.ofNat 8 b.val)) ^^^ laneXor 2 1 =
    Scalar.laneIndex 2 1 (~~~(Projection.scatter [0,1,2,3,4,5,6,7] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_2_1 (p : Word) :
    Scalar.laneIndex 2 1 p ^^^ laneXor 2 1 = Scalar.laneIndex 2 1 (~~~p) := by
  let cells : List Nat := [0,1,2,3,4,5,6,7]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 2 1) (by decide)
  have hh := finite_xor_2_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_2_2 : ∀ b : Fin 256,
    Scalar.laneIndex 2 2 (Projection.scatter [2,9,11,16,20,29,38,47] (BitVec.ofNat 8 b.val)) ^^^ laneXor 2 2 =
    Scalar.laneIndex 2 2 (~~~(Projection.scatter [2,9,11,16,20,29,38,47] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_2_2 (p : Word) :
    Scalar.laneIndex 2 2 p ^^^ laneXor 2 2 = Scalar.laneIndex 2 2 (~~~p) := by
  let cells : List Nat := [2,9,11,16,20,29,38,47]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 2 2) (by decide)
  have hh := finite_xor_2_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_3_0 : ∀ b : Fin 256,
    Scalar.laneIndex 3 0 (Projection.scatter [3,11,19,27,35,43,51,59] (BitVec.ofNat 8 b.val)) ^^^ laneXor 3 0 =
    Scalar.laneIndex 3 0 (~~~(Projection.scatter [3,11,19,27,35,43,51,59] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_3_0 (p : Word) :
    Scalar.laneIndex 3 0 p ^^^ laneXor 3 0 = Scalar.laneIndex 3 0 (~~~p) := by
  let cells : List Nat := [3,11,19,27,35,43,51,59]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 3 0) (by decide)
  have hh := finite_xor_3_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_3_1 : ∀ b : Fin 256,
    Scalar.laneIndex 3 1 (Projection.scatter [0,1,2,3,4,5,6,7] (BitVec.ofNat 8 b.val)) ^^^ laneXor 3 1 =
    Scalar.laneIndex 3 1 (~~~(Projection.scatter [0,1,2,3,4,5,6,7] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_3_1 (p : Word) :
    Scalar.laneIndex 3 1 p ^^^ laneXor 3 1 = Scalar.laneIndex 3 1 (~~~p) := by
  let cells : List Nat := [0,1,2,3,4,5,6,7]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 3 1) (by decide)
  have hh := finite_xor_3_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_3_2 : ∀ b : Fin 256,
    Scalar.laneIndex 3 2 (Projection.scatter [3,10,12,17,21,24,30,39] (BitVec.ofNat 8 b.val)) ^^^ laneXor 3 2 =
    Scalar.laneIndex 3 2 (~~~(Projection.scatter [3,10,12,17,21,24,30,39] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_3_2 (p : Word) :
    Scalar.laneIndex 3 2 p ^^^ laneXor 3 2 = Scalar.laneIndex 3 2 (~~~p) := by
  let cells : List Nat := [3,10,12,17,21,24,30,39]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 3 2) (by decide)
  have hh := finite_xor_3_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_4_0 : ∀ b : Fin 256,
    Scalar.laneIndex 4 0 (Projection.scatter [4,12,20,28,36,44,52,60] (BitVec.ofNat 8 b.val)) ^^^ laneXor 4 0 =
    Scalar.laneIndex 4 0 (~~~(Projection.scatter [4,12,20,28,36,44,52,60] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_4_0 (p : Word) :
    Scalar.laneIndex 4 0 p ^^^ laneXor 4 0 = Scalar.laneIndex 4 0 (~~~p) := by
  let cells : List Nat := [4,12,20,28,36,44,52,60]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 4 0) (by decide)
  have hh := finite_xor_4_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_4_1 : ∀ b : Fin 256,
    Scalar.laneIndex 4 1 (Projection.scatter [0,1,2,3,4,5,6,7] (BitVec.ofNat 8 b.val)) ^^^ laneXor 4 1 =
    Scalar.laneIndex 4 1 (~~~(Projection.scatter [0,1,2,3,4,5,6,7] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_4_1 (p : Word) :
    Scalar.laneIndex 4 1 p ^^^ laneXor 4 1 = Scalar.laneIndex 4 1 (~~~p) := by
  let cells : List Nat := [0,1,2,3,4,5,6,7]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 4 1) (by decide)
  have hh := finite_xor_4_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_4_2 : ∀ b : Fin 256,
    Scalar.laneIndex 4 2 (Projection.scatter [4,11,13,18,22,25,31,32] (BitVec.ofNat 8 b.val)) ^^^ laneXor 4 2 =
    Scalar.laneIndex 4 2 (~~~(Projection.scatter [4,11,13,18,22,25,31,32] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_4_2 (p : Word) :
    Scalar.laneIndex 4 2 p ^^^ laneXor 4 2 = Scalar.laneIndex 4 2 (~~~p) := by
  let cells : List Nat := [4,11,13,18,22,25,31,32]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 4 2) (by decide)
  have hh := finite_xor_4_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_5_0 : ∀ b : Fin 256,
    Scalar.laneIndex 5 0 (Projection.scatter [5,13,21,29,37,45,53,61] (BitVec.ofNat 8 b.val)) ^^^ laneXor 5 0 =
    Scalar.laneIndex 5 0 (~~~(Projection.scatter [5,13,21,29,37,45,53,61] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_5_0 (p : Word) :
    Scalar.laneIndex 5 0 p ^^^ laneXor 5 0 = Scalar.laneIndex 5 0 (~~~p) := by
  let cells : List Nat := [5,13,21,29,37,45,53,61]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 5 0) (by decide)
  have hh := finite_xor_5_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_5_1 : ∀ b : Fin 256,
    Scalar.laneIndex 5 1 (Projection.scatter [0,1,2,3,4,5,6,7] (BitVec.ofNat 8 b.val)) ^^^ laneXor 5 1 =
    Scalar.laneIndex 5 1 (~~~(Projection.scatter [0,1,2,3,4,5,6,7] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_5_1 (p : Word) :
    Scalar.laneIndex 5 1 p ^^^ laneXor 5 1 = Scalar.laneIndex 5 1 (~~~p) := by
  let cells : List Nat := [0,1,2,3,4,5,6,7]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 5 1) (by decide)
  have hh := finite_xor_5_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_5_2 : ∀ b : Fin 256,
    Scalar.laneIndex 5 2 (Projection.scatter [5,12,14,19,23,26,33,40] (BitVec.ofNat 8 b.val)) ^^^ laneXor 5 2 =
    Scalar.laneIndex 5 2 (~~~(Projection.scatter [5,12,14,19,23,26,33,40] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_5_2 (p : Word) :
    Scalar.laneIndex 5 2 p ^^^ laneXor 5 2 = Scalar.laneIndex 5 2 (~~~p) := by
  let cells : List Nat := [5,12,14,19,23,26,33,40]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 5 2) (by decide)
  have hh := finite_xor_5_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_6_0 : ∀ b : Fin 256,
    Scalar.laneIndex 6 0 (Projection.scatter [6,14,22,30,38,46,54,62] (BitVec.ofNat 8 b.val)) ^^^ laneXor 6 0 =
    Scalar.laneIndex 6 0 (~~~(Projection.scatter [6,14,22,30,38,46,54,62] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_6_0 (p : Word) :
    Scalar.laneIndex 6 0 p ^^^ laneXor 6 0 = Scalar.laneIndex 6 0 (~~~p) := by
  let cells : List Nat := [6,14,22,30,38,46,54,62]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 6 0) (by decide)
  have hh := finite_xor_6_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_6_1 : ∀ b : Fin 256,
    Scalar.laneIndex 6 1 (Projection.scatter [0,1,2,3,4,5,6,7] (BitVec.ofNat 8 b.val)) ^^^ laneXor 6 1 =
    Scalar.laneIndex 6 1 (~~~(Projection.scatter [0,1,2,3,4,5,6,7] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_6_1 (p : Word) :
    Scalar.laneIndex 6 1 p ^^^ laneXor 6 1 = Scalar.laneIndex 6 1 (~~~p) := by
  let cells : List Nat := [0,1,2,3,4,5,6,7]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 6 1) (by decide)
  have hh := finite_xor_6_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_6_2 : ∀ b : Fin 256,
    Scalar.laneIndex 6 2 (Projection.scatter [6,13,20,27,34,41,48] (BitVec.ofNat 8 b.val)) ^^^ laneXor 6 2 =
    Scalar.laneIndex 6 2 (~~~(Projection.scatter [6,13,20,27,34,41,48] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_6_2 (p : Word) :
    Scalar.laneIndex 6 2 p ^^^ laneXor 6 2 = Scalar.laneIndex 6 2 (~~~p) := by
  let cells : List Nat := [6,13,20,27,34,41,48]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 6 2) (by decide)
  have hh := finite_xor_6_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_7_0 : ∀ b : Fin 256,
    Scalar.laneIndex 7 0 (Projection.scatter [7,15,23,31,39,47,55,63] (BitVec.ofNat 8 b.val)) ^^^ laneXor 7 0 =
    Scalar.laneIndex 7 0 (~~~(Projection.scatter [7,15,23,31,39,47,55,63] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_7_0 (p : Word) :
    Scalar.laneIndex 7 0 p ^^^ laneXor 7 0 = Scalar.laneIndex 7 0 (~~~p) := by
  let cells : List Nat := [7,15,23,31,39,47,55,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 7 0) (by decide)
  have hh := finite_xor_7_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_7_1 : ∀ b : Fin 256,
    Scalar.laneIndex 7 1 (Projection.scatter [0,1,2,3,4,5,6,7] (BitVec.ofNat 8 b.val)) ^^^ laneXor 7 1 =
    Scalar.laneIndex 7 1 (~~~(Projection.scatter [0,1,2,3,4,5,6,7] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_7_1 (p : Word) :
    Scalar.laneIndex 7 1 p ^^^ laneXor 7 1 = Scalar.laneIndex 7 1 (~~~p) := by
  let cells : List Nat := [0,1,2,3,4,5,6,7]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 7 1) (by decide)
  have hh := finite_xor_7_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_7_2 : ∀ b : Fin 256,
    Scalar.laneIndex 7 2 (Projection.scatter [7,14,21,28,35,42,49,56] (BitVec.ofNat 8 b.val)) ^^^ laneXor 7 2 =
    Scalar.laneIndex 7 2 (~~~(Projection.scatter [7,14,21,28,35,42,49,56] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_7_2 (p : Word) :
    Scalar.laneIndex 7 2 p ^^^ laneXor 7 2 = Scalar.laneIndex 7 2 (~~~p) := by
  let cells : List Nat := [7,14,21,28,35,42,49,56]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 7 2) (by decide)
  have hh := finite_xor_7_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

end NeuralReversi.Solve1.Neon

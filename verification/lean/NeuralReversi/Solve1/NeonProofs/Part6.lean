import NeuralReversi.Solve1.NeonModel
import NeuralReversi.Solve1.NeonProofs.Part5
namespace NeuralReversi.Solve1.Neon
set_option maxRecDepth 100000
set_option maxHeartbeats 0
private theorem finite_xor_48_0 : ∀ b : Fin 256,
    Scalar.laneIndex 48 0 (Projection.scatter [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 b.val)) ^^^ laneXor 48 0 =
    Scalar.laneIndex 48 0 (~~~(Projection.scatter [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_48_0 (p : Word) :
    Scalar.laneIndex 48 0 p ^^^ laneXor 48 0 = Scalar.laneIndex 48 0 (~~~p) := by
  let cells : List Nat := [0,8,16,24,32,40,48,56]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 48 0) (by decide)
  have hh := finite_xor_48_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_48_1 : ∀ b : Fin 256,
    Scalar.laneIndex 48 1 (Projection.scatter [48,49,50,51,52,53,54,55] (BitVec.ofNat 8 b.val)) ^^^ laneXor 48 1 =
    Scalar.laneIndex 48 1 (~~~(Projection.scatter [48,49,50,51,52,53,54,55] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_48_1 (p : Word) :
    Scalar.laneIndex 48 1 p ^^^ laneXor 48 1 = Scalar.laneIndex 48 1 (~~~p) := by
  let cells : List Nat := [48,49,50,51,52,53,54,55]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 48 1) (by decide)
  have hh := finite_xor_48_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_48_2 : ∀ b : Fin 256,
    Scalar.laneIndex 48 2 (Projection.scatter [6,13,20,27,34,41,48] (BitVec.ofNat 8 b.val)) ^^^ laneXor 48 2 =
    Scalar.laneIndex 48 2 (~~~(Projection.scatter [6,13,20,27,34,41,48] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_48_2 (p : Word) :
    Scalar.laneIndex 48 2 p ^^^ laneXor 48 2 = Scalar.laneIndex 48 2 (~~~p) := by
  let cells : List Nat := [6,13,20,27,34,41,48]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 48 2) (by decide)
  have hh := finite_xor_48_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_49_0 : ∀ b : Fin 256,
    Scalar.laneIndex 49 0 (Projection.scatter [1,9,17,25,33,41,49,57] (BitVec.ofNat 8 b.val)) ^^^ laneXor 49 0 =
    Scalar.laneIndex 49 0 (~~~(Projection.scatter [1,9,17,25,33,41,49,57] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_49_0 (p : Word) :
    Scalar.laneIndex 49 0 p ^^^ laneXor 49 0 = Scalar.laneIndex 49 0 (~~~p) := by
  let cells : List Nat := [1,9,17,25,33,41,49,57]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 49 0) (by decide)
  have hh := finite_xor_49_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_49_1 : ∀ b : Fin 256,
    Scalar.laneIndex 49 1 (Projection.scatter [48,49,50,51,52,53,54,55] (BitVec.ofNat 8 b.val)) ^^^ laneXor 49 1 =
    Scalar.laneIndex 49 1 (~~~(Projection.scatter [48,49,50,51,52,53,54,55] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_49_1 (p : Word) :
    Scalar.laneIndex 49 1 p ^^^ laneXor 49 1 = Scalar.laneIndex 49 1 (~~~p) := by
  let cells : List Nat := [48,49,50,51,52,53,54,55]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 49 1) (by decide)
  have hh := finite_xor_49_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_49_2 : ∀ b : Fin 256,
    Scalar.laneIndex 49 2 (Projection.scatter [7,14,21,28,35,42,49,56] (BitVec.ofNat 8 b.val)) ^^^ laneXor 49 2 =
    Scalar.laneIndex 49 2 (~~~(Projection.scatter [7,14,21,28,35,42,49,56] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_49_2 (p : Word) :
    Scalar.laneIndex 49 2 p ^^^ laneXor 49 2 = Scalar.laneIndex 49 2 (~~~p) := by
  let cells : List Nat := [7,14,21,28,35,42,49,56]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 49 2) (by decide)
  have hh := finite_xor_49_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_50_0 : ∀ b : Fin 256,
    Scalar.laneIndex 50 0 (Projection.scatter [2,10,18,26,34,42,50,58] (BitVec.ofNat 8 b.val)) ^^^ laneXor 50 0 =
    Scalar.laneIndex 50 0 (~~~(Projection.scatter [2,10,18,26,34,42,50,58] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_50_0 (p : Word) :
    Scalar.laneIndex 50 0 p ^^^ laneXor 50 0 = Scalar.laneIndex 50 0 (~~~p) := by
  let cells : List Nat := [2,10,18,26,34,42,50,58]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 50 0) (by decide)
  have hh := finite_xor_50_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_50_1 : ∀ b : Fin 256,
    Scalar.laneIndex 50 1 (Projection.scatter [48,49,50,51,52,53,54,55] (BitVec.ofNat 8 b.val)) ^^^ laneXor 50 1 =
    Scalar.laneIndex 50 1 (~~~(Projection.scatter [48,49,50,51,52,53,54,55] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_50_1 (p : Word) :
    Scalar.laneIndex 50 1 p ^^^ laneXor 50 1 = Scalar.laneIndex 50 1 (~~~p) := by
  let cells : List Nat := [48,49,50,51,52,53,54,55]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 50 1) (by decide)
  have hh := finite_xor_50_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_50_2 : ∀ b : Fin 256,
    Scalar.laneIndex 50 2 (Projection.scatter [15,22,29,32,36,41,43,50] (BitVec.ofNat 8 b.val)) ^^^ laneXor 50 2 =
    Scalar.laneIndex 50 2 (~~~(Projection.scatter [15,22,29,32,36,41,43,50] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_50_2 (p : Word) :
    Scalar.laneIndex 50 2 p ^^^ laneXor 50 2 = Scalar.laneIndex 50 2 (~~~p) := by
  let cells : List Nat := [15,22,29,32,36,41,43,50]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 50 2) (by decide)
  have hh := finite_xor_50_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_51_0 : ∀ b : Fin 256,
    Scalar.laneIndex 51 0 (Projection.scatter [3,11,19,27,35,43,51,59] (BitVec.ofNat 8 b.val)) ^^^ laneXor 51 0 =
    Scalar.laneIndex 51 0 (~~~(Projection.scatter [3,11,19,27,35,43,51,59] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_51_0 (p : Word) :
    Scalar.laneIndex 51 0 p ^^^ laneXor 51 0 = Scalar.laneIndex 51 0 (~~~p) := by
  let cells : List Nat := [3,11,19,27,35,43,51,59]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 51 0) (by decide)
  have hh := finite_xor_51_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_51_1 : ∀ b : Fin 256,
    Scalar.laneIndex 51 1 (Projection.scatter [48,49,50,51,52,53,54,55] (BitVec.ofNat 8 b.val)) ^^^ laneXor 51 1 =
    Scalar.laneIndex 51 1 (~~~(Projection.scatter [48,49,50,51,52,53,54,55] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_51_1 (p : Word) :
    Scalar.laneIndex 51 1 p ^^^ laneXor 51 1 = Scalar.laneIndex 51 1 (~~~p) := by
  let cells : List Nat := [48,49,50,51,52,53,54,55]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 51 1) (by decide)
  have hh := finite_xor_51_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_51_2 : ∀ b : Fin 256,
    Scalar.laneIndex 51 2 (Projection.scatter [23,24,30,33,37,42,44,51] (BitVec.ofNat 8 b.val)) ^^^ laneXor 51 2 =
    Scalar.laneIndex 51 2 (~~~(Projection.scatter [23,24,30,33,37,42,44,51] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_51_2 (p : Word) :
    Scalar.laneIndex 51 2 p ^^^ laneXor 51 2 = Scalar.laneIndex 51 2 (~~~p) := by
  let cells : List Nat := [23,24,30,33,37,42,44,51]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 51 2) (by decide)
  have hh := finite_xor_51_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_52_0 : ∀ b : Fin 256,
    Scalar.laneIndex 52 0 (Projection.scatter [4,12,20,28,36,44,52,60] (BitVec.ofNat 8 b.val)) ^^^ laneXor 52 0 =
    Scalar.laneIndex 52 0 (~~~(Projection.scatter [4,12,20,28,36,44,52,60] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_52_0 (p : Word) :
    Scalar.laneIndex 52 0 p ^^^ laneXor 52 0 = Scalar.laneIndex 52 0 (~~~p) := by
  let cells : List Nat := [4,12,20,28,36,44,52,60]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 52 0) (by decide)
  have hh := finite_xor_52_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_52_1 : ∀ b : Fin 256,
    Scalar.laneIndex 52 1 (Projection.scatter [48,49,50,51,52,53,54,55] (BitVec.ofNat 8 b.val)) ^^^ laneXor 52 1 =
    Scalar.laneIndex 52 1 (~~~(Projection.scatter [48,49,50,51,52,53,54,55] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_52_1 (p : Word) :
    Scalar.laneIndex 52 1 p ^^^ laneXor 52 1 = Scalar.laneIndex 52 1 (~~~p) := by
  let cells : List Nat := [48,49,50,51,52,53,54,55]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 52 1) (by decide)
  have hh := finite_xor_52_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_52_2 : ∀ b : Fin 256,
    Scalar.laneIndex 52 2 (Projection.scatter [16,25,31,34,38,43,45,52] (BitVec.ofNat 8 b.val)) ^^^ laneXor 52 2 =
    Scalar.laneIndex 52 2 (~~~(Projection.scatter [16,25,31,34,38,43,45,52] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_52_2 (p : Word) :
    Scalar.laneIndex 52 2 p ^^^ laneXor 52 2 = Scalar.laneIndex 52 2 (~~~p) := by
  let cells : List Nat := [16,25,31,34,38,43,45,52]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 52 2) (by decide)
  have hh := finite_xor_52_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_53_0 : ∀ b : Fin 256,
    Scalar.laneIndex 53 0 (Projection.scatter [5,13,21,29,37,45,53,61] (BitVec.ofNat 8 b.val)) ^^^ laneXor 53 0 =
    Scalar.laneIndex 53 0 (~~~(Projection.scatter [5,13,21,29,37,45,53,61] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_53_0 (p : Word) :
    Scalar.laneIndex 53 0 p ^^^ laneXor 53 0 = Scalar.laneIndex 53 0 (~~~p) := by
  let cells : List Nat := [5,13,21,29,37,45,53,61]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 53 0) (by decide)
  have hh := finite_xor_53_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_53_1 : ∀ b : Fin 256,
    Scalar.laneIndex 53 1 (Projection.scatter [48,49,50,51,52,53,54,55] (BitVec.ofNat 8 b.val)) ^^^ laneXor 53 1 =
    Scalar.laneIndex 53 1 (~~~(Projection.scatter [48,49,50,51,52,53,54,55] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_53_1 (p : Word) :
    Scalar.laneIndex 53 1 p ^^^ laneXor 53 1 = Scalar.laneIndex 53 1 (~~~p) := by
  let cells : List Nat := [48,49,50,51,52,53,54,55]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 53 1) (by decide)
  have hh := finite_xor_53_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_53_2 : ∀ b : Fin 256,
    Scalar.laneIndex 53 2 (Projection.scatter [8,17,26,35,39,44,46,53] (BitVec.ofNat 8 b.val)) ^^^ laneXor 53 2 =
    Scalar.laneIndex 53 2 (~~~(Projection.scatter [8,17,26,35,39,44,46,53] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_53_2 (p : Word) :
    Scalar.laneIndex 53 2 p ^^^ laneXor 53 2 = Scalar.laneIndex 53 2 (~~~p) := by
  let cells : List Nat := [8,17,26,35,39,44,46,53]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 53 2) (by decide)
  have hh := finite_xor_53_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_54_0 : ∀ b : Fin 256,
    Scalar.laneIndex 54 0 (Projection.scatter [6,14,22,30,38,46,54,62] (BitVec.ofNat 8 b.val)) ^^^ laneXor 54 0 =
    Scalar.laneIndex 54 0 (~~~(Projection.scatter [6,14,22,30,38,46,54,62] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_54_0 (p : Word) :
    Scalar.laneIndex 54 0 p ^^^ laneXor 54 0 = Scalar.laneIndex 54 0 (~~~p) := by
  let cells : List Nat := [6,14,22,30,38,46,54,62]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 54 0) (by decide)
  have hh := finite_xor_54_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_54_1 : ∀ b : Fin 256,
    Scalar.laneIndex 54 1 (Projection.scatter [48,49,50,51,52,53,54,55] (BitVec.ofNat 8 b.val)) ^^^ laneXor 54 1 =
    Scalar.laneIndex 54 1 (~~~(Projection.scatter [48,49,50,51,52,53,54,55] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_54_1 (p : Word) :
    Scalar.laneIndex 54 1 p ^^^ laneXor 54 1 = Scalar.laneIndex 54 1 (~~~p) := by
  let cells : List Nat := [48,49,50,51,52,53,54,55]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 54 1) (by decide)
  have hh := finite_xor_54_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_54_2 : ∀ b : Fin 256,
    Scalar.laneIndex 54 2 (Projection.scatter [0,9,18,27,36,45,54,63] (BitVec.ofNat 8 b.val)) ^^^ laneXor 54 2 =
    Scalar.laneIndex 54 2 (~~~(Projection.scatter [0,9,18,27,36,45,54,63] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_54_2 (p : Word) :
    Scalar.laneIndex 54 2 p ^^^ laneXor 54 2 = Scalar.laneIndex 54 2 (~~~p) := by
  let cells : List Nat := [0,9,18,27,36,45,54,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 54 2) (by decide)
  have hh := finite_xor_54_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_55_0 : ∀ b : Fin 256,
    Scalar.laneIndex 55 0 (Projection.scatter [7,15,23,31,39,47,55,63] (BitVec.ofNat 8 b.val)) ^^^ laneXor 55 0 =
    Scalar.laneIndex 55 0 (~~~(Projection.scatter [7,15,23,31,39,47,55,63] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_55_0 (p : Word) :
    Scalar.laneIndex 55 0 p ^^^ laneXor 55 0 = Scalar.laneIndex 55 0 (~~~p) := by
  let cells : List Nat := [7,15,23,31,39,47,55,63]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 55 0) (by decide)
  have hh := finite_xor_55_0 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_55_1 : ∀ b : Fin 256,
    Scalar.laneIndex 55 1 (Projection.scatter [48,49,50,51,52,53,54,55] (BitVec.ofNat 8 b.val)) ^^^ laneXor 55 1 =
    Scalar.laneIndex 55 1 (~~~(Projection.scatter [48,49,50,51,52,53,54,55] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_55_1 (p : Word) :
    Scalar.laneIndex 55 1 p ^^^ laneXor 55 1 = Scalar.laneIndex 55 1 (~~~p) := by
  let cells : List Nat := [48,49,50,51,52,53,54,55]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 55 1) (by decide)
  have hh := finite_xor_55_1 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

private theorem finite_xor_55_2 : ∀ b : Fin 256,
    Scalar.laneIndex 55 2 (Projection.scatter [1,10,19,28,37,46,55] (BitVec.ofNat 8 b.val)) ^^^ laneXor 55 2 =
    Scalar.laneIndex 55 2 (~~~(Projection.scatter [1,10,19,28,37,46,55] (BitVec.ofNat 8 b.val))) := by decide +kernel

theorem xor_55_2 (p : Word) :
    Scalar.laneIndex 55 2 p ^^^ laneXor 55 2 = Scalar.laneIndex 55 2 (~~~p) := by
  let cells : List Nat := [1,10,19,28,37,46,55]
  have hp := Projection.masked_eq_scatter cells p (by decide) (Scalar.laneMask 55 2) (by decide)
  have hh := finite_xor_55_2 ⟨(Projection.gather cells p).toNat, (Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp, laneIndex_mask, laneIndex_not_mask] at hh
  exact hh

end NeuralReversi.Solve1.Neon

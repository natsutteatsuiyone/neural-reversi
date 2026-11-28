module
public import NeuralReversi.Bitboard.GeometryModel
public import NeuralReversi.Bitboard.GeometryCoordinates

@[expose] public section

namespace NeuralReversi.Bitboard

private theorem xorABA (a b : Bool) : (a ^^ (b ^^ a)) = b := by
  cases a <;> cases b <;> rfl

private theorem byteAt_getLsbD (bits : Bits) (k i : Nat) (hi : i < 8) :
    (byteAt bits k).getLsbD i = bits.getLsbD (8 * k + i) := by
  simp [byteAt, hi]

theorem vertical_geometry (bits : Bits) : ∀ i : Nat, i < 64 →
    (flipVertical bits).getLsbD i = bits.getLsbD (verticalIndex i) := by
  intro i hi
  simp only [flipVertical, BitVec.getLsbD_append]
  have hq : i / 8 = 0 ∨ i / 8 = 1 ∨ i / 8 = 2 ∨ i / 8 = 3 ∨
      i / 8 = 4 ∨ i / 8 = 5 ∨ i / 8 = 6 ∨ i / 8 = 7 := by omega
  rcases hq with hq | hq | hq | hq | hq | hq | hq | hq
  all_goals
    repeat (first | rw [ite_eq_left (by omega)] | rw [ite_eq_right (by omega)])
    rw [byteAt_getLsbD _ _ _ (by omega)]
    congr 1; unfold verticalIndex; omega

private theorem swapStage_getLsbD (bits mask : Bits) (k i : Nat) (hi : i < 64)
    (hmask : ∀ j : Fin 64,
      mask.getLsbD j.val = true → j.val + k < 64 ∧ mask.getLsbD (j.val + k) = false)
    (hmask_lo : ∀ j : Fin 64,
      mask.getLsbD j.val = false → k ≤ j.val ∧ mask.getLsbD (j.val - k) = true) :
    (((bits >>> k) &&& mask) ||| ((bits &&& mask) <<< k)).getLsbD i =
      bits.getLsbD (if mask.getLsbD i then i + k else i - k) := by
  simp only [BitVec.getLsbD_or, BitVec.getLsbD_and, BitVec.getLsbD_ushiftRight,
    BitVec.getLsbD_shiftLeft, hi, decide_true, Bool.true_and]
  cases hm : mask.getLsbD i with
  | true =>
    by_cases hki : k ≤ i
    · have hm2 : mask.getLsbD (i - k) = false := by
        cases hprev : mask.getLsbD (i - k) with
        | false => rfl
        | true =>
          have hcontra := (hmask ⟨i - k, by omega⟩ hprev).2
          rw [Nat.sub_add_cancel hki] at hcontra
          simp [hm] at hcontra
      simp [hm2, Nat.add_comm]
    · have hlt : i < k := by omega
      simp [hlt, Nat.add_comm]
  | false =>
    have ⟨hki, hm2⟩ := hmask_lo ⟨i, hi⟩ hm
    dsimp only at hki hm2
    have hnot : ¬(i < k) := by omega
    simp [hnot, hm2]

private def swapIndex (mask : Bits) (k i : Nat) : Nat :=
  if mask.getLsbD i then i + k else i - k

private theorem horizontal_masks :
    (∀ j : Fin 64, (0x5555555555555555 : Bits).getLsbD j.val = true →
        j.val + 1 < 64 ∧ (0x5555555555555555 : Bits).getLsbD (j.val + 1) = false) ∧
    (∀ j : Fin 64, (0x5555555555555555 : Bits).getLsbD j.val = false →
        1 ≤ j.val ∧ (0x5555555555555555 : Bits).getLsbD (j.val - 1) = true) ∧
    (∀ j : Fin 64, (0x3333333333333333 : Bits).getLsbD j.val = true →
        j.val + 2 < 64 ∧ (0x3333333333333333 : Bits).getLsbD (j.val + 2) = false) ∧
    (∀ j : Fin 64, (0x3333333333333333 : Bits).getLsbD j.val = false →
        2 ≤ j.val ∧ (0x3333333333333333 : Bits).getLsbD (j.val - 2) = true) ∧
    (∀ j : Fin 64, (0x0f0f0f0f0f0f0f0f : Bits).getLsbD j.val = true →
        j.val + 4 < 64 ∧ (0x0f0f0f0f0f0f0f0f : Bits).getLsbD (j.val + 4) = false) ∧
    (∀ j : Fin 64, (0x0f0f0f0f0f0f0f0f : Bits).getLsbD j.val = false →
        4 ≤ j.val ∧ (0x0f0f0f0f0f0f0f0f : Bits).getLsbD (j.val - 4) = true) ∧
    (∀ j : Fin 64,
      let i1 := swapIndex 0x0f0f0f0f0f0f0f0f 4 j.val
      let i2 := swapIndex 0x3333333333333333 2 i1
      let i3 := swapIndex 0x5555555555555555 1 i2
      i1 < 64 ∧ i2 < 64 ∧ i3 = horizontalIndex j.val) := by
  decide +kernel

theorem horizontal_geometry (bits : Bits) : ∀ i : Nat, i < 64 →
    (flipHorizontal bits).getLsbD i = bits.getLsbD (horizontalIndex i) := by
  intro i hi
  have hm := horizontal_masks
  have hidx := hm.2.2.2.2.2.2 ⟨i, hi⟩
  dsimp only at hidx
  let b1 : Bits := ((bits >>> 1) &&& 0x5555555555555555) ||| ((bits &&& 0x5555555555555555) <<< 1)
  let b2 : Bits := ((b1 >>> 2) &&& 0x3333333333333333) ||| ((b1 &&& 0x3333333333333333) <<< 2)
  have hs1 : (flipHorizontal bits).getLsbD i = b2.getLsbD (swapIndex 0x0f0f0f0f0f0f0f0f 4 i) :=
    swapStage_getLsbD b2 0x0f0f0f0f0f0f0f0f 4 i hi hm.2.2.2.2.1 hm.2.2.2.2.2.1
  have hs2 : b2.getLsbD (swapIndex 0x0f0f0f0f0f0f0f0f 4 i) =
      b1.getLsbD (swapIndex 0x3333333333333333 2 (swapIndex 0x0f0f0f0f0f0f0f0f 4 i)) :=
    swapStage_getLsbD b1 0x3333333333333333 2 _ hidx.1 hm.2.2.1 hm.2.2.2.1
  have hs3 : b1.getLsbD (swapIndex 0x3333333333333333 2 (swapIndex 0x0f0f0f0f0f0f0f0f 4 i)) =
      bits.getLsbD (swapIndex 0x5555555555555555 1 (swapIndex 0x3333333333333333 2 (swapIndex 0x0f0f0f0f0f0f0f0f 4 i))) :=
    swapStage_getLsbD bits 0x5555555555555555 1 _ hidx.2.1 hm.1 hm.2.1
  exact hs1.trans (hs2.trans (hs3.trans (congrArg bits.getLsbD hidx.2.2)))

private def deltaSwapIndex (mask : Bits) (delta i : Nat) : Nat :=
  if mask.getLsbD i then i - delta
  else if mask.getLsbD (i + delta) then i + delta
  else i

private theorem deltaSwapRaw_getLsbD (bits mask : Bits) (delta i : Nat) (hi : i < 64)
    (hdisj : ∀ j : Fin 64, mask.getLsbD j.val = true →
      delta ≤ j.val ∧ mask.getLsbD (j.val + delta) = false) :
    (deltaSwapRaw bits mask delta).getLsbD i = bits.getLsbD (deltaSwapIndex mask delta i) := by
  simp only [deltaSwapRaw, deltaSwapIndex, BitVec.getLsbD_xor, BitVec.getLsbD_and,
    BitVec.getLsbD_shiftLeft, BitVec.getLsbD_ushiftRight, hi, decide_true, Bool.true_and]
  cases hm1 : mask.getLsbD i with
  | true =>
    have ⟨hdi, hm2⟩ := hdisj ⟨i, hi⟩ hm1
    dsimp only at hdi hm2
    have hnot : ¬(i < delta) := by omega
    have hm2' : mask.getLsbD (delta + i) = false := by rwa [Nat.add_comm]
    simp [hnot, hm2']
  | false =>
    cases hm2 : mask.getLsbD (delta + i) with
    | true =>
      have hlt : delta + i < 64 := BitVec.lt_of_getLsbD hm2
      have hnot : ¬(delta + i < delta) := by omega
      have h1 : mask.getLsbD (i + delta) = true := by rwa [Nat.add_comm]
      rw [ite_eq_left h1]
      simp [hlt, hnot, Nat.add_sub_cancel_left, xorABA, Nat.add_comm]
    | false =>
      have h1 : mask.getLsbD (i + delta) = false := by rwa [Nat.add_comm]
      simp [h1]

private theorem mainDiagonal_masks :
    (∀ j : Fin 64, (0x0f0f0f0f00000000 : Bits).getLsbD j.val = true →
        28 ≤ j.val ∧ (0x0f0f0f0f00000000 : Bits).getLsbD (j.val + 28) = false) ∧
    (∀ j : Fin 64, (0x3333000033330000 : Bits).getLsbD j.val = true →
        14 ≤ j.val ∧ (0x3333000033330000 : Bits).getLsbD (j.val + 14) = false) ∧
    (∀ j : Fin 64, (0x5500550055005500 : Bits).getLsbD j.val = true →
        7 ≤ j.val ∧ (0x5500550055005500 : Bits).getLsbD (j.val + 7) = false) ∧
    (∀ j : Fin 64,
      let i1 := deltaSwapIndex 0x5500550055005500 7 j.val
      let i2 := deltaSwapIndex 0x3333000033330000 14 i1
      let i3 := deltaSwapIndex 0x0f0f0f0f00000000 28 i2
      i1 < 64 ∧ i2 < 64 ∧ i3 = mainDiagonalIndex j.val) := by
  decide +kernel

theorem mainDiagonal_geometry (bits : Bits) : ∀ i : Nat, i < 64 →
    (flipDiagA1H8 bits).getLsbD i = bits.getLsbD (mainDiagonalIndex i) := by
  intro i hi
  have hm := mainDiagonal_masks
  have hidx := hm.2.2.2 ⟨i, hi⟩
  dsimp only at hidx
  unfold flipDiagA1H8
  dsimp only
  rw [deltaSwapRaw_getLsbD _ 0x5500550055005500 7 i hi hm.2.2.1]
  rw [deltaSwapRaw_getLsbD _ 0x3333000033330000 14 _ hidx.1 hm.2.1]
  rw [deltaSwapRaw_getLsbD _ 0x0f0f0f0f00000000 28 _ hidx.2.1 hm.1]
  exact congrArg bits.getLsbD hidx.2.2

private theorem antiDiagonal_masks :
    (∀ j : Fin 64, (0xf0f0f0f000000000 : Bits).getLsbD j.val = true →
        36 ≤ j.val ∧ (0xf0f0f0f000000000 : Bits).getLsbD (j.val + 36) = false) ∧
    (∀ j : Fin 64, (0xcccc0000cccc0000 : Bits).getLsbD j.val = true →
        18 ≤ j.val ∧ (0xcccc0000cccc0000 : Bits).getLsbD (j.val + 18) = false) ∧
    (∀ j : Fin 64, (0xaa00aa00aa00aa00 : Bits).getLsbD j.val = true →
        9 ≤ j.val ∧ (0xaa00aa00aa00aa00 : Bits).getLsbD (j.val + 9) = false) ∧
    (∀ j : Fin 64,
      let i1 := deltaSwapIndex 0xaa00aa00aa00aa00 9 j.val
      let i2 := deltaSwapIndex 0xcccc0000cccc0000 18 i1
      let i3 := deltaSwapIndex 0xf0f0f0f000000000 36 i2
      i1 < 64 ∧ i2 < 64 ∧ i3 = antiDiagonalIndex j.val) := by
  decide +kernel

theorem antiDiagonal_geometry (bits : Bits) : ∀ i : Nat, i < 64 →
    (flipDiagA8H1 bits).getLsbD i = bits.getLsbD (antiDiagonalIndex i) := by
  intro i hi
  have hm := antiDiagonal_masks
  have hidx := hm.2.2.2 ⟨i, hi⟩
  dsimp only at hidx
  unfold flipDiagA8H1
  dsimp only
  rw [deltaSwapRaw_getLsbD _ 0xaa00aa00aa00aa00 9 i hi hm.2.2.1]
  rw [deltaSwapRaw_getLsbD _ 0xcccc0000cccc0000 18 _ hidx.1 hm.2.1]
  rw [deltaSwapRaw_getLsbD _ 0xf0f0f0f000000000 36 _ hidx.2.1 hm.1]
  exact congrArg bits.getLsbD hidx.2.2

theorem rotate90_geometry (bits : Bits) (i : Nat) (within : i < 64) :
    (rotate90 bits).getLsbD i = bits.getLsbD (rotate90Index i) := by
  rw [rotate90, vertical_geometry _ i within,
    antiDiagonal_geometry _ _ (geometryIndices_bounded i within).1]
  congr 1
  unfold antiDiagonalIndex verticalIndex rotate90Index
  omega

theorem rotate180_geometry (bits : Bits) (i : Nat) (within : i < 64) :
    (rotate180 bits).getLsbD i = bits.getLsbD (rotate180Index i) := by
  have index : 63 - i = rotate180Index i := by unfold rotate180Index; omega
  rw [rotate180, BitVec.getLsbD_reverse]
  simp only [BitVec.getMsbD, within, decide_true, Bool.true_and]
  exact congrArg bits.getLsbD index

theorem rotate270_geometry (bits : Bits) (i : Nat) (within : i < 64) :
    (rotate270 bits).getLsbD i = bits.getLsbD (rotate270Index i) := by
  rw [rotate270, vertical_geometry _ i within,
    mainDiagonal_geometry _ _ (geometryIndices_bounded i within).1]
  congr 1
  unfold mainDiagonalIndex verticalIndex rotate270Index
  omega

def GeometryContract : Prop :=
  ∀ (bits : Bits) (i : Nat), i < 64 →
    (flipVertical bits).getLsbD i = bits.getLsbD (verticalIndex i) ∧
    (flipHorizontal bits).getLsbD i = bits.getLsbD (horizontalIndex i) ∧
    (flipDiagA1H8 bits).getLsbD i = bits.getLsbD (mainDiagonalIndex i) ∧
    (flipDiagA8H1 bits).getLsbD i = bits.getLsbD (antiDiagonalIndex i) ∧
    (rotate90 bits).getLsbD i = bits.getLsbD (rotate90Index i) ∧
    (rotate180 bits).getLsbD i = bits.getLsbD (rotate180Index i) ∧
    (rotate270 bits).getLsbD i = bits.getLsbD (rotate270Index i)

theorem geometryContract_proven : GeometryContract := by
  intro bits i within
  exact ⟨vertical_geometry bits i within, horizontal_geometry bits i within,
    mainDiagonal_geometry bits i within, antiDiagonal_geometry bits i within,
    rotate90_geometry bits i within, rotate180_geometry bits i within,
    rotate270_geometry bits i within⟩

end NeuralReversi.Bitboard

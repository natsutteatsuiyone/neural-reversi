import NeuralReversi.Flip.WasmSimd.Model
import NeuralReversi.Flip.Neon.Properties

namespace NeuralReversi.Flip.WasmSimd
open NeuralReversi.Flip
open NeuralReversi.Flip.Avx512 (mask66 indexedValue indexedValue_typed)
open NeuralReversi.Flip.Neon (ls1b_and_player_eq reverse_or flipLeft_reverse_eq_flipRight
  getLsbD_reverse_eq)

set_option maxRecDepth 8192
set_option maxHeartbeats 2000000

theorem bitReverseByte_eq_reverse : ∀ b : BitVec 8, bitReverseByte b = b.reverse := by
  decide

private theorem getLsbD_byte_shift (b : BitVec 8) (k i : Nat) (hi : i < 64) :
    ((b.setWidth 64) <<< (k * 8)).getLsbD i =
      if k * 8 ≤ i ∧ i < k * 8 + 8 then b.getLsbD (i - k * 8) else false := by
  rw [BitVec.getLsbD_shiftLeft]
  by_cases h1 : k * 8 ≤ i
  · simp only [hi, decide_true, Bool.true_and, h1, decide_true, Bool.true_and]
    rw [BitVec.getLsbD_setWidth]
    by_cases h2 : i < k * 8 + 8
    · have hlt : i - k * 8 < 8 := by omega
      have hlt64 : i - k * 8 < 64 := by omega
      simp [h1, h2, hlt, hlt64]
    · have hge : 8 ≤ i - k * 8 := by omega
      simp [h2, BitVec.getLsbD_of_ge b (i - k * 8) hge]
  · simp [h1]

private theorem getLsbD_getByte64_reverse (x : Word) (k r : Nat) (hr : r < 8) :
    (getByte64 x k).reverse.getLsbD r = x.getLsbD (k * 8 + (7 - r)) := by
  simp only [BitVec.getLsbD_reverse, BitVec.getMsbD, hr, decide_true, Bool.true_and]
  unfold getByte64
  rw [BitVec.getLsbD_setWidth]
  have hlt : 7 - r < 8 := by omega
  simp only [hlt, decide_true, Bool.true_and, BitVec.getLsbD_ushiftRight]

theorem bitReverseU64_eq_reverse (x : Word) : bitReverseU64 x = x.reverse := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  rw [getLsbD_reverse_eq x i hi]
  unfold bitReverseU64 packBytes64
  simp only [bitReverseByte_eq_reverse, BitVec.getLsbD_or]
  have h0 := getLsbD_byte_shift (getByte64 x 7).reverse 0 i hi
  have h1 := getLsbD_byte_shift (getByte64 x 6).reverse 1 i hi
  have h2 := getLsbD_byte_shift (getByte64 x 5).reverse 2 i hi
  have h3 := getLsbD_byte_shift (getByte64 x 4).reverse 3 i hi
  have h4 := getLsbD_byte_shift (getByte64 x 3).reverse 4 i hi
  have h5 := getLsbD_byte_shift (getByte64 x 2).reverse 5 i hi
  have h6 := getLsbD_byte_shift (getByte64 x 1).reverse 6 i hi
  have h7 := getLsbD_byte_shift (getByte64 x 0).reverse 7 i hi
  simp only [Nat.zero_mul, BitVec.shiftLeft_zero] at h0
  rw [h0, h1, h2, h3, h4, h5, h6, h7]
  have hq : i / 8 = 0 ∨ i / 8 = 1 ∨ i / 8 = 2 ∨ i / 8 = 3 ∨
            i / 8 = 4 ∨ i / 8 = 5 ∨ i / 8 = 6 ∨ i / 8 = 7 := by omega
  rcases hq with hq | hq | hq | hq | hq | hq | hq | hq
  · have hlt : i < 8 := by omega
    have hif : (if 0 ≤ i ∧ i < 0 + 8 then (getByte64 x 7).reverse.getLsbD (i - 0) else false) =
        x.getLsbD (63 - i) := by
      rw [ite_eq_left (by omega), Nat.sub_zero, getLsbD_getByte64_reverse x 7 i hlt]
      congr 1; omega
    rw [hif, ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
      ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
      ite_eq_right (by omega)]
    simp
  · have hif : (if 1 * 8 ≤ i ∧ i < 1 * 8 + 8 then (getByte64 x 6).reverse.getLsbD (i - 1 * 8) else false) =
        x.getLsbD (63 - i) := by
      rw [ite_eq_left (by omega), getLsbD_getByte64_reverse x 6 (i - 8) (by omega)]
      congr 1; omega
    rw [ite_eq_right (by omega), hif, ite_eq_right (by omega), ite_eq_right (by omega),
      ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
      ite_eq_right (by omega)]
    simp
  · have hif : (if 2 * 8 ≤ i ∧ i < 2 * 8 + 8 then (getByte64 x 5).reverse.getLsbD (i - 2 * 8) else false) =
        x.getLsbD (63 - i) := by
      rw [ite_eq_left (by omega), getLsbD_getByte64_reverse x 5 (i - 16) (by omega)]
      congr 1; omega
    rw [ite_eq_right (by omega), ite_eq_right (by omega), hif, ite_eq_right (by omega),
      ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
      ite_eq_right (by omega)]
    simp
  · have hif : (if 3 * 8 ≤ i ∧ i < 3 * 8 + 8 then (getByte64 x 4).reverse.getLsbD (i - 3 * 8) else false) =
        x.getLsbD (63 - i) := by
      rw [ite_eq_left (by omega), getLsbD_getByte64_reverse x 4 (i - 24) (by omega)]
      congr 1; omega
    rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), hif,
      ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
      ite_eq_right (by omega)]
    simp
  · have hif : (if 4 * 8 ≤ i ∧ i < 4 * 8 + 8 then (getByte64 x 3).reverse.getLsbD (i - 4 * 8) else false) =
        x.getLsbD (63 - i) := by
      rw [ite_eq_left (by omega), getLsbD_getByte64_reverse x 3 (i - 32) (by omega)]
      congr 1; omega
    rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
      ite_eq_right (by omega), hif, ite_eq_right (by omega), ite_eq_right (by omega),
      ite_eq_right (by omega)]
    simp
  · have hif : (if 5 * 8 ≤ i ∧ i < 5 * 8 + 8 then (getByte64 x 2).reverse.getLsbD (i - 5 * 8) else false) =
        x.getLsbD (63 - i) := by
      rw [ite_eq_left (by omega), getLsbD_getByte64_reverse x 2 (i - 40) (by omega)]
      congr 1; omega
    rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
      ite_eq_right (by omega), ite_eq_right (by omega), hif, ite_eq_right (by omega),
      ite_eq_right (by omega)]
    simp
  · have hif : (if 6 * 8 ≤ i ∧ i < 6 * 8 + 8 then (getByte64 x 1).reverse.getLsbD (i - 6 * 8) else false) =
        x.getLsbD (63 - i) := by
      rw [ite_eq_left (by omega), getLsbD_getByte64_reverse x 1 (i - 48) (by omega)]
      congr 1; omega
    rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
      ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), hif,
      ite_eq_right (by omega)]
    simp
  · have hif : (if 7 * 8 ≤ i ∧ i < 7 * 8 + 8 then (getByte64 x 0).reverse.getLsbD (i - 7 * 8) else false) =
        x.getLsbD (63 - i) := by
      rw [ite_eq_left (by omega), getLsbD_getByte64_reverse x 0 (i - 56) (by omega)]
      congr 1; omega
    rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
      ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
      ite_eq_right (by omega), hif]
    simp

theorem flipLeftLane_eq_scalar (mask player notOpponent : Word) :
    (mask &&& ((((mask &&& notOpponent) &&& (0#64 - (mask &&& notOpponent))) &&& player) - 1#64)) &&&
      ~~~(u64EqMask (((mask &&& notOpponent) &&& (0#64 - (mask &&& notOpponent))) &&& player) 0#64) =
      Scalar.flipLeft mask player notOpponent := by
  have hcomm : mask &&& notOpponent = notOpponent &&& mask := BitVec.and_comm _ _
  have hneg : 0#64 - (notOpponent &&& mask) = -(notOpponent &&& mask) := BitVec.zero_sub _
  rw [hcomm, hneg]
  unfold Scalar.flipLeft u64EqMask
  dsimp only
  by_cases h : ((notOpponent &&& mask) &&& -(notOpponent &&& mask)) &&& player = 0#64
  · simp [h]
  · have hbeq : ((((notOpponent &&& mask) &&& -(notOpponent &&& mask)) &&& player) == 0#64) = false :=
      beq_false_of_ne h
    rw [hbeq]
    simp only [Bool.false_eq_true, ↓reduceIte]
    have hnot0 : ~~~(0#64) = BitVec.allOnes 64 := by decide
    rw [hnot0, BitVec.and_allOnes]
    have hne : ((notOpponent &&& mask) &&& -(notOpponent &&& mask)) &&& player ≠ 0 := h
    rw [ite_eq_left hne, ls1b_and_player_eq (notOpponent &&& mask) player h]

theorem flipLeftPair_eq_scalar (m0 m1 player notOpponent : Word) :
    flipLeftPair (m0, m1) (u64x2Splat player) (u64x2Splat notOpponent)
        (u64x2Splat 0#64) (u64x2Splat 1#64) =
      (Scalar.flipLeft m0 player notOpponent, Scalar.flipLeft m1 player notOpponent) := by
  unfold flipLeftPair v128And v128Not u64x2Sub u64x2Eq u64x2Splat
  apply Prod.ext <;> exact flipLeftLane_eq_scalar _ player notOpponent

theorem flipIndex_eq_indexedValue (pos : Fin 66) (player opponent : Word) :
    flipIndex pos player opponent = indexedValue pos player opponent := by
  unfold flipIndex BoardCtx.flip1 BoardCtx.new flipIndexPrepared bitReverseU64x2
  dsimp only
  rw [bitReverseU64_eq_reverse player, bitReverseU64_eq_reverse (~~~opponent)]
  have hm0 : wasmMask pos 0 = mask66 pos 0 := rfl
  have hm1 : wasmMask pos 1 = mask66 pos 1 := rfl
  have hm2 : wasmMask pos 2 = mask66 pos 2 := rfl
  have hm3 : wasmMask pos 3 = mask66 pos 3 := rfl
  have hm4 : wasmMask pos 4 = (mask66 pos 4).reverse := rfl
  have hm5 : wasmMask pos 5 = (mask66 pos 5).reverse := rfl
  have hm6 : wasmMask pos 6 = (mask66 pos 6).reverse := rfl
  have hm7 : wasmMask pos 7 = (mask66 pos 7).reverse := rfl
  rw [hm0, hm1, hm2, hm3, hm4, hm5, hm6, hm7]
  change foldOrPair (v128Or
    (v128Or
      (flipLeftPair (mask66 pos 0, mask66 pos 1) (u64x2Splat player) (u64x2Splat (~~~opponent))
        (u64x2Splat 0#64) (u64x2Splat 1#64))
      (flipLeftPair (mask66 pos 2, mask66 pos 3) (u64x2Splat player) (u64x2Splat (~~~opponent))
        (u64x2Splat 0#64) (u64x2Splat 1#64)))
    (let p := v128Or
      (flipLeftPair ((mask66 pos 4).reverse, (mask66 pos 5).reverse)
        (u64x2Splat player.reverse) (u64x2Splat (~~~opponent).reverse)
        (u64x2Splat 0#64) (u64x2Splat 1#64))
      (flipLeftPair ((mask66 pos 6).reverse, (mask66 pos 7).reverse)
        (u64x2Splat player.reverse) (u64x2Splat (~~~opponent).reverse)
        (u64x2Splat 0#64) (u64x2Splat 1#64))
     (bitReverseU64 p.1, bitReverseU64 p.2))) = _
  rw [flipLeftPair_eq_scalar, flipLeftPair_eq_scalar,
    flipLeftPair_eq_scalar, flipLeftPair_eq_scalar]
  unfold foldOrPair v128Or
  dsimp only
  rw [bitReverseU64_eq_reverse, bitReverseU64_eq_reverse]
  rw [reverse_or, reverse_or]
  rw [flipLeft_reverse_eq_flipRight, flipLeft_reverse_eq_flipRight,
    flipLeft_reverse_eq_flipRight, flipLeft_reverse_eq_flipRight]
  unfold indexedValue
  ac_rfl

theorem flip_eq_scalarValue (sq : Square) (player opponent : Word) :
    flip sq player opponent = value sq player opponent := by
  unfold flip
  rw [flipIndex_eq_indexedValue _ player opponent]
  exact indexedValue_typed ⟨sq.val, by have := sq.isLt; omega⟩ sq.isLt player opponent

theorem flip_coordinate (sq : Square) (player opponent : Word) (i : Nat) :
    (flip sq player opponent).getLsbD i = true ↔ CoordinateBit sq player opponent i := by
  rw [flip_eq_scalarValue sq player opponent]
  exact value_coordinate sq player opponent i

theorem flip2_eq (x0 x1 : Fin 66) (player opponent : Word) :
    (BoardCtx.new player opponent).flip2 x0 x1 =
      (indexedValue x0 player opponent, indexedValue x1 player opponent) := by
  unfold BoardCtx.flip2
  exact Prod.ext (flipIndex_eq_indexedValue x0 player opponent)
    (flipIndex_eq_indexedValue x1 player opponent)

theorem flip3_eq (x0 x1 x2 : Fin 66) (player opponent : Word) :
    (BoardCtx.new player opponent).flip3 x0 x1 x2 =
      (indexedValue x0 player opponent, indexedValue x1 player opponent,
        indexedValue x2 player opponent) := by
  unfold BoardCtx.flip3
  exact Prod.ext (flipIndex_eq_indexedValue x0 player opponent)
    (Prod.ext (flipIndex_eq_indexedValue x1 player opponent)
      (flipIndex_eq_indexedValue x2 player opponent))

theorem flip4_eq (x0 x1 x2 x3 : Fin 66) (player opponent : Word) :
    (BoardCtx.new player opponent).flip4 x0 x1 x2 x3 =
      (indexedValue x0 player opponent, indexedValue x1 player opponent,
        indexedValue x2 player opponent, indexedValue x3 player opponent) := by
  unfold BoardCtx.flip4
  exact Prod.ext (flipIndex_eq_indexedValue x0 player opponent)
    (Prod.ext (flipIndex_eq_indexedValue x1 player opponent)
      (Prod.ext (flipIndex_eq_indexedValue x2 player opponent)
        (flipIndex_eq_indexedValue x3 player opponent)))

def WasmSimdFlipContract : Prop :=
  ∀ (sq : Square) (player opponent : Word),
    flip sq player opponent = value sq player opponent ∧
    ∀ i, (flip sq player opponent).getLsbD i = true ↔ CoordinateBit sq player opponent i

theorem wasmSimdFlipContract_proven : WasmSimdFlipContract :=
  fun sq player opponent =>
    ⟨flip_eq_scalarValue sq player opponent,
     flip_coordinate sq player opponent⟩

end NeuralReversi.Flip.WasmSimd

import NeuralReversi.Bitboard.DeltaSwapModel

namespace NeuralReversi.Bitboard

theorem deltaShiftedBit_correct (bits : Bits) (shift i : Nat) :
    (bits <<< shift).getLsbD i = deltaShiftedBit bits shift i := by
  rw [BitVec.getLsbD_shiftLeft]
  unfold deltaShiftedBit
  by_cases within : i < 64
  · by_cases lower : shift ≤ i
    · have notLower : ¬i < shift := by omega
      simp [within, lower, notLower]
    · have below : i < shift := by omega
      simp [within, lower, below]
  · simp [within]

theorem deltaSwapRaw_bit (bits mask : Bits) (shift i : Nat) :
    (deltaSwapRaw bits mask shift).getLsbD i = deltaOutputBit bits mask shift i := by
  simp only [deltaSwapRaw, BitVec.getLsbD_xor, BitVec.getLsbD_and,
    BitVec.getLsbD_ushiftRight, deltaShiftedBit_correct, deltaOutputBit, deltaTemporaryBit]

def DeltaSwapContract : Prop :=
  ∀ (bits mask : Bits) (delta : BitVec 32),
    ∀ (checked : Bool) (i : Nat), i < 64 →
      (deltaSwap checked bits mask delta).map (fun value => value.getLsbD i) =
        if checked && decide (64 ≤ delta.toNat) then
          .error "attempt to shift left with overflow"
        else .ok (deltaOutputBit bits mask (delta.toNat % 64) i)

theorem deltaSwapContract_proven : DeltaSwapContract := by
  intro bits mask delta checked i _
  unfold deltaSwap
  split <;> simp only [Except.map, deltaSwapRaw_bit]

end NeuralReversi.Bitboard

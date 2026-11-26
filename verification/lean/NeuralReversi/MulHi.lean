import NeuralReversi.TranspositionTable.Clear

namespace NeuralReversi.MulHi

open TranspositionTable.Clear (mulHi64 mulHi64_lt)

def evaluate (a b : BitVec 64) : BitVec 64 :=
  (((a.setWidth 128) * (b.setWidth 128)) >>> (64 : Nat)).setWidth 64

def WordContract : Prop :=
  ∀ a b, (evaluate a b).toNat = mulHi64 a b ∧
    (0 < b.toNat → (evaluate a b).toNat < b.toNat)

theorem wordContract_proven : WordContract := by
  intro a b
  have aFits : a.toNat < 2 ^ 128 := by have := a.isLt; omega
  have bFits : b.toNat < 2 ^ 128 := by have := b.isLt; omega
  have productFits : a.toNat * b.toNat < 2 ^ 128 := BitVec.toNat_mul_toNat_lt
  have resultFits : a.toNat * b.toNat / 2 ^ 64 < 2 ^ 64 := by
    apply (Nat.div_lt_iff_lt_mul (by decide : 0 < 2 ^ 64)).mpr
    exact productFits
  have value : (evaluate a b).toNat = mulHi64 a b := by
    unfold evaluate mulHi64 TranspositionTable.SeqLock.modulus
    rw [BitVec.toNat_setWidth, BitVec.toNat_ushiftRight, Nat.shiftRight_eq_div_pow, BitVec.toNat_mul]
    rw [BitVec.toNat_setWidth, BitVec.toNat_setWidth, Nat.mod_eq_of_lt aFits, Nat.mod_eq_of_lt bFits,
      Nat.mod_eq_of_lt productFits, Nat.mod_eq_of_lt resultFits]
  refine ⟨value, fun positive => ?_⟩
  rw [value]
  exact mulHi64_lt a b positive

def IntervalContract : Prop :=
  ∀ a b bucket,
    (evaluate a b).toNat = bucket ↔
      bucket * 2 ^ 64 ≤ a.toNat * b.toNat ∧ a.toNat * b.toNat < (bucket + 1) * 2 ^ 64

theorem intervalContract_proven : IntervalContract := by
  intro a b bucket
  rw [(wordContract_proven a b).1]
  unfold mulHi64 TranspositionTable.SeqLock.modulus
  let product := a.toNat * b.toNat
  change product / 2 ^ 64 = bucket ↔ bucket * 2 ^ 64 ≤ product ∧ product < (bucket + 1) * 2 ^ 64
  have positive : 0 < 2 ^ 64 := by decide
  rw [← Nat.le_div_iff_mul_le positive, ← Nat.div_lt_iff_lt_mul positive]
  omega

end NeuralReversi.MulHi

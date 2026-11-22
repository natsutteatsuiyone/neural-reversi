import NeuralReversi.ScaledScore.Model
namespace NeuralReversi.ScaledScore

theorem negative_magnitude (a : Word) (ha : a.msb = true) :
    a.toInt = - ((-a).toNat : Int) := by
  rw [BitVec.toInt_eq_msb_cond, ha,
    BitVec.toNat_neg_of_pos (BitVec.pos_of_msb ha)]
  have bound := a.isLt
  change (a.toNat : Int) - 4294967296 = - ((4294967296 - a.toNat : Nat) : Int)
  omega

theorem udiv_value (a b : Word) :
    (a.udiv b).toInt = ((a.toNat / b.toNat : Nat) : Int).bmod 4294967296 := by
  rw [BitVec.toInt_eq_toNat_bmod]
  rfl

theorem neg_udiv_value (a b : Word) :
    (-(a.udiv b)).toInt = (-((a.toNat / b.toNat : Nat) : Int)).bmod 4294967296 := by
  rw [BitVec.toInt_neg, udiv_value]
  exact Int.bmod_neg_bmod

theorem sdiv_wrap (a b : Word) :
    (a.sdiv b).toInt = (a.toInt.tdiv b.toInt).bmod 4294967296 := by
  rw [BitVec.sdiv_eq]
  cases ha : a.msb <;> cases hb : b.msb
  · rw [BitVec.toInt_eq_toNat_of_msb ha, BitVec.toInt_eq_toNat_of_msb hb]
    exact udiv_value a b
  · rw [BitVec.toInt_eq_toNat_of_msb ha, negative_magnitude b hb,
      Int.tdiv_neg, ← Int.ofNat_tdiv]
    exact neg_udiv_value a (-b)
  · rw [negative_magnitude a ha, BitVec.toInt_eq_toNat_of_msb hb,
      Int.neg_tdiv, ← Int.ofNat_tdiv]
    exact neg_udiv_value (-a) b
  · rw [negative_magnitude a ha, negative_magnitude b hb,
      Int.neg_tdiv_neg, ← Int.ofNat_tdiv]
    exact udiv_value (-a) (-b)


theorem tdiv_bounds (a b : Int) (ha : -2147483648 ≤ a) (ha' : a < 2147483648)
    (noOverflow : ¬ (a = -2147483648 ∧ b = -1)) :
    -2147483648 ≤ a.tdiv b ∧ a.tdiv b < 2147483648 := by
  have haAbs : a.natAbs ≤ 2147483648 := by omega
  have hqAbs := Int.natAbs_tdiv_le_natAbs a b
  have lower : -2147483648 ≤ a.tdiv b := by omega
  refine ⟨lower, ?_⟩
  by_cases hmin : a = -2147483648
  · subst a
    have hb : b ≠ -1 := by omega
    by_cases hlarge : 1 < b.natAbs
    · have hstrict := Nat.div_lt_self (n := 2147483648) (k := b.natAbs) (by decide) hlarge
      rw [Int.natAbs_tdiv] at hqAbs
      have habs : (Int.tdiv (-2147483648) b).natAbs < 2147483648 := by
        rw [Int.natAbs_tdiv]
        exact hstrict
      omega
    · have hbsmall : b = 0 ∨ b = 1 := by omega
      rcases hbsmall with rfl | rfl <;> simp
  · have haAbsStrict : a.natAbs < 2147483648 := by omega
    omega


theorem sdiv_no_wrap (a b : Word)
    (noOverflow : ¬ (a = BitVec.intMin 32 ∧ b = -1)) :
    (a.sdiv b).toInt = a.toInt.tdiv b.toInt := by
  have safe : ¬ (a.toInt = -2147483648 ∧ b.toInt = -1) := by
    rintro ⟨ha, hb⟩
    apply noOverflow
    exact ⟨BitVec.eq_of_toInt_eq ha, BitVec.eq_of_toInt_eq hb⟩
  have bounds := tdiv_bounds a.toInt b.toInt (BitVec.le_toInt a) (BitVec.toInt_lt (x := a)) safe
  rw [sdiv_wrap]
  exact Int.bmod_eq_of_le bounds.1 bounds.2

end NeuralReversi.ScaledScore


import NeuralReversi.EndgameTiming

namespace NeuralReversi.Binary64Arithmetic

open EndgameTiming (Representable53 decode maxFinite quantum round53 roundingContract_proven)
open ScaledScore (distance roundEvenQuotient round_bounds round_nearest round_tie_even)

def precisionShift (numerator fractionalBits : Nat) : Nat :=
  max fractionalBits (numerator.log2 - 52)

def roundDyadic (numerator fractionalBits : Nat) : Nat :=
  let shift := precisionShift numerator fractionalBits
  roundEvenQuotient numerator shift * 2 ^ (shift - fractionalBits)

private theorem log2_scaled {n : Nat} (nonzero : n ≠ 0) (shift : Nat) :
    (n * 2 ^ shift).log2 = n.log2 + shift := by
  apply (Nat.log2_eq_iff (Nat.mul_ne_zero nonzero (Nat.ne_of_gt (Nat.two_pow_pos _)))).mpr
  have low := Nat.mul_le_mul_right (2 ^ shift) (Nat.log2_self_le nonzero)
  have high := Nat.mul_lt_mul_of_pos_right (Nat.lt_log2_self (n := n)) (Nat.two_pow_pos shift)
  constructor
  · rwa [← Nat.pow_add] at low
  · rwa [← Nat.pow_add, show n.log2 + 1 + shift = n.log2 + shift + 1 by omega] at high

private theorem scaled_repr (n shift : Nat) :
    Representable53 (n * 2 ^ shift) ↔ Representable53 n := by
  by_cases zero : n = 0
  · simp [zero]
  · rw [Representable53, Representable53, log2_scaled zero]
    by_cases room : 52 ≤ n.log2
    · rw [show n.log2 + shift - 52 = (n.log2 - 52) + shift by omega, Nat.pow_add]
      exact ⟨Nat.dvd_of_mul_dvd_mul_right (Nat.two_pow_pos _), fun h => Nat.mul_dvd_mul_right h _⟩
    · have lower : n.log2 - 52 = 0 := by omega
      rw [lower, Nat.pow_zero]
      exact ⟨fun _ => Nat.one_dvd _, fun _ =>
        Nat.dvd_trans (Nat.pow_dvd_pow 2 (by omega : n.log2 + shift - 52 ≤ shift)) (Nat.dvd_mul_left _ _)⟩

private theorem small_repr (n : Nat) (bound : n ≤ 2 ^ 53) : Representable53 n := by
  by_cases edge : n = 2 ^ 53
  · rw [edge, Representable53, Nat.log2_two_pow]
    exact Nat.pow_dvd_pow 2 (by decide)
  · by_cases zero : n = 0
    · subst n
      exact Nat.dvd_zero _
    · have logBound : n.log2 < 53 := by
        apply Nat.lt_of_not_ge
        intro above
        have power := Nat.pow_le_pow_right (n := 2) (by decide) above
        have floor := Nat.log2_self_le zero
        omega
      simp [Representable53, show n.log2 - 52 = 0 by omega]

private theorem round_normalized (n shift : Nat) (onlyPrecision : shift ≤ n.log2 - 52) :
    roundDyadic n shift * 2 ^ shift = round53 n := by
  rw [roundDyadic, precisionShift, Nat.max_eq_right onlyPrecision, round53, Nat.mul_assoc,
    ← Nat.pow_add, Nat.sub_add_cancel onlyPrecision]

def DyadicRoundingContract : Prop :=
  ∀ n fractionalBits,
    let value := roundDyadic n fractionalBits
    let shift := precisionShift n fractionalBits
    Representable53 value ∧
    (∀ other, Representable53 other →
      distance n ((value * 2 ^ fractionalBits : Nat) : Int) 1 ≤
        distance n ((other * 2 ^ fractionalBits : Nat) : Int) 1) ∧
    (2 ^ shift = 2 * (n % 2 ^ shift) → roundEvenQuotient n shift % 2 = 0)

theorem dyadicRoundingContract_proven : DyadicRoundingContract := by
  intro n fractionalBits value shift
  refine ⟨?_, ?_, round_tie_even n shift⟩
  · by_cases normal : fractionalBits ≤ n.log2 - 52
    · apply (scaled_repr _ _).1
      rw [show value * 2 ^ fractionalBits = round53 n from round_normalized n fractionalBits normal]
      exact (roundingContract_proven.1 n).1
    · have usesFraction : shift = fractionalBits := by unfold shift precisionShift; omega
      have less : n < 2 ^ (fractionalBits + 53) := Nat.lt_of_lt_of_le Nat.lt_log2_self
        (Nat.pow_le_pow_right (n := 2) (by decide) (by omega))
      have quotient : n / 2 ^ fractionalBits < 2 ^ 53 := by
        apply (Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _)).2
        rwa [← Nat.pow_add, Nat.add_comm]
      have rounded := (round_bounds n fractionalBits).2
      apply small_repr
      change roundEvenQuotient n shift * 2 ^ (shift - fractionalBits) ≤ 2 ^ 53
      rw [usesFraction, Nat.sub_self, Nat.pow_zero, Nat.mul_one]
      omega
  · intro other representable
    by_cases normal : fractionalBits ≤ n.log2 - 52
    · rw [show value * 2 ^ fractionalBits = round53 n from round_normalized n fractionalBits normal]
      exact (roundingContract_proven.1 n).2 _ ((scaled_repr _ _).2 representable)
    · have usesFraction : value = roundEvenQuotient n fractionalBits := by
        simp [value, roundDyadic, precisionShift, Nat.max_eq_left (by omega : n.log2 - 52 ≤ fractionalBits)]
      rw [usesFraction]
      simpa only [Int.natCast_mul, Int.natCast_one, Int.mul_one, distance] using round_nearest n fractionalBits other

def finiteUnits (bits : BitVec 64) : Option Int :=
  match decode (bits &&& 0x7fffffffffffffff#64) with
  | some (.finite value) => some (if bits.msb then -(Int.ofNat value) else Int.ofNat value)
  | _ => none

def roundSigned (numerator : Int) (fractionalBits : Nat) : Option Int :=
  let absolute := roundDyadic numerator.natAbs fractionalBits
  if maxFinite < absolute then none
  else some (if numerator < 0 then -(Int.ofNat absolute) else Int.ofNat absolute)

def add (a b : Int) : Option Int := roundSigned (a + b) 0
def sub (a b : Int) : Option Int := roundSigned (a - b) 0
def mul (a b : Int) : Option Int := roundSigned (a * b) 1074

def FiniteContract : Prop :=
  (∀ numerator fractionalBits units, roundSigned numerator fractionalBits = some units →
    units.natAbs ≤ maxFinite ∧ Representable53 units.natAbs) ∧
  (∀ score : Nat, score ≤ 32767 →
    score * quantum ≤ maxFinite ∧ Representable53 (score * quantum))

set_option exponentiation.threshold 4096 in
set_option maxRecDepth 8192 in
theorem finiteContract_proven : FiniteContract := by
  constructor
  · intro numerator fractionalBits units rounded
    let magnitude := roundDyadic numerator.natAbs fractionalBits
    change (if maxFinite < magnitude then none else
      some (if numerator < 0 then -(Int.ofNat magnitude) else Int.ofNat magnitude)) = some units at rounded
    split at rounded
    · cases rounded
    · rename_i fits
      have representation := (dyadicRoundingContract_proven numerator.natAbs fractionalBits).1
      rw [← Option.some.inj rounded]
      by_cases negative : numerator < 0 <;> simp [negative] <;> exact ⟨Nat.le_of_not_gt fits, representation⟩
  · intro score bounded
    have limit : 32767 * quantum ≤ maxFinite := by decide +kernel
    exact ⟨Nat.le_trans (Nat.mul_le_mul_right quantum bounded) limit,
      (scaled_repr score 1074).2 (small_repr score (by omega))⟩

set_option exponentiation.threshold 4096 in
set_option maxRecDepth 8192 in
example :
    mul (Int.ofNat quantum / 2) 1 = some 0 ∧
    mul (Int.ofNat quantum / 2) 3 = some 2 ∧
    mul (Int.ofNat quantum / 2) (-3) = some (-2) := by decide +kernel

end NeuralReversi.Binary64Arithmetic

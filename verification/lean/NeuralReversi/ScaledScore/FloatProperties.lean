import NeuralReversi.ScaledScore.FloatRounding
set_option maxRecDepth 4096
namespace NeuralReversi.ScaledScore

theorem scale_pack (s : Bool) (e f : Nat) (he : 0 < e) (he' : e < 256)
    (hf : f < 8388608) :
    scaleIntegerCastBy256 (packF32 s e f) = packF32 s (e - 8) f := by
  have hnonzero : packF32 s e f ≠ 0#32 := by
    intro h
    have hz := congrArg f32Exponent h
    rw [pack_exponent s e f he'] at hz
    change e = 0 at hz
    omega
  simp only [scaleIntegerCastBy256, ite_eq_right hnonzero,
    pack_sign, pack_exponent s e f he', pack_fraction s e f hf]

def RepresentsRoundedMagnitude (rounded : Nat) (bits : FloatBits) : Prop :=
  let E := f32Exponent bits
  let M := 8388608 + f32Fraction bits
  119 ≤ E ∧ E ≤ 150 ∧
    (142 ≤ E → rounded = M * 2 ^ (E - 142)) ∧
    (E < 142 → rounded * 2 ^ (142 - E) = M)

def ScaledRepresentation (n : Nat) (bits : FloatBits) : Prop :=
  RepresentsRoundedMagnitude (roundedMagnitude n) bits

theorem nonzero_output (raw : Word) (hnz : raw.toInt.natAbs ≠ 0) :
    f32Sign (toDiscDiffF32 ⟨raw⟩) = decide (raw.toInt < 0) ∧
    ScaledRepresentation raw.toInt.natAbs (toDiscDiffF32 ⟨raw⟩) := by
  let n := raw.toInt.natAbs
  let sign := decide (raw.toInt < 0)
  have hn : n ≤ 2147483648 := by
    have hlo := BitVec.le_toInt raw
    have hhi := BitVec.toInt_lt (x := raw)
    dsimp [n]
    omega
  have he := magnitude_exponent_bound n hn
  by_cases small : n.log2 ≤ 23
  · have hb := small_significand_bounds n hnz small
    have hf : n * 2 ^ (23 - n.log2) - 8388608 < 8388608 := by omega
    have out : toDiscDiffF32 ⟨raw⟩ =
        packF32 sign (n.log2 + 119) (n * 2 ^ (23 - n.log2) - 8388608) := by
      change scaleIntegerCastBy256 (if n = 0 then 0#32 else if n.log2 ≤ 23 then
          packF32 sign (n.log2 + 127) (n * 2 ^ (23 - n.log2) - 8388608)
          else if 2 ^ 24 ≤ roundEvenQuotient n (precisionShift n) then
            packF32 sign (n.log2 + 128) 0
          else packF32 sign (n.log2 + 127) (roundEvenQuotient n (precisionShift n) - 8388608)) = _
      rw [ite_eq_right hnz, ite_eq_left small, scale_pack _ _ _ (by omega) (by omega) hf]
      have hx : n.log2 + 127 - 8 = n.log2 + 119 := by omega
      rw [hx]
    rw [out, pack_sign]
    refine ⟨rfl, ?_⟩
    simp only [ScaledRepresentation, RepresentsRoundedMagnitude]
    rw [pack_exponent sign (n.log2 + 119) _ (by omega), pack_fraction sign (n.log2 + 119) _ hf]
    have hm : 8388608 + (n * 2 ^ (23 - n.log2) - 8388608) = n * 2 ^ (23 - n.log2) := by omega
    rw [hm]
    refine ⟨by omega, by omega, ?_, ?_⟩
    · intro hh
      have hexact : n.log2 = 23 := by omega
      have val := small_scaled_relation n small
      simp only [hexact] at val ⊢
      simpa using val
    · intro hh
      exact small_scaled_relation n small
  · have large : 23 < n.log2 := by omega
    have hb := large_significand_bounds n hnz large
    by_cases carry : 2 ^ 24 ≤ roundEvenQuotient n (precisionShift n)
    · have hec := carry_exponent_bound n hn carry
      have out : toDiscDiffF32 ⟨raw⟩ = packF32 sign (n.log2 + 120) 0 := by
        change scaleIntegerCastBy256 (if n = 0 then 0#32 else if n.log2 ≤ 23 then
          packF32 sign (n.log2 + 127) (n * 2 ^ (23 - n.log2) - 8388608)
          else if 2 ^ 24 ≤ roundEvenQuotient n (precisionShift n) then
            packF32 sign (n.log2 + 128) 0
          else packF32 sign (n.log2 + 127) (roundEvenQuotient n (precisionShift n) - 8388608)) = _
        rw [ite_eq_right hnz, ite_eq_right small, ite_eq_left carry,
          scale_pack _ _ _ (by omega) (by omega) (by decide)]
        have hx : n.log2 + 128 - 8 = n.log2 + 120 := by omega
        rw [hx]
      rw [out, pack_sign]
      refine ⟨rfl, ?_⟩
      simp only [ScaledRepresentation, RepresentsRoundedMagnitude]
      rw [pack_exponent sign (n.log2 + 120) 0 (by omega), pack_fraction sign (n.log2 + 120) 0 (by decide)]
      simp only [Nat.add_zero]
      refine ⟨by omega, by omega, ?_, ?_⟩
      · intro hh
        exact carry_scaled_relation n hnz large carry
      · intro hh; omega
    · have hf : roundEvenQuotient n (precisionShift n) - 8388608 < 8388608 := by omega
      have out : toDiscDiffF32 ⟨raw⟩ = packF32 sign (n.log2 + 119)
          (roundEvenQuotient n (precisionShift n) - 8388608) := by
        change scaleIntegerCastBy256 (if n = 0 then 0#32 else if n.log2 ≤ 23 then
          packF32 sign (n.log2 + 127) (n * 2 ^ (23 - n.log2) - 8388608)
          else if 2 ^ 24 ≤ roundEvenQuotient n (precisionShift n) then
            packF32 sign (n.log2 + 128) 0
          else packF32 sign (n.log2 + 127) (roundEvenQuotient n (precisionShift n) - 8388608)) = _
        rw [ite_eq_right hnz, ite_eq_right small, ite_eq_right carry,
          scale_pack _ _ _ (by omega) (by omega) hf]
        have hx : n.log2 + 127 - 8 = n.log2 + 119 := by omega
        rw [hx]
      rw [out, pack_sign]
      refine ⟨rfl, ?_⟩
      simp only [ScaledRepresentation, RepresentsRoundedMagnitude]
      rw [pack_exponent sign (n.log2 + 119) _ (by omega), pack_fraction sign (n.log2 + 119) _ hf]
      have hm : 8388608 + (roundEvenQuotient n (precisionShift n) - 8388608) =
          roundEvenQuotient n (precisionShift n) := by omega
      rw [hm]
      refine ⟨by omega, by omega, ?_, ?_⟩
      · intro hh
        exact large_scaled_relation n
      · intro hh; omega


theorem zero_output (raw : Word) (hz : raw.toInt.natAbs = 0) :
    toDiscDiffF32 ⟨raw⟩ = 0#32 := by
  change scaleIntegerCastBy256 (integerCastBits raw) = 0#32
  simp only [integerCastBits, hz, ite_true]
  rfl

theorem output_zero_iff (raw : Word) : toDiscDiffF32 ⟨raw⟩ = 0#32 ↔ raw = 0#32 := by
  constructor
  · intro hz
    by_cases hn : raw.toInt.natAbs = 0
    · apply BitVec.eq_of_toInt_eq
      change raw.toInt = 0
      omega
    · have h := (nonzero_output raw hn).2.1
      rw [hz] at h
      change 119 ≤ 0 at h
      omega
  · intro h
    subst raw
    rfl

def magnitudeError (input rounded : Nat) : Nat := ((input : Int) - rounded).natAbs

def NearestEvenMagnitude (input rounded : Nat) : Prop :=
  let unit := 2 ^ (input.log2 - 23)
  rounded % unit = 0 ∧
    (∀ candidate : Int,
      magnitudeError input rounded ≤ ((input : Int) - candidate * unit).natAbs) ∧
    (2 * magnitudeError input rounded = unit → (rounded / unit) % 2 = 0)

theorem round_tie_even_from_error (n shift : Nat)
    (tie : 2 * distance n (roundEvenQuotient n shift) (2 ^ shift) = 2 ^ shift) :
    roundEvenQuotient n shift % 2 = 0 := by
  have hu : 0 < 2 ^ shift := Nat.two_pow_pos shift
  have hr := Nat.mod_lt n hu
  have hn : (n : Int) = (n / 2 ^ shift : Nat) * (2 ^ shift : Nat) + (n % 2 ^ shift : Nat) := by
    have decomposition := Nat.mod_add_div n (2 ^ shift)
    rw [Nat.mul_comm (2 ^ shift)] at decomposition
    have h := congrArg (fun x : Nat => (x : Int)) decomposition
    simp only [Int.natCast_add, Int.natCast_mul] at h
    omega
  have inputTie : 2 ^ shift = 2 * (n % 2 ^ shift) := by
    simp only [roundEvenQuotient, distance] at tie
    split at tie
    · rw [Int.natCast_add, Int.natCast_one, Int.add_mul, Int.one_mul] at tie
      omega
    · omega
  exact round_tie_even n shift inputTie

theorem roundedMagnitude_nearest (n : Nat) : NearestEvenMagnitude n (roundedMagnitude n) := by
  have hu : 0 < 2 ^ precisionShift n := Nat.two_pow_pos _
  have error : magnitudeError n (roundedMagnitude n) =
      distance n (roundEvenQuotient n (precisionShift n)) (2 ^ precisionShift n) := by
    simp only [magnitudeError, roundedMagnitude, distance, Int.natCast_mul]
  change roundedMagnitude n % 2 ^ precisionShift n = 0 ∧
    (∀ candidate : Int, magnitudeError n (roundedMagnitude n) ≤
      ((n : Int) - candidate * (2 ^ precisionShift n : Nat)).natAbs) ∧
    (2 * magnitudeError n (roundedMagnitude n) = 2 ^ precisionShift n →
      (roundedMagnitude n / 2 ^ precisionShift n) % 2 = 0)
  refine ⟨by simp [roundedMagnitude], ?_, ?_⟩
  · rw [error]
    exact round_nearest n (precisionShift n)
  · intro tie
    have even := round_tie_even_from_error n (precisionShift n) (by rwa [error] at tie)
    simpa only [roundedMagnitude, Nat.mul_div_cancel _ hu] using even

def RepresentsConversionResult (raw : Word) (rounded : Nat) : Prop :=
  if raw = 0#32 then toDiscDiffF32 ⟨raw⟩ = 0#32 ∧ rounded = 0
  else f32Sign (toDiscDiffF32 ⟨raw⟩) = decide (raw.toInt < 0) ∧
    RepresentsRoundedMagnitude rounded (toDiscDiffF32 ⟨raw⟩)

theorem model_represents_rounded (raw : Word) :
    RepresentsConversionResult raw (roundedMagnitude raw.toInt.natAbs) := by
  by_cases zero : raw = 0#32
  · subst raw
    change toDiscDiffF32 ⟨0#32⟩ = 0#32 ∧ roundedMagnitude 0 = 0
    exact ⟨rfl, rfl⟩
  · have nonzero : raw.toInt.natAbs ≠ 0 := by
      intro h
      apply zero
      apply BitVec.eq_of_toInt_eq
      change raw.toInt = 0
      omega
    simpa only [RepresentsConversionResult, zero, ↓reduceIte, ScaledRepresentation]
      using nonzero_output raw nonzero

def FloatConversionContract : Prop :=
  ∀ raw : Word, ∃ rounded : Nat,
    RepresentsConversionResult raw rounded ∧ NearestEvenMagnitude raw.toInt.natAbs rounded

theorem floatConversionContract_proven : FloatConversionContract := by
  intro raw
  exact ⟨roundedMagnitude raw.toInt.natAbs, model_represents_rounded raw,
    roundedMagnitude_nearest raw.toInt.natAbs⟩

def FloatPrecisionContract : Prop :=
  ∀ raw : Word, ∃ rounded : Nat,
    RepresentsConversionResult raw rounded ∧
    2 * magnitudeError raw.toInt.natAbs rounded ≤ 2 ^ (raw.toInt.natAbs.log2 - 23) ∧
    (raw.toInt.natAbs ≤ 16777216 → rounded = raw.toInt.natAbs)

theorem floatPrecisionContract_proven : FloatPrecisionContract := by
  intro raw
  refine ⟨roundedMagnitude raw.toInt.natAbs, model_represents_rounded raw, ?_,
    exact_small_magnitude raw.toInt.natAbs⟩
  simpa only [magnitudeError, roundedMagnitude, distance, Int.natCast_mul, precisionShift]
    using rounding_half_error raw.toInt.natAbs (precisionShift raw.toInt.natAbs)

end NeuralReversi.ScaledScore

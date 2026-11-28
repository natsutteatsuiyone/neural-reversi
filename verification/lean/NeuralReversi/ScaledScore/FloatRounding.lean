module
public import NeuralReversi.ScaledScore.FloatModel

@[expose] public section

namespace NeuralReversi.ScaledScore

def distance (n : Nat) (q : Int) (unit : Nat) : Nat :=
  ((n : Int) - q * unit).natAbs

theorem floor_nearest (n q unit rem : Nat) (hu : 0 < unit)
    (hn : n = q * unit + rem) (hr : rem < unit) (half : 2 * rem ≤ unit) :
    ∀ k : Int, distance n q unit ≤ distance n k unit := by
  have hni : (n : Int) = (q : Int) * unit + rem := by
    have h := congrArg (fun x : Nat => (x : Int)) hn
    simpa only [Int.natCast_add, Int.natCast_mul] using h
  intro k
  unfold distance
  by_cases hk : k ≤ (q : Int)
  · have hmul := Int.mul_le_mul_of_nonneg_right hk (show (0 : Int) ≤ unit by omega)
    omega
  · have hk' : (q : Int) + 1 ≤ k := by omega
    have hmul := Int.mul_le_mul_of_nonneg_right hk' (show (0 : Int) ≤ unit by omega)
    rw [Int.add_mul, Int.one_mul] at hmul
    omega

theorem ceil_nearest (n q unit rem : Nat) (hu : 0 < unit)
    (hn : n = q * unit + rem) (hr : rem < unit) (half : unit ≤ 2 * rem) :
    ∀ k : Int, distance n ((q : Int) + 1) unit ≤ distance n k unit := by
  have hni : (n : Int) = (q : Int) * unit + rem := by
    have h := congrArg (fun x : Nat => (x : Int)) hn
    simpa only [Int.natCast_add, Int.natCast_mul] using h
  intro k
  unfold distance
  rw [Int.add_mul, Int.one_mul]
  by_cases hk : k ≤ (q : Int)
  · have hmul := Int.mul_le_mul_of_nonneg_right hk (show (0 : Int) ≤ unit by omega)
    omega
  · have hk' : (q : Int) + 1 ≤ k := by omega
    have hmul := Int.mul_le_mul_of_nonneg_right hk' (show (0 : Int) ≤ unit by omega)
    rw [Int.add_mul, Int.one_mul] at hmul
    omega


theorem round_nearest (n shift : Nat) :
    ∀ k : Int, distance n (roundEvenQuotient n shift) (2 ^ shift) ≤ distance n k (2 ^ shift) := by
  have hu : 0 < 2 ^ shift := Nat.two_pow_pos shift
  have hr := Nat.mod_lt n hu
  have hn : n = n / 2 ^ shift * 2 ^ shift + n % 2 ^ shift := by
    have h := Nat.mod_add_div n (2 ^ shift)
    rw [Nat.mul_comm (2 ^ shift)] at h
    omega
  simp only [roundEvenQuotient]
  split <;> rename_i hround
  · simpa only [Int.natCast_add, Int.natCast_one] using
      ceil_nearest n (n / 2 ^ shift) (2 ^ shift) (n % 2 ^ shift) hu hn hr (by omega)
  · exact floor_nearest n (n / 2 ^ shift) (2 ^ shift) (n % 2 ^ shift) hu hn hr (by omega)

theorem round_tie_even (n shift : Nat) (htie : 2 ^ shift = 2 * (n % 2 ^ shift)) :
    roundEvenQuotient n shift % 2 = 0 := by
  simp only [roundEvenQuotient]
  split <;> omega


theorem round_bounds (n shift : Nat) :
    n / 2 ^ shift ≤ roundEvenQuotient n shift ∧
    roundEvenQuotient n shift ≤ n / 2 ^ shift + 1 := by
  simp only [roundEvenQuotient]
  split <;> omega

theorem large_significand_bounds (n : Nat) (hn : n ≠ 0) (he : 23 < n.log2) :
    2 ^ 23 ≤ roundEvenQuotient n (precisionShift n) ∧
    roundEvenQuotient n (precisionShift n) ≤ 2 ^ 24 := by
  have hnlow := Nat.log2_self_le hn
  have hnhi := Nat.lt_log2_self (n := n)
  have p23 : 2 ^ 23 * 2 ^ (n.log2 - 23) = 2 ^ n.log2 := by
    rw [← Nat.pow_add]
    congr 1
    omega
  have p24 : 2 ^ 24 * 2 ^ (n.log2 - 23) = 2 ^ (n.log2 + 1) := by
    rw [← Nat.pow_add]
    congr 1
    omega
  have hqlo : 2 ^ 23 ≤ n / 2 ^ (n.log2 - 23) := by
    apply (Nat.le_div_iff_mul_le (Nat.two_pow_pos _)).mpr
    rwa [p23]
  have hqhi : n / 2 ^ (n.log2 - 23) < 2 ^ 24 := by
    apply (Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _)).mpr
    rwa [p24]
  have hb := round_bounds n (precisionShift n)
  unfold precisionShift at *
  omega

theorem pack_sign (s : Bool) (e f : Nat) : f32Sign (packF32 s e f) = s := by
  simp only [f32Sign, packF32, BitVec.msb_append]
  simp

theorem pack_exponent (s : Bool) (e f : Nat) (he : e < 256) :
    f32Exponent (packF32 s e f) = e := by
  unfold f32Exponent packF32
  rw [BitVec.extractLsb'_append_eq_of_add_le (by decide)]
  rw [BitVec.extractLsb'_append_eq_left]
  change e % 256 = e
  exact Nat.mod_eq_of_lt he

theorem pack_fraction (s : Bool) (e f : Nat) (hf : f < 8388608) :
    f32Fraction (packF32 s e f) = f := by
  unfold f32Fraction packF32
  rw [BitVec.extractLsb'_append_eq_of_add_le (by decide)]
  rw [BitVec.extractLsb'_append_eq_right]
  change f % 8388608 = f
  exact Nat.mod_eq_of_lt hf


theorem magnitude_exponent_bound (n : Nat) (hn : n ≤ 2147483648) : n.log2 ≤ 31 := by
  by_cases hz : n = 0
  · subst n; decide
  · have h : n.log2 < 32 := (Nat.log2_lt hz).mpr (by omega)
    omega

theorem small_significand_bounds (n : Nat) (hn : n ≠ 0) (he : n.log2 ≤ 23) :
    2 ^ 23 ≤ n * 2 ^ (23 - n.log2) ∧ n * 2 ^ (23 - n.log2) < 2 ^ 24 := by
  have low := Nat.mul_le_mul_right (2 ^ (23 - n.log2)) (Nat.log2_self_le hn)
  have high := Nat.mul_lt_mul_of_pos_right (Nat.lt_log2_self (n := n))
    (Nat.two_pow_pos (23 - n.log2))
  have hpow : 2 ^ n.log2 * 2 ^ (23 - n.log2) = 2 ^ 23 := by
    rw [← Nat.pow_add]
    congr 1
    omega
  have hpow' : 2 ^ (n.log2 + 1) * 2 ^ (23 - n.log2) = 2 ^ 24 := by
    rw [← Nat.pow_add]
    congr 1
    omega
  rw [hpow] at low
  rw [hpow'] at high
  exact ⟨low, high⟩

theorem round_zero_shift (n : Nat) : roundEvenQuotient n 0 = n := by
  simp [roundEvenQuotient]
  omega

theorem carry_exponent_bound (n : Nat) (hn : n ≤ 2147483648)
    (hc : 2 ^ 24 ≤ roundEvenQuotient n (precisionShift n)) : n.log2 ≤ 30 := by
  have he := magnitude_exponent_bound n hn
  by_cases h31 : n.log2 = 31
  · have hnz : n ≠ 0 := by intro h; subst n; simp at h31
    have hlow := Nat.log2_self_le hnz
    rw [h31] at hlow
    have hvalue : n = 2147483648 := by omega
    subst n
    have hcalc : roundEvenQuotient 2147483648 (precisionShift 2147483648) = 8388608 := by
      decide
    rw [hcalc] at hc
    omega
  · omega

def roundedMagnitude (n : Nat) : Nat := roundEvenQuotient n (precisionShift n) * 2 ^ precisionShift n

theorem small_scaled_relation (n : Nat) (he : n.log2 ≤ 23) :
    roundedMagnitude n * 2 ^ (142 - (n.log2 + 119)) = n * 2 ^ (23 - n.log2) := by
  have hs : precisionShift n = 0 := by unfold precisionShift; omega
  have hx : 142 - (n.log2 + 119) = 23 - n.log2 := by omega
  simp only [roundedMagnitude, hs, round_zero_shift, Nat.pow_zero, Nat.mul_one, hx]

theorem large_scaled_relation (n : Nat) :
    roundedMagnitude n = roundEvenQuotient n (precisionShift n) * 2 ^ ((n.log2 + 119) - 142) := by
  have hx : (n.log2 + 119) - 142 = n.log2 - 23 := by omega
  simp only [roundedMagnitude, precisionShift, hx]

set_option maxRecDepth 4096 in
theorem carry_scaled_relation (n : Nat) (hnz : n ≠ 0) (he : 23 < n.log2)
    (hc : 2 ^ 24 ≤ roundEvenQuotient n (precisionShift n)) :
    roundedMagnitude n = 2 ^ 23 * 2 ^ ((n.log2 + 120) - 142) := by
  have bounds := large_significand_bounds n hnz he
  have hq : roundEvenQuotient n (precisionShift n) = 2 ^ 24 := by omega
  rw [roundedMagnitude, hq, ← Nat.pow_add, ← Nat.pow_add]
  congr 1
  unfold precisionShift
  omega


theorem rounding_half_error (n shift : Nat) :
    2 * distance n (roundEvenQuotient n shift) (2 ^ shift) ≤ 2 ^ shift := by
  have hu : 0 < 2 ^ shift := Nat.two_pow_pos shift
  have hr := Nat.mod_lt n hu
  have hn : (n : Int) = (n / 2 ^ shift : Nat) * (2 ^ shift : Nat) + (n % 2 ^ shift : Nat) := by
    have h := Nat.mod_add_div n (2 ^ shift)
    rw [Nat.mul_comm (2 ^ shift)] at h
    have h' := congrArg (fun x : Nat => (x : Int)) h
    simp only [Int.natCast_add, Int.natCast_mul] at h'
    omega
  simp only [roundEvenQuotient]
  split <;> rename_i h
  · unfold distance
    rw [Int.natCast_add, Int.natCast_one, Int.add_mul, Int.one_mul]
    omega
  · unfold distance
    omega

theorem exact_small_magnitude (n : Nat) (hn : n ≤ 16777216) : roundedMagnitude n = n := by
  by_cases hz : n = 0
  · subst n; rfl
  have he : n.log2 ≤ 24 := by
    have h := (Nat.log2_lt hz).mpr (show n < 2 ^ 25 by omega)
    omega
  by_cases h24 : n.log2 = 24
  · have hlo := Nat.log2_self_le hz
    rw [h24] at hlo
    have hv : n = 16777216 := by omega
    subst n
    decide
  · have hs : precisionShift n = 0 := by unfold precisionShift; omega
    simp only [roundedMagnitude, hs, round_zero_shift, Nat.pow_zero, Nat.mul_one]

end NeuralReversi.ScaledScore

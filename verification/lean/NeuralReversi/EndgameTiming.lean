import NeuralReversi.EndgameProgress
import NeuralReversi.ScaledScore.FloatRounding

namespace NeuralReversi.EndgameTiming

open ScaledScore (roundEvenQuotient round_bounds)

private theorem quotient_mono (shift : Nat) {a b : Nat} (ordered : a ≤ b) :
    roundEvenQuotient a shift ≤ roundEvenQuotient b shift := by
  have floorOrder := Nat.div_le_div_right (c := 2 ^ shift) ordered
  by_cases different : a / 2 ^ shift < b / 2 ^ shift
  · have up := (round_bounds a shift).2
    have down := (round_bounds b shift).1
    omega
  · have same : a / 2 ^ shift = b / 2 ^ shift := by omega
    have aParts := Nat.mod_add_div a (2 ^ shift)
    have bParts := Nat.mod_add_div b (2 ^ shift)
    have remainder : a % 2 ^ shift ≤ b % 2 ^ shift := by
      rw [same] at aParts
      omega
    simp only [roundEvenQuotient, same]
    split <;> split <;> omega

def round53 (units : Nat) : Nat :=
  let shift := units.log2 - 52
  roundEvenQuotient units shift * 2 ^ shift

private theorem power_multiple {power shift : Nat} (le : shift ≤ power) :
    2 ^ (power - shift) * 2 ^ shift = 2 ^ power := by
  rw [← Nat.pow_add, Nat.sub_add_cancel le]

private theorem round53_binade {n : Nat} (nonzero : n ≠ 0) :
    2 ^ n.log2 ≤ round53 n ∧ round53 n ≤ 2 ^ (n.log2 + 1) := by
  let shift := n.log2 - 52
  let unit := 2 ^ shift
  have unitPositive : 0 < unit := Nat.two_pow_pos _
  have shiftLe : shift ≤ n.log2 := Nat.sub_le _ _
  have left : 2 ^ (n.log2 - shift) * unit = 2 ^ n.log2 := power_multiple shiftLe
  have right : 2 ^ (n.log2 + 1 - shift) * unit = 2 ^ (n.log2 + 1) :=
    power_multiple (Nat.le_succ_of_le shiftLe)
  have low : 2 ^ (n.log2 - shift) ≤ n / unit :=
    (Nat.le_div_iff_mul_le unitPositive).mpr (left ▸ Nat.log2_self_le nonzero)
  have high : n / unit < 2 ^ (n.log2 + 1 - shift) :=
    (Nat.div_lt_iff_lt_mul unitPositive).mpr (right ▸ Nat.lt_log2_self)
  have rounded : n / unit ≤ roundEvenQuotient n shift ∧ roundEvenQuotient n shift ≤ n / unit + 1 :=
    round_bounds n shift
  change 2 ^ n.log2 ≤ roundEvenQuotient n shift * unit ∧
    roundEvenQuotient n shift * unit ≤ 2 ^ (n.log2 + 1)
  rw [← left, ← right]
  exact ⟨Nat.mul_le_mul_right _ (Nat.le_trans low rounded.1),
    Nat.mul_le_mul_right _ (Nat.le_trans rounded.2 high)⟩

theorem round53_mono {a b : Nat} (ordered : a ≤ b) : round53 a ≤ round53 b := by
  by_cases zero : a = 0
  · subst a
    exact Nat.zero_le _
  have bNonzero : b ≠ 0 := by omega
  have exponent : a.log2 ≤ b.log2 :=
    (Nat.le_log2 bNonzero).mpr (Nat.le_trans (Nat.log2_self_le zero) ordered)
  by_cases same : a.log2 = b.log2
  · simp only [round53, same]
    exact Nat.mul_le_mul_right _ (quotient_mono _ ordered)
  · have gap : a.log2 + 1 ≤ b.log2 := by omega
    exact Nat.le_trans (round53_binade zero).2
      (Nat.le_trans (Nat.pow_le_pow_right (by decide) gap) (round53_binade bNonzero).1)

def Representable53 (units : Nat) : Prop := 2 ^ (units.log2 - 52) ∣ units

private theorem round53_repr (n : Nat) : Representable53 (round53 n) := by
  by_cases zero : n = 0
  · subst n
    exact ⟨0, rfl⟩
  obtain ⟨lower, upper⟩ := round53_binade zero
  by_cases carry : round53 n = 2 ^ (n.log2 + 1)
  · rw [carry, Representable53, Nat.log2_two_pow]
    refine ⟨2 ^ (n.log2 + 1 - (n.log2 + 1 - 52)), ?_⟩
    rw [Nat.mul_comm, power_multiple (Nat.sub_le _ _)]
  · have positive : round53 n ≠ 0 := by
      have power := Nat.two_pow_pos n.log2
      omega
    have exponent : (round53 n).log2 = n.log2 :=
      (Nat.log2_eq_iff positive).mpr ⟨lower, Nat.lt_of_le_of_ne upper carry⟩
    rw [Representable53, exponent]
    exact ⟨roundEvenQuotient n (n.log2 - 52), Nat.mul_comm _ _⟩

open ScaledScore (distance) in
private theorem nearest_grid (n multiple : Nat) :
    distance n (round53 n : Nat) 1 ≤ distance n (multiple * 2 ^ (n.log2 - 52) : Nat) 1 := by
  simpa [round53, distance, Int.natCast_mul] using
    ScaledScore.round_nearest n (n.log2 - 52) multiple

open ScaledScore (distance) in
private theorem round53_nearest (n target : Nat) (represented : Representable53 target) :
    distance n (round53 n : Nat) 1 ≤ distance n target 1 := by
  by_cases zero : n = 0
  · subst n
    exact Nat.zero_le _
  let shift := n.log2 - 52
  have low := Nat.log2_self_le zero
  have high := Nat.lt_log2_self (n := n)
  by_cases below : target < 2 ^ n.log2
  · have nearest := nearest_grid n (2 ^ (n.log2 - shift))
    change distance n (round53 n : Nat) 1 ≤ distance n (2 ^ (n.log2 - shift) * 2 ^ shift : Nat) 1 at nearest
    rw [power_multiple (Nat.sub_le _ _)] at nearest
    exact Nat.le_trans nearest (by unfold distance; omega)
  · by_cases above : 2 ^ (n.log2 + 1) ≤ target
    · have nearest := nearest_grid n (2 ^ (n.log2 + 1 - shift))
      change distance n (round53 n : Nat) 1 ≤ distance n (2 ^ (n.log2 + 1 - shift) * 2 ^ shift : Nat) 1 at nearest
      have bound : shift ≤ n.log2 + 1 := Nat.le_succ_of_le (Nat.sub_le _ _)
      rw [power_multiple bound] at nearest
      exact Nat.le_trans nearest (by unfold distance; omega)
    · have positive : target ≠ 0 := by
        have power := Nat.two_pow_pos n.log2
        omega
      have exponent : target.log2 = n.log2 :=
        (Nat.log2_eq_iff positive).mpr ⟨Nat.le_of_not_gt below, Nat.lt_of_not_ge above⟩
      rw [Representable53, exponent] at represented
      obtain ⟨multiple, exactTarget⟩ := represented
      rw [exactTarget, Nat.mul_comm]
      exact nearest_grid n multiple

open ScaledScore (distance) in
def RoundingContract : Prop :=
  (∀ n, Representable53 (round53 n) ∧
    ∀ target, Representable53 target → distance n (round53 n : Nat) 1 ≤ distance n target 1) ∧
  (∀ a b, a ≤ b → round53 a ≤ round53 b) ∧
  (∀ n,
    let shift := n.log2 - 52
    2 ^ shift = 2 * (n % 2 ^ shift) → roundEvenQuotient n shift % 2 = 0)

theorem roundingContract_proven : RoundingContract :=
  ⟨fun n => ⟨round53_repr n, round53_nearest n⟩, fun _ _ => round53_mono,
    fun n => ScaledScore.round_tie_even n (n.log2 - 52)⟩

inductive Magnitude where
  | finite (units : Nat)
  | infinity
  | nan
  deriving DecidableEq, Repr

def quantum : Nat := 2 ^ 1074

def maxFinite : Nat := (2 ^ 53 - 1) * 2 ^ 2045

def decode (word : BitVec 64) : Option Magnitude :=
  let exponent := (word.extractLsb' 52 11).toNat
  let fraction := (word.extractLsb' 0 52).toNat
  if exponent = 2047 ∧ fraction ≠ 0 then some .nan
  else if word.msb && (exponent != 0 || fraction != 0) then none
  else if exponent = 2047 then some .infinity
  else some (.finite (if exponent = 0 then fraction else (2 ^ 52 + fraction) * 2 ^ (exponent - 1)))

set_option exponentiation.threshold 4096 in
theorem decoded_finite (word : BitVec 64) {units : Nat}
    (decoded : decode word = some (.finite units)) :
    Representable53 units ∧ units ≤ maxFinite := by
  simp only [decode] at decoded
  split at decoded
  · cases decoded
  split at decoded
  · cases decoded
  split at decoded
  · cases decoded
  rename_i finite
  let e := (word.extractLsb' 52 11).toNat
  let f := (word.extractLsb' 0 52).toNat
  have top : e ≤ 2046 := by
    change e ≠ 2047 at finite
    have width := (word.extractLsb' 52 11).isLt
    change e < 2048 at width
    omega
  have fraction : f < 2 ^ 52 := (word.extractLsb' 0 52).isLt
  have equal : (if e = 0 then f else (2 ^ 52 + f) * 2 ^ (e - 1)) = units :=
    Magnitude.finite.inj (Option.some.inj decoded)
  by_cases subnormal : e = 0
  · rw [ite_eq_left subnormal] at equal
    subst units
    have small : f.log2 ≤ 52 := by
      by_cases zero : f = 0
      · simp [zero]
      · exact Nat.le_of_lt ((Nat.log2_lt zero).mpr fraction)
    constructor
    · unfold Representable53
      rw [Nat.sub_eq_zero_of_le small, Nat.pow_zero]
      exact Nat.one_dvd _
    · have cap : 2 ^ 52 ≤ maxFinite := by
        calc
          2 ^ 52 ≤ 2 ^ 2045 := Nat.pow_le_pow_right (by decide) (by decide)
          _ = 1 * 2 ^ 2045 := (Nat.one_mul _).symm
          _ ≤ maxFinite := Nat.mul_le_mul_right _ (by decide : (1 : Nat) ≤ 2 ^ 53 - 1)
      exact Nat.le_trans (Nat.le_of_lt fraction) cap
  · rw [ite_eq_right subnormal] at equal
    subst units
    let q := 2 ^ 52 + f
    let shift := e - 1
    have scale : 0 < 2 ^ shift := Nat.two_pow_pos _
    have positive : q * 2 ^ shift ≠ 0 := Nat.mul_ne_zero (by dsimp [q]; omega) (Nat.ne_of_gt scale)
    have low : 2 ^ (52 + shift) ≤ q * 2 ^ shift := by
      rw [Nat.pow_add]
      exact Nat.mul_le_mul_right _ (by dsimp [q]; omega)
    have high : q * 2 ^ shift < 2 ^ ((52 + shift) + 1) := by
      rw [show 52 + shift + 1 = 53 + shift by omega, Nat.pow_add]
      exact Nat.mul_lt_mul_of_pos_right (by dsimp [q]; omega) scale
    have log := (Nat.log2_eq_iff positive).mpr ⟨low, high⟩
    refine ⟨?_, ?_⟩
    · change Representable53 (q * 2 ^ shift)
      unfold Representable53
      rw [log, show 52 + shift - 52 = shift by omega]
      exact ⟨q, Nat.mul_comm _ _⟩
    · have shiftLe : shift ≤ 2045 := by dsimp [shift]; omega
      exact Nat.mul_le_mul (by dsimp [q]; omega) (Nat.pow_le_pow_right (by decide) shiftLe)

def DecodeContract : Prop :=
  ∀ (word : BitVec 64) (units : Nat), decode word = some (.finite units) →
    Representable53 units ∧ units ≤ maxFinite

theorem decodeContract_proven : DecodeContract := fun word _ => decoded_finite word

def castUnits (elapsed : Nat) : Nat := round53 elapsed * quantum

def elapsedProduct (elapsed : Nat) : Magnitude → Magnitude
  | .finite units =>
    let product := round53 (round53 elapsed * units)
    if maxFinite < product then .infinity else .finite product
  | .infinity => if elapsed = 0 then .nan else .infinity
  | .nan => .nan

def underBudget (budget elapsed : Nat) (factor : Magnitude) : Bool :=
  match elapsedProduct elapsed factor with
  | .finite units => decide (units < castUnits budget)
  | .infinity | .nan => false

theorem underBudget_mono (budget : Nat) (factor : Magnitude) {earlier later : Nat}
    (time : earlier ≤ later) (continued : underBudget budget later factor = true) :
    underBudget budget earlier factor = true := by
  cases factor with
  | finite units =>
    have products : round53 (round53 earlier * units) ≤ round53 (round53 later * units) :=
      round53_mono (Nat.mul_le_mul_right _ (round53_mono time))
    have within : round53 (round53 later * units) ≤ maxFinite := by
      by_cases outside : maxFinite < round53 (round53 later * units)
      · simp [underBudget, elapsedProduct, outside] at continued
      · exact Nat.le_of_not_gt outside
    have earlierWithin := Nat.le_trans products within
    have strict : round53 (round53 later * units) < castUnits budget := by
      simpa [underBudget, elapsedProduct, Nat.not_lt_of_ge within] using continued
    simp [underBudget, elapsedProduct, Nat.not_lt_of_ge earlierWithin, Nat.lt_of_le_of_lt products strict]
  | infinity =>
    by_cases zero : later = 0 <;> simp [underBudget, elapsedProduct, zero] at continued
  | nan => cases continued

structure Prediction where
  budgetMs : Nat
  factor : Nat → Magnitude

def softAllows (prediction : Option Prediction) (level elapsed : Nat) : Bool :=
  match prediction with
  | none => true
  | some fixed => underBudget fixed.budgetMs elapsed (fixed.factor level)

def beforeDeadline (deadline : Option Nat) (elapsed : Nat) : Bool :=
  (deadline.map (fun limit => decide (elapsed < limit))).getD true

def check (deadline : Option Nat) (prediction : Option Prediction) (level hardSample softSample : Nat) :
    Bool :=
  beforeDeadline deadline hardSample &&
    softAllows prediction level softSample

def TimingContract : Prop :=
  (∀ prediction : Option Prediction, EndgameProgress.StableBudget (softAllows prediction)) ∧
  (∀ (deadline : Option Nat) (prediction : Option Prediction) (level hard₁ hard₂ soft₁ soft₂ : Nat),
    hard₁ ≤ hard₂ → soft₁ ≤ soft₂ →
    check deadline prediction level hard₂ soft₂ = true → check deadline prediction level hard₁ soft₁ = true)

theorem timingContract_proven : TimingContract := by
  have soft (prediction : Option Prediction) : EndgameProgress.StableBudget (softAllows prediction) := by
    intro level earlier later time continued
    cases prediction with
    | none => rfl
    | some fixed => exact underBudget_mono fixed.budgetMs (fixed.factor level) time continued
  refine ⟨soft, fun deadline prediction level hard₁ hard₂ soft₁ soft₂ orderedHard orderedSoft continued => ?_⟩
  unfold check at continued ⊢
  rw [Bool.and_eq_true] at continued ⊢
  obtain ⟨hard, permitted⟩ := continued
  have beforeHard : beforeDeadline deadline hard₁ = true := by
    cases deadline with
    | none => rfl
    | some limit =>
      have below : hard₂ < limit := of_decide_eq_true hard
      exact decide_eq_true (Nat.lt_of_le_of_lt orderedHard below)
  exact ⟨beforeHard, soft prediction level soft₁ soft₂ orderedSoft permitted⟩

open SearchPublication (Completed Row capture finish)
open EndgameProgress (AtLeast Clock ClockOrder depthAllowed lastLevel runNormal)

def sampledMs (nanoseconds : Nat) : Nat :=
  (BitVec.ofNat 64 (nanoseconds / 1000000)).toNat

def SampleContract : Prop :=
  ∀ (earlier later : Clock),
    ClockOrder earlier later → (∀ level, later level / 1000000 < 2 ^ 64) →
    ClockOrder (sampledMs ∘ earlier) (sampledMs ∘ later)

theorem sampleContract_proven : SampleContract := by
  intro earlier later ordered fits level
  have truncated := Nat.div_le_div_right (c := 1000000) (ordered level)
  have earlierFits := Nat.lt_of_le_of_lt truncated (fits level)
  simpa only [Function.comp_apply, sampledMs, BitVec.toNat_ofNat,
    Nat.mod_eq_of_lt earlierFits, Nat.mod_eq_of_lt (fits level)] using truncated

def NormalTimingContract : Prop :=
  ∀ (minimum depth : Nat) (timed : Bool) (depthFor : Nat → Nat) (prediction : Option Prediction)
    (previous : List Clock) (last : Clock) (rows : Nat → Nat → List Row)
    (finalRows : List Row) (inherited : Completed),
    (timed = true ∨ depth ≤ depthFor 1) →
    (∀ earlier ∈ previous, ClockOrder earlier last) →
    (∀ level, last level / 1000000 < 2 ^ 64) →
    let allowed := depthAllowed timed depthFor depth
    let times := previous.map (fun clock => sampledMs ∘ clock) ++ [sampledMs ∘ last]
    inherited.marks.length = times.length →
    let keep := softAllows prediction
    let state := runNormal minimum depth allowed keep rows times 0 (SearchPublication.beginPhase inherited)
    ∃ final, lastLevel allowed keep (sampledMs ∘ last) = some final ∧
      (minimum ≤ final → AtLeast depth final state.marks ∧
        finish minimum state (.normal ⟨depth, final⟩ finalRows) = some (capture finalRows ⟨depth, final⟩ true)) ∧
      (final < minimum → finish minimum state (.normal ⟨depth, final⟩ finalRows) = state.last)

theorem normalTimingContract_proven : NormalTimingContract := by
  intro minimum depth timed depthFor prediction previous last rows finalRows inherited entry ordered fits
  refine EndgameProgress.normalRunContract_proven minimum depth timed depthFor (softAllows prediction)
    (previous.map fun clock => sampledMs ∘ clock) (sampledMs ∘ last) rows finalRows inherited
    entry (timingContract_proven.1 prediction) ?_
  intro earlier observed
  obtain ⟨clock, present, rfl⟩ := List.mem_map.mp observed
  exact sampleContract_proven clock last (ordered clock present) fits

example : round53 (2 ^ 53 + 1) = 2 ^ 53 ∧ round53 (2 ^ 53 + 3) = 2 ^ 53 + 4 := by decide

set_option exponentiation.threshold 4096 in
set_option maxRecDepth 8192 in
example :
    let factor := Magnitude.finite (3 * 2 ^ 1073)
    underBudget 15 9 factor = true ∧ underBudget 15 10 factor = false ∧
    underBudget 15 0 .infinity = false := by decide +kernel

end NeuralReversi.EndgameTiming

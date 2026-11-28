module
public import NeuralReversi.Eval.SmallNetwork.Forward

@[expose] public section

namespace NeuralReversi.Eval.SmallNetwork

open InputLayer (sumRows)

private theorem sumRows_bounds (bias : Int) (value lower upper : Nat → Int) (count : Nat)
    (laneBounds : ∀ i, i < count → lower i ≤ value i ∧ value i ≤ upper i) :
    sumRows bias lower count ≤ sumRows bias value count ∧
    sumRows bias value count ≤ sumRows bias upper count := by
  induction count with
  | zero => exact ⟨Int.le_refl _, Int.le_refl _⟩
  | succ n ih =>
    have bounds := ih (fun i hi => laneBounds i (by omega))
    have last := laneBounds n (by omega)
    rw [sumRows, sumRows, sumRows]
    exact ⟨by omega, by omega⟩

def HiddenIntervalContract : Prop :=
  ∀ bias (selected lower upper : Nat → Int),
    (∀ i, i < 32 → lower i ≤ selected i ∧ selected i ≤ upper i) →
    let lo := sumRows bias lower 32
    let hi := sumRows bias upper 32
    let exactValue := sumRows bias selected 32
    lo ≤ exactValue ∧ exactValue ≤ hi ∧
    (-32768 ≤ lo → hi ≤ 32767 → referenceHidden bias selected = exactValue)

theorem hiddenIntervalContract_proven : HiddenIntervalContract := by
  intro bias selected lower upper valid
  have bounds := sumRows_bounds bias selected lower upper 32 valid
  refine ⟨bounds.1, bounds.2, ?_⟩
  intro h₀ h₁
  unfold referenceHidden
  apply BitVec.toInt_ofInt_eq_self (by decide)
  all_goals simp <;> omega

def weightedLower (lo hi weight : Int) : Int :=
  min (activation lo * weight) (activation hi * weight)

def weightedUpper (lo hi weight : Int) : Int :=
  max (activation lo * weight) (activation hi * weight)

private theorem weightedBounds {lo current hi : Int} (lower : lo ≤ current) (upper : current ≤ hi)
    (weight : Int) :
    weightedLower lo hi weight ≤ activation current * weight ∧
      activation current * weight ≤ weightedUpper lo hi weight := by
  have lo := activationMonotoneContract_proven _ _ lower
  have hi := activationMonotoneContract_proven _ _ upper
  unfold weightedLower weightedUpper
  by_cases sign : 0 ≤ weight
  · have a := Int.mul_le_mul_of_nonneg_right lo sign
    have b := Int.mul_le_mul_of_nonneg_right hi sign
    omega
  · have a := Int.mul_le_mul_of_nonpos_right lo (by omega : weight ≤ 0)
    have b := Int.mul_le_mul_of_nonpos_right hi (by omega : weight ≤ 0)
    omega

def OutputIntervalContract : Prop :=
  ∀ bias (hidden lower upper weights : Nat → Int),
    (∀ i, i < 128 → lower i ≤ hidden i ∧ hidden i ≤ upper i) →
    let lo := sumRows bias (fun i => weightedLower (lower i) (upper i) (weights i)) 128
    let hi := sumRows bias (fun i => weightedUpper (lower i) (upper i) (weights i)) 128
    let exactValue := sumRows bias (fun i => activation (hidden i) * weights i) 128
    lo ≤ exactValue ∧ exactValue ≤ hi ∧
    (-2147483648 ≤ lo → hi ≤ 2147483647 → (BitVec.ofInt 32 exactValue).toInt = exactValue)

theorem outputIntervalContract_proven : OutputIntervalContract := by
  intro bias hidden lower upper weights valid
  have bounds := sumRows_bounds bias (fun i => activation (hidden i) * weights i)
    (fun i => weightedLower (lower i) (upper i) (weights i))
    (fun i => weightedUpper (lower i) (upper i) (weights i)) 128
    (fun i h => weightedBounds (valid i h).1 (valid i h).2 (weights i))
  exact ⟨bounds.1, bounds.2, fun h₀ h₁ => BitVec.toInt_ofInt_eq_self (by decide)
    (by simp; omega) (by simp; omega)⟩

end NeuralReversi.Eval.SmallNetwork

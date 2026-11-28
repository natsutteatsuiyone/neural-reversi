module
public import NeuralReversi.MidgameAspiration.Termination

@[expose] public section

namespace NeuralReversi.MidgameAspiration

open ScoreRefresh

def InitialArithmeticContract : Prop :=
  ∀ (stats : History) (helper : Nat), stats.valid →
    FitsSigned 64 (stats.meanSquared.natAbs : Int) ∧
    FitsSigned 32 (bonus stats) ∧
    FitsSigned 32 (512 + 64 * (helper % 8 : Nat)) ∧
    FitsSigned 32 (initialWidth stats helper) ∧
    FitsSigned 32 (stats.average - initialWidth stats helper) ∧
    FitsSigned 32 (stats.average + initialWidth stats helper)

theorem initialArithmeticContract_proven : InitialArithmeticContract := by
  intro stats helper valid
  obtain ⟨lo, hi, mss⟩ := (History.bounds valid)
  have b := bonus_bounds valid
  have d := initialWidth_bounds valid helper
  have h := Nat.mod_lt helper (by decide : 0 < 8)
  unfold squareLimit at mss
  unfold FitsSigned
  omega

def growScore (checked : Bool) (delta : Score) : Except String Score := do
  let half ← divInt checked delta 2
  addScore checked delta half

theorem grow_arithmetic (checked : Bool) (d : Score) (positive : 0 < d.raw.toInt)
    (bound : d.raw.toInt ≤ 46542822) : observe (growScore checked d) = .ok (grow d.raw.toInt) := by
  let half : Score := ⟨d.raw.sdiv 2⟩
  have divide : divInt checked d 2 = .ok half := by simp [divInt, half]
  have value : half.raw.toInt = d.raw.toInt / 2 := by
    change (d.raw.sdiv 2#32).toInt = _
    rw [ScaledScore.sdiv_no_wrap _ _ (by simp)]
    change d.raw.toInt.tdiv 2 = d.raw.toInt / 2
    exact Int.tdiv_eq_ediv_of_nonneg (by omega)
  simp only [growScore, divide, bind, Except.bind]
  change observe (addWords checked d.raw half.raw) = _
  rw [addWords_spec, value, Aspiration.safe_math checked _ _ (by unfold ScaledScore.outsideI32; omega)]
  rfl

def RetryArithmeticContract : Prop :=
  ∀ (Payload : Type) (search : Aspiration.Trials Payload) (start reached : Attempt) (n : Nat),
    FiniteTrials search → start.window → 512 ≤ start.delta → start.delta ≤ 106288 →
    Reaches search 0 start n reached →
    n < 16 ∧ reached.window ∧ 512 ≤ reached.delta ∧ reached.delta ≤ 46542822 ∧
    ∀ (checked : Bool) (score alpha beta delta : Score),
      alpha.raw.toInt = reached.alpha → beta.raw.toInt = reached.beta → delta.raw.toInt = reached.delta →
      FiniteScore score.raw.toInt →
      (Aspiration.widenScore checked score alpha beta delta).map Aspiration.readWindow =
        .ok (Aspiration.widen score.raw.toInt reached.alpha reached.beta reached.delta) ∧
      observe (growScore checked delta) = .ok (grow reached.delta)

theorem retryArithmeticContract_proven : RetryArithmeticContract := by
  intro Payload search start reached n finite valid lo hi path
  obtain ⟨_, returned, _⟩ := terminationContract_proven Payload search start finite valid lo
  have before := reaches_before_return path returned
  obtain ⟨window, width⟩ := reaches_window path finite valid (by omega)
  have lower : 512 ≤ reached.delta := by
    rw [width]
    exact Int.le_trans lo (growN_mono_steps (m := 0) (by omega) (Nat.zero_le n))
  have upper : reached.delta ≤ 46542822 := by
    rw [width]
    exact Int.le_trans (growN_mono_steps (n := 15) (by omega) (by omega))
      (Int.le_trans (growN_mono 15 hi) (by decide))
  refine ⟨before, window, lower, upper, ?_⟩
  intro checked score alpha beta delta a b d finiteScore
  obtain ⟨wl, ab, wh⟩ := window
  obtain ⟨sl, sh⟩ := finiteScore
  constructor
  · rw [← a, ← b, ← d]
    exact Aspiration.widen_arithmetic checked score alpha beta delta
      (by omega) (by omega) (by omega) (by omega) (by omega) (by omega)
  · rw [← d]
    exact grow_arithmetic checked delta (by omega) (by omega)

def SearchContract : Prop :=
  ∀ (Payload : Type) (search : Aspiration.Trials Payload) (depth helper : Nat)
    (scores : Option (List Int)),
    (∀ history ∈ scores, ∀ score ∈ history, FiniteScore score) → FiniteTrials search →
    let start := initial depth (scores.map (·.foldl History.update History.empty)) helper
    ∃ result, run search 16 0 start = some result ∧ Witness search 0 16 start result ∧
      ∀ n attempt, Reaches search 0 start n attempt →
        n < 16 ∧ attempt.window ∧ 512 ≤ attempt.delta ∧ attempt.delta ≤ 46542822

theorem searchContract_proven : SearchContract := by
  intro Payload search depth helper scores finiteHistory finiteTrials
  let stats := scores.map (·.foldl History.update History.empty)
  have known : ∀ h ∈ stats, h.valid := by
    intro h member
    cases scores with
    | none => simp [stats] at member
    | some samples =>
      simp [stats] at member
      subst h
      exact (historyContract_proven samples (finiteHistory samples (by simp))).1
  let start := initial depth stats helper
  have ⟨valid, low, high⟩ := initialContract_proven depth helper stats known
  obtain ⟨result, returned, witness⟩ := terminationContract_proven Payload search start finiteTrials valid low
  refine ⟨result, returned, witness, fun n attempt path => ?_⟩
  obtain ⟨before, window, lo, hi, _⟩ :=
    retryArithmeticContract_proven Payload search start attempt n finiteTrials valid low high path
  exact ⟨before, window, lo, hi⟩

end NeuralReversi.MidgameAspiration

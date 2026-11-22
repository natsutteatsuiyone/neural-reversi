import NeuralReversi.MidgameAspiration.History

namespace NeuralReversi.MidgameAspiration

structure Attempt where
  alpha : Int
  beta : Int
  delta : Int
  deriving DecidableEq, Repr

def Attempt.window (attempt : Attempt) : Prop := Aspiration.Window attempt.alpha attempt.beta

def bonus (stats : History) : Int := (stats.meanSquared.natAbs / 10193 : Nat)

def initialWidth (stats : History) (helper : Nat) : Int := 512 + 64 * (helper % 8 : Nat) + bonus stats

theorem bonus_bounds {stats : History} (valid : stats.valid) : 0 ≤ bonus stats ∧ bonus stats ≤ 105328 := by
  have quotient : stats.meanSquared.natAbs / 10193 ≤ 105328 :=
    Nat.div_le_div_right (c := 10193) (History.bounds valid).2.2
  unfold bonus
  omega

theorem initialWidth_bounds {stats : History} (valid : stats.valid) (helper : Nat) :
    512 ≤ initialWidth stats helper ∧ initialWidth stats helper ≤ 106288 := by
  have := Nat.mod_lt helper (by decide : 0 < 8)
  have := bonus_bounds valid
  unfold initialWidth
  omega

def initial (depth : Nat) (history : Option History) (helper : Nat) : Attempt :=
  if depth < 5 then ⟨-32767, 32767, 512⟩
  else
    match history with
    | none => ⟨-32767, 32767, 512⟩
    | some stats =>
      let delta := initialWidth stats helper
      ⟨max (stats.average - delta) (-32767), min (stats.average + delta) 32767, delta⟩

def InitialContract : Prop :=
  ∀ (depth helper : Nat) (history : Option History), (∀ stats ∈ history, stats.valid) →
    let attempt := initial depth history helper
    attempt.window ∧ 512 ≤ attempt.delta ∧ attempt.delta ≤ 106288

theorem initialContract_proven : InitialContract := by
  intro depth helper history valid
  have full : (⟨-32767, 32767, 512⟩ : Attempt).window ∧
      (512 : Int) ≤ 512 ∧ (512 : Int) ≤ 106288 := ⟨⟨by decide, by decide, by decide⟩, by decide, by decide⟩
  change (initial depth history helper).window ∧
    512 ≤ (initial depth history helper).delta ∧ (initial depth history helper).delta ≤ 106288
  unfold initial
  split
  · exact full
  · cases history with
    | none => exact full
    | some stats =>
      let width := initialWidth stats helper
      change Aspiration.Window (max (stats.average - width) (-32767))
        (min (stats.average + width) 32767) ∧ 512 ≤ width ∧ width ≤ 106288
      have statsValid := valid stats (by simp)
      obtain ⟨lo, hi, _⟩ := (History.bounds statsValid)
      have wide : 512 ≤ width ∧ width ≤ 106288 := initialWidth_bounds statsValid helper
      exact ⟨⟨by omega, by omega, by omega⟩, wide⟩

def grow (delta : Int) : Int := delta + delta / 2

def growN : Nat → Int → Int
  | 0, delta => delta
  | n + 1, delta => grow (growN n delta)

theorem growN_step (n : Nat) (delta : Int) : growN n (grow delta) = growN (n + 1) delta := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [growN, ih]

theorem grow_mono {a b : Int} (bound : a ≤ b) : grow a ≤ grow b := by
  unfold grow
  omega

theorem growN_mono (n : Nat) {a b : Int} (bound : a ≤ b) : growN n a ≤ growN n b := by
  induction n with
  | zero => exact bound
  | succ n ih => exact grow_mono ih

private theorem growN_nonneg (n : Nat) {d : Int} (nonneg : 0 ≤ d) : 0 ≤ growN n d := by
  induction n with
  | zero => exact nonneg
  | succ n ih => change 0 ≤ grow (growN n d); unfold grow; omega

theorem growN_mono_steps {m n : Nat} {d : Int} (nonneg : 0 ≤ d) (bound : m ≤ n) :
    growN m d ≤ growN n d := by
  induction bound with
  | refl => exact Int.le_refl _
  | @step n _ ih =>
    apply Int.le_trans ih
    change growN _ _ ≤ grow (growN _ _)
    have := growN_nonneg n nonneg
    unfold grow
    omega

def Attempt.retry (attempt : Attempt) (score : Int) : Option Attempt :=
  (Aspiration.widen score attempt.alpha attempt.beta attempt.delta).map fun (a, b) =>
    ⟨a, b, grow attempt.delta⟩

theorem retry_window {attempt next : Attempt} {score : Int}
    (window : attempt.window) (finite : FiniteScore score) (pos : 0 < attempt.delta)
    (step : attempt.retry score = some next) :
    next.window ∧ next.delta = grow attempt.delta := by
  unfold Attempt.retry at step
  cases widened : Aspiration.widen score attempt.alpha attempt.beta attempt.delta with
  | none => simp [widened] at step
  | some bounds =>
    obtain ⟨alpha, beta⟩ := bounds
    rw [widened] at step
    cases Option.some.inj step
    exact ⟨Aspiration.nextWindow window finite.1 finite.2 pos widened, rfl⟩

structure Result (Payload : Type) where
  index : Nat
  attempt : Attempt
  trial : Aspiration.Trial Payload

def Result.clean (result : Result Payload) : Bool := result.index == 0

def run (search : Aspiration.Trials Payload) : Nat → Nat → Attempt → Option (Result Payload)
  | 0, _, _ => none
  | fuel + 1, i, attempt =>
    let trial := search i attempt.alpha attempt.beta
    if trial.aborted then some ⟨i, attempt, trial⟩
    else
      match attempt.retry trial.score with
      | none => some ⟨i, attempt, trial⟩
      | some next => run search fuel (i + 1) next

def FiniteTrials (search : Aspiration.Trials Payload) : Prop :=
  ∀ i a b, Aspiration.Window a b → (search i a b).aborted = false → FiniteScore (search i a b).score

theorem retry_none {attempt : Attempt} {score : Int} (stop : attempt.retry score = none) :
    attempt.alpha < score ∧ score < attempt.beta := by
  apply Aspiration.widen_none (delta := attempt.delta)
  simpa [Attempt.retry] using stop

theorem run_aborted (search : Aspiration.Trials Payload) (attempt : Attempt) (i fuel : Nat)
    (abort : (search i attempt.alpha attempt.beta).aborted = true) :
    run search (fuel + 1) i attempt = some ⟨i, attempt, search i attempt.alpha attempt.beta⟩ := by
  simp [run, abort]

structure Witness (search : Aspiration.Trials Payload) (i fuel : Nat) (start : Attempt)
    (result : Result Payload) : Prop where
  lower : i ≤ result.index
  upper : result.index < i + fuel
  window : result.attempt.window
  delta : result.attempt.delta = growN (result.index - i) start.delta
  source : result.trial = search result.index result.attempt.alpha result.attempt.beta
  inside : result.trial.aborted = false →
    result.attempt.alpha < result.trial.score ∧ result.trial.score < result.attempt.beta

theorem run_witness (search : Aspiration.Trials Payload) (finite : FiniteTrials search)
    (window : start.window) (pos : 0 < start.delta) (found : run search fuel i start = some result) :
    Witness search i fuel start result := by
  induction fuel generalizing i start with
  | zero => cases found
  | succ fuel ih =>
    rw [run] at found
    cases aborted : (search i start.alpha start.beta).aborted
    · simp only [aborted, Bool.false_eq_true, ite_false] at found
      cases step : start.retry (search i start.alpha start.beta).score with
      | none =>
        rw [step] at found
        cases Option.some.inj found
        exact ⟨Nat.le_refl _, by change i < i + (fuel + 1); omega, window,
          by simp [growN], rfl, fun _ => retry_none step⟩
      | some next =>
        rw [step] at found
        obtain ⟨nextWindow, nextWidth⟩ := retry_window window (finite _ _ _ window aborted) pos step
        have positive : 0 < next.delta := by rw [nextWidth]; unfold grow; omega
        obtain ⟨lo, hi, valid, width, source, inside⟩ := ih nextWindow positive found
        refine ⟨by omega, by omega, valid, ?_, source, inside⟩
        rw [width, nextWidth, growN_step, show result.index - (i + 1) + 1 = result.index - i by omega]
    · simp only [aborted, ite_true] at found
      cases Option.some.inj found
      refine ⟨Nat.le_refl _, by change i < i + (fuel + 1); omega, window,
        by simp [growN], rfl, ?_⟩
      simp [aborted]

example : initial 5 (some ⟨-31, -10193⟩) 7 = ⟨-992, 930, 961⟩ := by decide

example : growN 12 512 = 66429 := by decide

example :
    (run (fun i _ _ => ⟨if i = 0 then 32767 else 0, (), i == 0⟩) 16 0 ⟨-1, 1, 512⟩).map
      (fun result => (result.attempt, result.clean)) = some (⟨-1, 1, 512⟩, true) := by decide

end NeuralReversi.MidgameAspiration

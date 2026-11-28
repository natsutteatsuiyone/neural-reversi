module
public import NeuralReversi.ScoreRefresh.Properties

@[expose] public section

namespace NeuralReversi.Aspiration

def widen (score alpha beta delta : Int) : Option (Int × Int) :=
  if score ≤ alpha then some (max (score - delta) (-32767), alpha)
  else if beta ≤ score then some (max (beta - delta) alpha, min (score + delta) 32767)
  else none

abbrev Search := Nat → Int → Int → Int

structure Window (alpha beta : Int) : Prop where
  lower : -32767 ≤ alpha
  ordered : alpha < beta
  upper : beta ≤ 32767

structure SoundResult (value alpha beta score : Int) : Prop where
  lower : -32767 < score
  upper : score < 32767
  failLow : score ≤ alpha → value ≤ score
  failHigh : beta ≤ score → score ≤ value
  exact : alpha < score → score < beta → score = value

def Sound (search : Search) (value : Int) : Prop :=
  ∀ i alpha beta, Window alpha beta → SoundResult value alpha beta (search i alpha beta)

structure Trial (Payload : Type) where
  score : Int
  payload : Payload
  aborted : Bool

abbrev Trials (Payload : Type) := Nat → Int → Int → Trial Payload

structure Accepted (Payload : Type) where
  index : Nat
  alpha : Int
  beta : Int
  trial : Trial Payload

def runWith (search : Trials Payload) : Nat → Nat → Int → Int → Int → Option (Accepted Payload)
  | 0, _, _, _, _ => none
  | fuel + 1, i, alpha, beta, delta =>
    let trial := search i alpha beta
    if trial.aborted then none
    else
      match widen trial.score alpha beta delta with
      | none => some ⟨i, alpha, beta, trial⟩
      | some (a, b) => runWith search fuel (i + 1) a b (delta + delta)

def run (search : Search) (fuel i : Nat) (alpha beta delta : Int) : Option Int :=
  (runWith (fun i a b => ⟨search i a b, (), false⟩) fuel i alpha beta delta).map (·.trial.score)

private theorem run_step (search : Search) (fuel i : Nat) (alpha beta delta : Int) :
    run search (fuel + 1) i alpha beta delta =
      match widen (search i alpha beta) alpha beta delta with
      | none => some (search i alpha beta)
      | some (a, b) => run search fuel (i + 1) a b (delta + delta) := by
  simp only [run, runWith, Bool.false_eq_true, ite_false]
  split <;> rfl

theorem nextWindow {alpha beta score delta a b : Int}
    (w : Window alpha beta) (lo : -32767 < score) (hi : score < 32767)
    (pos : 0 < delta) (step : widen score alpha beta delta = some (a, b)) : Window a b := by
  obtain ⟨_, _, _⟩ := w
  unfold widen at step
  split at step
  · cases step
    constructor <;> omega
  · split at step
    · cases step
      constructor <;> omega
    · contradiction

theorem widen_none {alpha beta score delta : Int} (finished : widen score alpha beta delta = none) :
    alpha < score ∧ score < beta := by
  unfold widen at finished
  split at finished
  · contradiction
  · split at finished
    · contradiction
    · omega

def scores (search : Trials Payload) : Search := fun i a b => (search i a b).score

def TrialSound (search : Trials Payload) (value : Int) : Prop :=
  ∀ i a b, Window a b → (search i a b).aborted = false →
    SoundResult value a b (search i a b).score

structure Witness (search : Trials Payload) (i fuel : Nat) (result : Accepted Payload) : Prop where
  lower : i ≤ result.index
  upper : result.index < i + fuel
  window : Window result.alpha result.beta
  source : result.trial = search result.index result.alpha result.beta
  completed : result.trial.aborted = false
  inside : result.alpha < result.trial.score ∧ result.trial.score < result.beta

private theorem widen_bounded {alpha beta score delta a b : Int}
    (lower : -32767 ≤ alpha) (alphaUpper : alpha ≤ 32767) (upper : beta ≤ 32767)
    (positive : 0 < delta) (step : widen score alpha beta delta = some (a, b)) :
    -32767 ≤ a ∧ a ≤ 32767 ∧ b ≤ 32767 := by
  unfold widen at step
  split at step
  · cases step
    constructor
    · exact Int.le_max_right _ _
    · constructor <;> omega
  · split at step
    · cases step
      refine ⟨Int.le_trans lower (Int.le_max_right _ _), ?_, Int.min_le_right _ _⟩
      omega
    · contradiction

theorem runWith_witness (search : Trials Payload)
    (window : Window alpha beta) (positive : 0 < delta)
    (returns : runWith search fuel i alpha beta delta = some result) :
    Witness search i fuel result := by
  have lower := window.lower
  have upper := window.upper
  have alphaUpper : alpha ≤ 32767 := by have := window.ordered; omega
  clear window
  induction fuel generalizing i alpha beta delta with
  | zero => simp [runWith] at returns
  | succ fuel ih =>
    rw [runWith] at returns
    cases stopped : (search i alpha beta).aborted
    · simp only [stopped, Bool.false_eq_true, ite_false] at returns
      cases step : widen (search i alpha beta).score alpha beta delta with
      | none =>
        rw [step] at returns
        cases Option.some.inj returns
        have inside := widen_none step
        exact ⟨Nat.le_refl _, by change i < i + (fuel + 1); omega,
          ⟨lower, by change alpha < beta; omega, upper⟩, rfl, stopped, inside⟩
      | some next =>
        obtain ⟨a, b⟩ := next
        rw [step] at returns
        obtain ⟨aLower, aUpper, bUpper⟩ := widen_bounded lower alphaUpper upper positive step
        obtain ⟨first, last, valid, same, completed, inside⟩ :=
          ih (by omega) returns aLower bUpper aUpper
        exact ⟨by omega, by omega, valid, same, completed, inside⟩
    · simp [stopped] at returns

theorem runWith_aborted (search : Trials Payload) (stopped : (search i alpha beta).aborted = true) :
    runWith search (fuel + 1) i alpha beta delta = none := by
  simp [runWith, stopped]

theorem runWith_scores (search : Trials Payload) (sound : TrialSound search value)
    (uninterrupted : ∀ i a b, Window a b → (search i a b).aborted = false)
    (window : Window alpha beta) (positive : 0 < delta) :
    (runWith search fuel i alpha beta delta).map (·.trial.score) = run (scores search) fuel i alpha beta delta := by
  induction fuel generalizing i alpha beta delta with
  | zero => rfl
  | succ fuel ih =>
    have stopped := uninterrupted i alpha beta window
    rw [runWith, run_step]
    simp only [stopped, Bool.false_eq_true, ite_false, scores]
    cases step : widen (search i alpha beta).score alpha beta delta with
    | none => rfl
    | some next =>
      obtain ⟨a, b⟩ := next
      have trial := sound i alpha beta window stopped
      exact ih (nextWindow window trial.lower trial.upper positive step) (by omega)

private theorem inside {search : Search} {value alpha beta delta : Int} {i fuel : Nat}
    (sound : Sound search value) (w : Window alpha beta)
    (lo : alpha < value) (hi : value < beta) :
    run search (fuel + 1) i alpha beta delta = some value := by
  have result := sound i alpha beta w
  have low : ¬ search i alpha beta ≤ alpha := fun h => (by have := result.failLow h; omega)
  have high : ¬ beta ≤ search i alpha beta := fun h => (by have := result.failHigh h; omega)
  rw [run_step, widen, ite_eq_right low, ite_eq_right high]
  exact congrArg some (result.exact (by omega) (by omega))

private theorem largeTwo {search : Search} {value alpha beta delta : Int} {i fuel : Nat}
    (sound : Sound search value) (w : Window alpha beta) (hi : value < 32767)
    (above : alpha < value) (large : 65536 ≤ delta) :
    run search (fuel + 2) i alpha beta delta = some value := by
  have result := sound i alpha beta w
  have low : ¬ search i alpha beta ≤ alpha := fun h => (by have := result.failLow h; omega)
  rw [run_step, widen, ite_eq_right low]
  by_cases high : beta ≤ search i alpha beta
  · have bound := result.failHigh high
    have scoreLower := result.lower
    obtain ⟨lower, ordered, upper⟩ := w
    rw [ite_eq_left high, show max (beta - delta) alpha = alpha by omega,
      show min (search i alpha beta + delta) 32767 = 32767 by omega]
    exact inside sound ⟨lower, by omega, Int.le_refl _⟩ above hi
  · rw [ite_eq_right high]
    exact congrArg some (result.exact (by omega) (by omega))

private theorem largeThree {search : Search} {value alpha beta delta : Int} {i : Nat}
    (sound : Sound search value) (lo : -32767 < value) (hi : value < 32767)
    (w : Window alpha beta) (large : 65536 ≤ delta) :
    run search 3 i alpha beta delta = some value := by
  by_cases above : alpha < value
  · exact largeTwo (fuel := 1) sound w hi above large
  have result := sound i alpha beta w
  obtain ⟨lower, ordered, upper⟩ := w
  have low : search i alpha beta ≤ alpha := by
    apply Int.le_of_not_gt
    intro aboveAlpha
    by_cases high : beta ≤ search i alpha beta
    · have := result.failHigh high
      omega
    · have := result.exact (by omega) (by omega)
      omega
  have scoreUpper := result.upper
  have scoreLower := result.lower
  rw [run_step, widen, ite_eq_left low, show max (search i alpha beta - delta) (-32767) = -32767 by omega]
  exact largeTwo (fuel := 0) sound ⟨Int.le_refl _, by omega, by omega⟩ hi lo (by omega)

private def doubled : Nat → Int → Int
  | 0, delta => delta
  | n + 1, delta => doubled n (delta + delta)

private theorem warmup {search : Search} {value : Int} (sound : Sound search value)
    (lo : -32767 < value) (hi : value < 32767) (rounds : Nat) {i : Nat}
    {alpha beta delta : Int} (w : Window alpha beta) (pos : 0 < delta)
    (large : 65536 ≤ doubled rounds delta) :
    run search (rounds + 3) i alpha beta delta = some value := by
  induction rounds generalizing i alpha beta delta with
  | zero => exact largeThree sound lo hi w large
  | succ n ih =>
    have result := sound i alpha beta w
    rw [Nat.succ_add, run_step]
    cases step : widen (search i alpha beta) alpha beta delta with
    | some ab =>
      obtain ⟨a, b⟩ := ab
      exact ih (nextWindow w result.lower result.upper pos step) (by omega) large
    | none =>
      unfold widen at step
      split at step
      · contradiction
      · split at step
        · contradiction
        · exact congrArg some (result.exact (by omega) (by omega))

def EndgameContract : Prop :=
  ∀ (search : Search) (value alpha beta : Int), Sound search value →
    -32767 < value → value < 32767 → Window alpha beta →
    run search 11 0 alpha beta 256 = some value

theorem endgameContract_proven : EndgameContract := by
  intro search value alpha beta sound lo hi w
  exact warmup sound lo hi 8 w (by decide) (by decide)

def AcceptedContract : Prop :=
  ∀ (Payload : Type) (search : Trials Payload) (value alpha beta : Int),
    TrialSound search value → -32767 < value → value < 32767 → Window alpha beta →
    (∀ i a b, Window a b → (search i a b).aborted = false) →
    ∃ result, runWith search 11 0 alpha beta 256 = some result ∧
      Witness search 0 11 result ∧ result.trial.score = value

theorem acceptedContract_proven : AcceptedContract := by
  intro Payload search value alpha beta sound lo hi window uninterrupted
  have numeric : Sound (scores search) value :=
    fun i a b valid => sound i a b valid (uninterrupted i a b valid)
  have projected := (runWith_scores (fuel := 11) (i := 0) search sound uninterrupted window (by decide : (0 : Int) < 256)).trans
    (endgameContract_proven (scores search) value alpha beta numeric lo hi window)
  cases found : runWith search 11 0 alpha beta 256 with
  | none => rw [found] at projected; cases projected
  | some result =>
    rw [found] at projected
    exact ⟨result, rfl, runWith_witness search window (by decide) found, Option.some.inj projected⟩

example :
    runWith (fun i _ _ => ⟨if i = 0 then 4 else 0, (), i == 1⟩) 11 0 (-1) 1 256 = none := rfl

open ScoreRefresh

def higher (a b : Score) : Score := if a.raw.toInt ≤ b.raw.toInt then b else a
def lower (a b : Score) : Score := if a.raw.toInt ≤ b.raw.toInt then a else b

def widenScore (checked : Bool) (score alpha beta delta : Score) :
    Except String (Option (Score × Score)) := do
  if score.raw.toInt ≤ alpha.raw.toInt then
    let low ← subScore checked score delta
    let floor ← negate checked scoreInf
    return some (higher low floor, alpha)
  else if beta.raw.toInt ≤ score.raw.toInt then
    let low ← subScore checked beta delta
    let high ← addScore checked score delta
    return some (higher low alpha, lower high scoreInf)
  else
    return none

def readWindow (window : Option (Score × Score)) : Option (Int × Int) :=
  window.map fun (alpha, beta) => (alpha.raw.toInt, beta.raw.toInt)

private theorem higher_value (a b : Score) :
    (higher a b).raw.toInt = max a.raw.toInt b.raw.toInt := by
  unfold higher
  split <;> rename_i h
  · exact (Int.max_eq_right h).symm
  · exact (Int.max_eq_left (by omega)).symm

private theorem lower_value (a b : Score) :
    (lower a b).raw.toInt = min a.raw.toInt b.raw.toInt := by
  unfold lower
  split <;> rename_i h
  · exact (Int.min_eq_left h).symm
  · exact (Int.min_eq_right (by omega)).symm

private theorem successful_score {result : Except String Score} {n : Int}
    (h : observe result = .ok n) : ∃ s, result = .ok s ∧ s.raw.toInt = n := by
  cases result with
  | error e => cases h
  | ok s => exact ⟨s, rfl, Except.ok.inj h⟩

theorem safe_math (checked : Bool) (n : Int) (err : String)
    (h : ¬ ScaledScore.outsideI32 n) :
    ScaledScore.checkedMathSpec checked n err = .ok n := by
  cases checked
  · simp only [ScaledScore.checkedMathSpec, Bool.false_and, Bool.false_eq_true, ↓reduceIte]
    unfold ScaledScore.outsideI32 at h
    exact congrArg Except.ok (Int.bmod_eq_of_le (by omega) (by omega))
  · exact ScaledScore.checkedMathSpec_safe n err h

theorem widen_arithmetic (checked : Bool) (score alpha beta delta : Score)
    (hslo : -32767 ≤ score.raw.toInt) (hshi : score.raw.toInt ≤ 32767)
    (hblo : -32767 ≤ beta.raw.toInt) (hbhi : beta.raw.toInt ≤ 32767)
    (hdlo : 0 < delta.raw.toInt) (hdhi : delta.raw.toInt ≤ 2147450880) :
    (widenScore checked score alpha beta delta).map readWindow =
      .ok (widen score.raw.toInt alpha.raw.toInt beta.raw.toInt delta.raw.toInt) := by
  by_cases low : score.raw.toInt ≤ alpha.raw.toInt
  · obtain ⟨s, sub, val⟩ := successful_score ((subWords_spec checked score.raw delta.raw).trans
      (safe_math checked _ _ (by unfold ScaledScore.outsideI32; omega)))
    have floor : negate checked scoreInf = .ok ⟨-32767⟩ := by cases checked <;> rfl
    change subScore checked score delta = _ at sub
    simp [widenScore, widen, low, sub, floor, readWindow, higher_value, val, Except.map,
      bind, Except.bind, pure, Except.pure]
  · by_cases high : beta.raw.toInt ≤ score.raw.toInt
    · obtain ⟨s, sub, sv⟩ := successful_score ((subWords_spec checked beta.raw delta.raw).trans
        (safe_math checked _ _ (by unfold ScaledScore.outsideI32; omega)))
      obtain ⟨t, add, tv⟩ := successful_score ((addWords_spec checked score.raw delta.raw).trans
        (safe_math checked _ _ (by unfold ScaledScore.outsideI32; omega)))
      change subScore checked beta delta = _ at sub
      change addScore checked score delta = _ at add
      simp [widenScore, widen, low, high, sub, add, readWindow, higher_value, lower_value, sv, tv,
        scoreInf, Except.map, bind, Except.bind, pure, Except.pure]
    · simp [widenScore, widen, low, high, readWindow, Except.map, pure, Except.pure]

def ArithmeticContract : Prop := ∀ (checked : Bool) (score alpha beta delta : Score),
  -32767 ≤ score.raw.toInt → score.raw.toInt ≤ 32767 →
  -32767 ≤ alpha.raw.toInt → alpha.raw.toInt ≤ 32767 →
  -32767 ≤ beta.raw.toInt → beta.raw.toInt ≤ 32767 →
  0 < delta.raw.toInt → delta.raw.toInt ≤ 262144 →
  (widenScore checked score alpha beta delta).map readWindow =
    .ok (widen score.raw.toInt alpha.raw.toInt beta.raw.toInt delta.raw.toInt) ∧
  observe (addScore checked delta delta) = .ok (2 * delta.raw.toInt)

theorem arithmeticContract_proven : ArithmeticContract := by
  intro checked score alpha beta delta hslo hshi _ _ hblo hbhi hdlo hdhi
  constructor
  · exact widen_arithmetic checked score alpha beta delta hslo hshi hblo hbhi hdlo (by omega)
  · change observe (addWords checked delta.raw delta.raw) = _
    rw [addWords_spec, safe_math checked _ _ (by unfold ScaledScore.outsideI32; omega)]
    congr 1 <;> omega

def SearchContract : Prop := EndgameContract ∧ ArithmeticContract

theorem searchContract_proven : SearchContract :=
  ⟨endgameContract_proven, arithmeticContract_proven⟩

end NeuralReversi.Aspiration

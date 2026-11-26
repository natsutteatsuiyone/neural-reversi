import NeuralReversi.MidgameAspiration.Model

namespace NeuralReversi.MidgameAspiration

private def rank (alpha beta : Int) : Nat :=
  if alpha = -32767 then (if beta = 32767 then 0 else 1)
  else if beta = 32767 then 2 else 3

private theorem rank_bound (alpha beta : Int) : rank alpha beta ≤ 3 := by
  unfold rank
  split <;> split <;> decide

private theorem large_retry_decreases {start next : Attempt} {score : Int}
    (window : start.window) (large : 65536 ≤ start.delta) (finite : FiniteScore score)
    (step : start.retry score = some next) :
    rank next.alpha next.beta < rank start.alpha start.beta := by
  obtain ⟨alo, ordered, bhi⟩ := window
  obtain ⟨lo, hi⟩ := finite
  unfold Attempt.retry Aspiration.widen at step
  split at step
  · rename_i low
    have floor : max (score - start.delta) (-32767) = -32767 := by omega
    rw [floor] at step
    cases Option.some.inj step
    have oldFloor : start.alpha ≠ -32767 := by omega
    have oldCeiling : start.alpha ≠ 32767 := by omega
    simp only [rank, oldFloor, oldCeiling, ↓reduceIte]
    split <;> decide
  · split at step
    · rename_i high
      have same : max (start.beta - start.delta) start.alpha = start.alpha := by omega
      have ceiling : min (score + start.delta) 32767 = 32767 := by omega
      rw [same, ceiling] at step
      cases Option.some.inj step
      have oldCeiling : start.beta ≠ 32767 := by omega
      simp only [rank, oldCeiling, ↓reduceIte]
      split <;> decide
    · cases step

/-- The retry budget applies even when each trial mutates the table and the root rows. -/
def HasFuel : Nat → Attempt → Prop
  | 0, _ => False
  | n + 1, start => ∀ score, FiniteScore score → ∀ next, start.retry score = some next → HasFuel n next

private theorem finish_large (window : start.window) (large : 65536 ≤ start.delta)
    (bound : rank start.alpha start.beta < fuel) : HasFuel fuel start := by
  induction fuel generalizing start with
  | zero => omega
  | succ fuel ih =>
    intro score finite next step
    have smaller := large_retry_decreases window large finite step
    obtain ⟨valid,width⟩ := retry_window window finite (by omega) step
    exact ih valid (by rw [width]; unfold grow; omega) (by omega)

private theorem warmup (window : start.window) (positive : 0 < start.delta)
    (large : 65536 ≤ growN rounds start.delta) : HasFuel (rounds + 4) start := by
  induction rounds generalizing start with
  | zero => exact finish_large window large (Nat.lt_succ_of_le (rank_bound ..))
  | succ rounds ih =>
    rw [Nat.succ_add]
    intro score finite next step
    obtain ⟨valid,width⟩ := retry_window window finite positive step
    exact ih valid (by rw [width]; unfold grow; omega) (by rwa [width,growN_step])

theorem enough_retries (valid : start.window) (width : 512 ≤ start.delta) : HasFuel 16 start :=
  warmup (rounds := 12) valid (by omega)
    (Int.le_trans (by decide : (65536 : Int) ≤ growN 12 512) (growN_mono 12 width))

private theorem run_with_fuel (search : Aspiration.Trials Payload) (finite : FiniteTrials search)
    (valid : start.window) (positive : 0 < start.delta) (budget : HasFuel fuel start) :
    ∃ result, run search fuel i start = some result := by
  induction fuel generalizing i start with
  | zero => contradiction
  | succ fuel ih =>
    rw [run]
    split
    · exact ⟨_,rfl⟩
    next live =>
      cases step : start.retry (search i start.alpha start.beta).score with
      | none => exact ⟨_,rfl⟩
      | some next =>
        have word := finite _ _ _ valid (by simpa using live)
        obtain ⟨win,width⟩ := retry_window valid word positive step
        exact ih win (by rw [width]; unfold grow; omega) (budget _ word next step)

def TerminationContract : Prop :=
  ∀ (Payload : Type) (search : Aspiration.Trials Payload) (start : Attempt),
    FiniteTrials search → start.window → 512 ≤ start.delta →
    ∃ result, run search 16 0 start = some result ∧ Witness search 0 16 start result

theorem terminationContract_proven : TerminationContract := by
  intro Payload search start finite valid delta
  obtain ⟨result, returned⟩ := run_with_fuel (i := 0) search finite valid (by omega) (enough_retries valid delta)
  exact ⟨result, returned, run_witness search finite valid (by omega) returned⟩

def AcceptanceContract : Prop :=
  ∀ (Payload : Type) (search : Aspiration.Trials Payload) (start : Attempt),
    FiniteTrials search → start.window → 512 ≤ start.delta →
    (∀ i a b, Aspiration.Window a b → (search i a b).aborted = false) →
    ∃ result, run search 16 0 start = some result ∧
      result.attempt.alpha < result.trial.score ∧ result.trial.score < result.attempt.beta ∧
      result.trial = search result.index result.attempt.alpha result.attempt.beta ∧
      (result.clean = true ↔ result.index = 0)

theorem acceptanceContract_proven : AcceptanceContract := by
  intro Payload search start finite valid width uninterrupted
  obtain ⟨result, returned, witness⟩ := terminationContract_proven Payload search start finite valid width
  have completed : result.trial.aborted = false := by
    rw [witness.source]
    exact uninterrupted _ _ _ witness.window
  obtain ⟨lo, hi⟩ := witness.inside completed
  exact ⟨result, returned, lo, hi, witness.source, by simp [Result.clean]⟩

inductive Reaches (search : Aspiration.Trials Payload) : Nat → Attempt → Nat → Attempt → Prop
  | here (i : Nat) (start : Attempt) : Reaches search i start 0 start
  | later {i start rounds next reached}
      (running : (search i start.alpha start.beta).aborted = false)
      (retry : start.retry (search i start.alpha start.beta).score = some next)
      (rest : Reaches search (i + 1) next rounds reached) :
      Reaches search i start (rounds + 1) reached

theorem reaches_window {search : Aspiration.Trials Payload}
    (path : Reaches search i start rounds reached) (finite : FiniteTrials search)
    (valid : start.window) (positive : 0 < start.delta) :
    reached.window ∧ reached.delta = growN rounds start.delta := by
  induction path with
  | here => exact ⟨valid, rfl⟩
  | @later i start rounds next _ running retry _ ih =>
    obtain ⟨window, width⟩ := retry_window valid (finite i _ _ valid running) positive retry
    obtain ⟨valid, values⟩ := ih window (by rw [width]; unfold grow; omega)
    exact ⟨valid, by rw [values, width, growN_step]⟩

theorem reaches_before_return {search : Aspiration.Trials Payload}
    (path : Reaches search i start rounds reached) (returned : run search fuel i start = some result) :
    rounds < fuel := by
  induction path generalizing fuel with
  | here =>
    cases fuel with
    | zero => cases returned
    | succ fuel => exact Nat.zero_lt_succ fuel
  | later running retry _ ih =>
    cases fuel with
    | zero => cases returned
    | succ fuel =>
      rw [run, ite_eq_right (by simp [running]), retry] at returned
      exact Nat.succ_lt_succ (ih returned)

def AbortContract : Prop :=
  ∀ (Payload : Type) (search : Aspiration.Trials Payload) (start reached : Attempt) (n : Nat),
    Reaches search 0 start n reached →
    (search n reached.alpha reached.beta).aborted = true →
    ∃ result, run search (n + 1) 0 start = some result ∧ result.attempt = reached ∧
      result.trial = search n reached.alpha reached.beta ∧ (result.clean = true ↔ n = 0)

theorem abortContract_proven : AbortContract := by
  have stops {Payload : Type} {search : Aspiration.Trials Payload} {i n start reached}
      (path : Reaches search i start n reached) (abort : (search (i + n) reached.alpha reached.beta).aborted = true) :
      run search (n + 1) i start = some ⟨i + n, reached, search (i + n) reached.alpha reached.beta⟩ := by
    induction path with
    | here => exact run_aborted _ _ _ 0 abort
    | @later i start n next reached running retry _ ih =>
      rw [run, ite_eq_right (by simp [running]), retry]
      simpa only [show i + 1 + n = i + (n + 1) by omega] using
        ih (by simpa only [show i + 1 + n = i + (n + 1) by omega] using abort)
  intro Payload search start reached n path abort
  exact ⟨⟨n, reached, _⟩, by simpa only [Nat.zero_add] using stops path (by simpa using abort),
    rfl, rfl, by simp [Result.clean]⟩

private def edgeTrials : Aspiration.Trials Unit := fun i a b =>
  ⟨if i % 2 = 0 then min b 32766 else max a (-32766), (), false⟩

example : ((run edgeTrials 16 0 ⟨-1, 1, 512⟩).map (·.attempt)) =
    some ⟨-32767, 14391, 44286⟩ := by decide

example : ((run edgeTrials 4 0 ⟨-1, 1, 65536⟩).map (·.attempt)) =
    some ⟨-32767, 32767, 221184⟩ := by decide

end NeuralReversi.MidgameAspiration

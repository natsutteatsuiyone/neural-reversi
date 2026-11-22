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

private theorem finish_large (search : Aspiration.Trials Payload) (finite : FiniteTrials search)
    (window : start.window) (large : 65536 ≤ start.delta)
    (fuelBound : rank start.alpha start.beta < fuel) :
    ∃ result, run search fuel i start = some result := by
  induction fuel generalizing i start with
  | zero => omega
  | succ fuel ih =>
    rw [run]
    cases abort : (search i start.alpha start.beta).aborted
    · simp only [Bool.false_eq_true, ite_false]
      cases step : start.retry (search i start.alpha start.beta).score with
      | none => exact ⟨_, rfl⟩
      | some next =>
        have value := finite _ _ _ window abort
        have smaller := large_retry_decreases window large value step
        obtain ⟨valid, width⟩ := retry_window window value (by omega) step
        exact ih valid (by rw [width]; unfold grow; omega) (by omega)
    · simp only [ite_true]
      exact ⟨_, rfl⟩

private theorem warmup (search : Aspiration.Trials Payload) (finite : FiniteTrials search)
    (window : start.window) (positive : 0 < start.delta)
    (large : 65536 ≤ growN rounds start.delta) :
    ∃ result, run search (rounds + 4) i start = some result := by
  induction rounds generalizing i start with
  | zero =>
    exact finish_large search finite window large (Nat.lt_succ_of_le (rank_bound ..))
  | succ rounds ih =>
    rw [Nat.succ_add, run]
    cases abort : (search i start.alpha start.beta).aborted
    · simp only [Bool.false_eq_true, ite_false]
      cases step : start.retry (search i start.alpha start.beta).score with
      | none => exact ⟨_, rfl⟩
      | some next =>
        obtain ⟨valid, width⟩ := retry_window window (finite _ _ _ window abort) positive step
        exact ih valid (by rw [width]; unfold grow; omega) (by rwa [width, growN_step])
    · simp only [ite_true]
      exact ⟨_, rfl⟩

def TerminationContract : Prop :=
  ∀ (Payload : Type) (search : Aspiration.Trials Payload) (start : Attempt),
    FiniteTrials search → start.window → 512 ≤ start.delta →
    ∃ result, run search 16 0 start = some result ∧ Witness search 0 16 start result

theorem terminationContract_proven : TerminationContract := by
  intro Payload search start finite valid delta
  have large : (65536 : Int) ≤ growN 12 512 := by decide
  obtain ⟨result, returned⟩ :=
    warmup (i := 0) (rounds := 12) search finite valid (by omega)
      (Int.le_trans large (growN_mono 12 delta))
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

import NeuralReversi.Threading.SpinLock

namespace NeuralReversi.Threading.SpinLock

structure Backoff where
  busyPolls : Nat
  exponent : Nat

def spinLimit : Nat := 100

def maxExponent : Nat := 6

def Backoff.busy (state : Backoff) : Backoff × Bool :=
  if spinLimit ≤ state.busyPolls + 1 then (⟨0, 0⟩, true)
  else ({ state with busyPolls := state.busyPolls + 1 }, false)

def Backoff.retry (state : Backoff) : Backoff × Nat :=
  let next := if state.exponent < maxExponent then state.exponent + 1 else state.exponent
  ({ state with exponent := next }, 2 ^ state.exponent)

inductive Attempt
  | busyPoll
  | recheckBusy
  | lostCas
  | acquired
  deriving DecidableEq

def observe (busy recheck casSucceeded : Bool) : Attempt :=
  if busy then .busyPoll
  else if recheck then .recheckBusy
  else if casSucceeded then .acquired
  else .lostCas

def ControlContract : Prop :=
  (∀ busy recheck completed,
    observe busy recheck completed = .acquired ↔
      busy = false ∧ recheck = false ∧ completed = true) ∧
  ∀ state : Backoff, state.busyPolls < spinLimit → state.exponent ≤ maxExponent →
    (let (afterBusy, yielded) := state.busy
     afterBusy.busyPolls < spinLimit ∧ afterBusy.exponent ≤ maxExponent ∧
     (yielded = true ↔ state.busyPolls + 1 = spinLimit) ∧
     (yielded = true → afterBusy.busyPolls = 0 ∧ afterBusy.exponent = 0) ∧
     (yielded = false → afterBusy.busyPolls = state.busyPolls + 1 ∧
       afterBusy.exponent = state.exponent)) ∧
    (let (afterRetry, spins) := state.retry
     afterRetry.busyPolls = state.busyPolls ∧
     afterRetry.exponent = min (state.exponent + 1) maxExponent ∧
     1 ≤ spins ∧ spins ≤ 64)

theorem controlContract_proven : ControlContract := by
  refine ⟨?_, ?_⟩
  · intro busy recheck completed
    cases busy <;> cases recheck <;> cases completed <;> decide
  · intro state busy exponent
    have retry : state.retry.1.exponent = min (state.exponent + 1) maxExponent := by
      change (if state.exponent < maxExponent then state.exponent + 1 else state.exponent) = _
      by_cases more : state.exponent < maxExponent
      · rw [ite_eq_left more, Nat.min_eq_left (by omega)]
      · rw [ite_eq_right more, Nat.min_eq_right (by omega)]
        omega
    have lower : 1 ≤ 2 ^ state.exponent := Nat.one_le_pow _ _ (by decide)
    have upper : 2 ^ state.exponent ≤ 64 := Nat.pow_le_pow_right (by decide) exponent
    refine ⟨?_, rfl, retry, lower, upper⟩
    by_cases limit : spinLimit ≤ state.busyPolls + 1
    · have reaches : state.busyPolls + 1 = spinLimit := by omega
      have step : state.busy = (⟨0, 0⟩, true) := by rw [Backoff.busy, ite_eq_left limit]
      simp [step, reaches, spinLimit, maxExponent]
    · have next : state.busyPolls + 1 < spinLimit := by omega
      have short : state.busyPolls + 1 ≠ spinLimit := by omega
      have step : state.busy = (⟨state.busyPolls + 1, state.exponent⟩, false) := by
        rw [Backoff.busy, ite_eq_right limit]
      simp [step, next, short, exponent]

def replay (state : Backoff) : List Attempt → Backoff
  | [] | .acquired :: _ => state
  | .busyPoll :: rest => replay state.busy.1 rest
  | .recheckBusy :: rest | .lostCas :: rest => replay state.retry.1 rest

private theorem replay_bounds (state : Backoff) (attempts : List Attempt)
    (busy : state.busyPolls < spinLimit) (exponent : state.exponent ≤ maxExponent) :
    (replay state attempts).busyPolls < spinLimit ∧ (replay state attempts).exponent ≤ maxExponent := by
  induction attempts generalizing state with
  | nil => exact ⟨busy, exponent⟩
  | cons attempt attempts ih =>
    have control := controlContract_proven.2 state busy exponent
    have busyBound : state.busy.1.busyPolls < spinLimit ∧ state.busy.1.exponent ≤ maxExponent :=
      ⟨control.1.1, control.1.2.1⟩
    have retryBound : state.retry.1.exponent ≤ maxExponent := calc
      state.retry.1.exponent = min (state.exponent + 1) maxExponent := control.2.2.1
      _ ≤ maxExponent := Nat.min_le_right _ _
    cases attempt with
    | busyPoll => exact ih state.busy.1 busyBound.1 busyBound.2
    | recheckBusy => exact ih state.retry.1 busy retryBound
    | lostCas => exact ih state.retry.1 busy retryBound
    | acquired => exact ⟨busy, exponent⟩

def BackoffTraceContract : Prop :=
  ∀ attempts, let state := replay ⟨0, 0⟩ attempts
    state.busyPolls < spinLimit ∧ state.exponent ≤ maxExponent

theorem backoffTraceContract_proven : BackoffTraceContract :=
  fun attempts => replay_bounds ⟨0, 0⟩ attempts (by decide) (by decide)

end NeuralReversi.Threading.SpinLock

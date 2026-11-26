import NeuralReversi.PvsParallel.SplitSchedulerMir.Program

namespace NeuralReversi.PvsParallel.SplitSchedulerMir

open Threading
open AtomicWord (Event Read Write SequencedBefore signed32At)
open CutoffVisibility (History HappensBefore)

set_option maxRecDepth 8000 in
set_option maxHeartbeats 3000000 in
private theorem idle_closed : ∀ pc, awaiting pc = true → preserves awaiting (idle pc) = true := by decide

private theorem exit_of_guarded {code : Fin n → Op n} {writes : List Write} {pc : Fin n}
    {log : List (Guard writes)} (waiting : Fin n → Bool)
    (closed : ∀ i, waiting i = true → preserves waiting (code i) = true)
    (run : Returned code writes pc log) (start : waiting pc = true)
    (openPool : Guard.shutdown true ∉ log) :
    ∃ pre src post, log = pre ++ .mask src :: post ∧ (writes.get src).value = 0 := by
  induction run with
  | done h =>
    specialize closed _ start
    simp [h, preserves] at closed
  | next h mem _ ih =>
    specialize closed _ start
    rw [h] at closed
    exact ih (List.all_eq_true.mp closed _ mem) openPool
  | @shutdown pc running closedPC b more h _ ih =>
    cases b with
    | true => simp at openPool
    | false =>
      obtain ⟨pre, src, post, eq, zero⟩ := ih (by simpa [h, preserves] using closed _ start)
        (fun bad => openPool (List.mem_cons_of_mem _ bad))
      exact ⟨.shutdown false :: pre, src, post, by simp [eq], zero⟩
  | @mask pc busy empty more src h _ ih =>
    by_cases zero : (writes.get src).value = 0
    · exact ⟨[], src, more, rfl, zero⟩
    · obtain ⟨pre, found, post, eq, isZero⟩ := ih (by rw [ite_eq_right zero]; simpa [h, preserves] using closed _ start)
        (fun bad => openPool (List.mem_cons_of_mem _ bad))
      exact ⟨.mask src :: pre, found, post, by simp [eq], isZero⟩

theorem idle_acquired_zero {mask : List Write} {log : List (Guard mask)}
    (run : Returned idle mask 0 log) (openPool : Guard.shutdown true ∉ log) :
    ∃ pre src post, log = pre ++ .mask src :: post ∧ (mask.get src).value = 0 :=
  exit_of_guarded awaiting idle_closed run (by decide) openPool

private def quiet (pc : Fin 80) := pc.val == 16 || pc.val == 78 || pc.val == 79

private def silent : Op 80 → Bool
  | .any targets => targets.all quiet
  | .done => true
  | _ => false

private def exitsOnZero : Op 80 → Bool
  | .mask _ empty => quiet empty
  | _ => true

set_option maxRecDepth 8000 in
set_option maxHeartbeats 3000000 in
private theorem exits : (∀ pc, quiet pc = true → silent (idle pc) = true) ∧
    (∀ pc, exitsOnZero (idle pc) = true) := by decide

private theorem silent_guards {pc} (run : Returned idle mask pc log) (atExit : quiet pc = true) : log = [] := by
  induction run with
  | done => rfl
  | next op mem _ ih => exact ih (List.all_eq_true.mp (by simpa [op,silent] using exits.1 _ atExit) _ mem)
  | shutdown op _ _ => simpa [op,silent] using exits.1 _ atExit
  | mask src op _ _ => simpa [op,silent] using exits.1 _ atExit

/-- The zero-mask branch goes directly to the idle-loop return. Its acquire is the
    last guard, so the owner's later cancellation check uses a known clock position. -/
theorem after_zero {pc} (run : Returned idle mask pc log) (atRead : log = pre ++ .mask src :: post)
    (zero : (mask.get src).value = 0) : post = [] := by
  induction run generalizing pre with
  | done => cases pre <;> simp_all
  | next _ _ _ ih => exact ih atRead
  | shutdown op tail ih =>
    cases pre with
    | nil => simp at atRead
    | cons _ pre => exact ih (List.cons.inj atRead).2
  | @mask pc busy empty more source op tail ih =>
    cases pre with
    | nil =>
      obtain ⟨eq,rfl⟩ := List.cons.inj atRead
      cases eq
      rw [ite_eq_left zero] at tail
      exact silent_guards tail (by simpa [op,exitsOnZero] using exits.2 pc)
    | cons _ pre => exact ih (List.cons.inj atRead).2

structure Clock where
  worker : Nat
  tick : Nat → Nat
  advances : ∀ i j, i < j → tick i < tick j

def Clock.atStep (clock : Clock) (i : Nat) : Event := ⟨clock.worker, clock.tick i⟩

private theorem Clock.before (clock : Clock) {i j : Nat} (lt : i < j) :
    SequencedBefore (clock.atStep i) (clock.atStep j) := ⟨rfl, clock.advances i j lt⟩

inductive OwnerStep (mask : List Write)
  | clear | mark | seed | assign | finalize | best | guard (g : Guard mask)

private def initSteps {mask : List Write} : Phase → List (OwnerStep mask)
  | .clear => [.clear] | .mark => [.mark] | .seed => [.seed] | .assign => [.assign] | _ => []

def ownerSteps (log : List (Guard mask)) : List (OwnerStep mask) := ownerPhases.flatMap fun
  | .init => initPhases.flatMap initSteps
  | .idle => log.map .guard
  | .finalize => [.finalize]
  | .best => [.best]
  | _ => []

theorem ownerSteps_eq (log : List (Guard mask)) : ownerSteps log =
    [.clear, .mark, .seed, .assign] ++ log.map .guard ++ [.finalize, .best] := by
  simp [ownerSteps, ownerPhases, initPhases, initSteps]

/-- Read-from and modification order come from the atomic execution of this slot. The clock
    enumerates the projected owner actions, allowing arbitrary gaps between them. -/
structure OwnerMemory (history : History) (mask scores : List Write) (act : SplitCompletion.Activation mask)
    (clock : Clock) (log : List (Guard mask)) (scoreSource : Fin scores.length) where
  ownerBit : (mask.get act.start).event = clock.atStep 1
  maskBeforeReuse : ∀ pre src post, log = pre ++ .mask src :: post →
    ∀ i : Fin mask.length, i.val = act.start.val + act.actions.length + 1 →
      SequencedBefore (clock.atStep (4 + pre.length)) (mask.get i).event
  maskWR : ∀ pre src post, log = pre ++ .mask src :: post → ∀ i : Fin mask.length,
    HappensBefore history (mask.get i).event (clock.atStep (4 + pre.length)) → i.val ≤ src.val
  maskRW : ∀ pre src post, log = pre ++ .mask src :: post → ∀ i : Fin mask.length,
    HappensBefore history (clock.atStep (4 + pre.length)) (mask.get i).event → src.val < i.val
  sync : ∀ pre src post, log = pre ++ .mask src :: post → ∀ i : Fin mask.length,
    AtomicWord.InReleaseSequence mask i src →
      history.otherSync (mask.get i).event (clock.atStep (4 + pre.length))
  scoreWR : ∀ i : Fin scores.length,
    HappensBefore history (scores.get i).event (clock.atStep (log.length + 5)) → i.val ≤ scoreSource.val
  scoreRW : ∀ i : Fin scores.length,
    HappensBefore history (clock.atStep (log.length + 5)) (scores.get i).event → scoreSource.val < i.val

def scoreLoad (scores : List Write) (source : Fin scores.length) (clock : Clock) (log : List (Guard mask)) :
    Read scores := ⟨clock.atStep (log.length + 5), source, false⟩

theorem retired_after_assignment (history : History) (mask scores : List Write) (act : SplitCompletion.Activation mask)
    (clock : Clock) (log : List (Guard mask)) (source : Fin scores.length)
    (run : Returned idle mask 0 log) (openPool : Guard.shutdown true ∉ log)
    (memory : OwnerMemory history mask scores act clock log source) :
    ∃ retired : SplitCompletion.Retirement history mask act,
      SequencedBefore retired.read.event (scoreLoad scores source clock log).event ∧
      SequencedBefore (clock.atStep 3) retired.read.event ∧
      retired.read.source.val = act.start.val + act.actions.length ∧
      retired.read.event = clock.atStep (3 + log.length) := by
  obtain ⟨pre, src, post, eq, zero⟩ := idle_acquired_zero run openPool
  have last := after_zero run eq zero
  let retired : SplitCompletion.Retirement history mask act := {
    read := ⟨clock.atStep (4 + pre.length), src, true⟩
    acquire := rfl
    zero := zero
    afterOwnerBit := by rw [memory.ownerBit]; exact clock.before (by omega)
    beforeReuse := memory.maskBeforeReuse pre src post eq
    writeRead := memory.maskWR pre src post eq
    readWrite := memory.maskRW pre src post eq
    synchronize := fun i _ rel => memory.sync pre src post eq i rel }
  refine ⟨retired, clock.before ?_, clock.before (by change 3 < 4 + pre.length; omega),
    (SplitCompletion.emptyReadContract_proven history mask act retired).1,?_⟩
  · simp [eq] <;> omega
  · change clock.atStep (4 + pre.length) = clock.atStep (3 + log.length)
    apply congrArg clock.atStep
    simp [eq,last] <;> omega

theorem retired_before_score (history : History) (mask scores : List Write) (act : SplitCompletion.Activation mask)
    (clock : Clock) (log : List (Guard mask)) (source : Fin scores.length)
    (run : Returned idle mask 0 log) (openPool : Guard.shutdown true ∉ log)
    (memory : OwnerMemory history mask scores act clock log source) :
    ∃ retired : SplitCompletion.Retirement history mask act,
      SequencedBefore retired.read.event (scoreLoad scores source clock log).event ∧
      SequencedBefore (clock.atStep 2) retired.read.event ∧
      retired.read.source.val = act.start.val + act.actions.length := by
  obtain ⟨r,load,assigned,atEnd,_⟩ := retired_after_assignment history mask scores act clock log source run openPool memory
  exact ⟨r,load,⟨assigned.1,Nat.lt_trans (clock.advances 2 3 (by decide)) assigned.2⟩,atEnd⟩

/-- The last score writer is either the owner before the idle-loop guard or a worker before its
    release. This branch-independent premise also covers initialization when no report raised best. -/
theorem returned_last_score (history : History) (mask scores : List Write) (act : SplitCompletion.Activation mask)
    (clock : Clock) (log : List (Guard mask)) (source last : Fin scores.length)
    (run : Returned idle mask 0 log) (openPool : Guard.shutdown true ∉ log)
    (memory : OwnerMemory history mask scores act clock log source)
    (origin : (scores.get last).event = clock.atStep 2 ∨
      ∃ i : Fin act.actions.length, (act.actions.get i).release = true ∧
        SequencedBefore (scores.get last).event (act.actions.get i).event)
    (noReuse : ∀ i : Fin scores.length, i.val = last.val + 1 →
      SequencedBefore (scoreLoad scores source clock log).event (scores.get i).event) :
    source = last ∧ signed32At scores source = signed32At scores last := by
  obtain ⟨retired, order, afterSeed, _⟩ := retired_before_score history mask scores act clock log source run openPool memory
  apply SplitReturn.Observation.scoreReadContract_proven history scores (scoreLoad scores source clock log) last retired
  refine ⟨?_, order, noReuse, memory.scoreWR, memory.scoreRW⟩
  rcases origin with first | worker
  · exact Or.inl (first ▸ afterSeed)
  · exact Or.inr worker

end NeuralReversi.PvsParallel.SplitSchedulerMir

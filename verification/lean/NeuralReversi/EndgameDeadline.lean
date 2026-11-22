import NeuralReversi.EndgameTiming
import NeuralReversi.Threading.SearchAbort

namespace NeuralReversi.EndgameDeadline

open EndgameTiming (beforeDeadline)
open SearchPublication (Attempt Completed Snapshot finish step)
open Threading.AtomicWord (Event Read SequencedBefore Write)
open Threading.CutoffVisibility (Coherent FlagRead HappensBefore History)
open Threading.SearchAbort (Stopped generation request requestFor)
open Threading.SplitShutdown (MessageRead)

structure SignalEffect where
  expired : Bool
  marked : Bool
  abortWrite : Option (BitVec 64)
  deriving DecidableEq

def checkTime (deadline : Option Nat) (elapsed : Nat) (marked : Bool) (word : BitVec 64)
    (target : Option (BitVec 32)) : SignalEffect :=
  if beforeDeadline deadline elapsed then ⟨false, marked, none⟩
  else ⟨true, true, target.elim (some (request word)) (requestFor word)⟩

def CheckContract : Prop :=
  ∀ (elapsed limit : Nat) (marked : Bool) (word : BitVec 64) (target : Option (BitVec 32)),
    checkTime none elapsed marked word target = ⟨false, marked, none⟩ ∧
    (elapsed < limit → checkTime (some limit) elapsed marked word target = ⟨false, marked, none⟩) ∧
    (limit ≤ elapsed →
      let next := checkTime (some limit) elapsed marked word target
      next.expired = true ∧ next.marked = true ∧
        (target = none ∨ target = some (generation word) →
          next.abortWrite = some (request word) ∧ Stopped (request word)) ∧
        (∀ old, target = some old → old ≠ generation word → next.abortWrite = none))

theorem checkContract_proven : CheckContract := by
  intro elapsed limit marked word target
  refine ⟨rfl, fun early => by simp [checkTime, beforeDeadline, early], fun expired => ?_⟩
  simp only [checkTime, beforeDeadline, Option.map_some, Option.getD_some, Nat.not_lt_of_ge expired,
    decide_false, Bool.false_eq_true, ite_false]
  obtain ⟨_, _, _, stopped, requestAt, _, _⟩ := Threading.SearchAbort.generationWordContract_proven word
  refine ⟨trivial, trivial, ?_, ?_⟩
  · rintro (rfl | rfl)
    · exact ⟨rfl, stopped⟩
    · exact ⟨by simpa using requestAt (generation word), stopped⟩
  · intro old same stale
    subst target
    simpa [Ne.symm stale] using requestAt old

inductive Boundary where
  | nextLine
  | observedAbort
  | expired
  deriving DecidableEq

def afterLine (aborted : Bool) (deadline : Option Nat) (elapsed : Nat) : Boundary :=
  if aborted then .observedAbort
  else if beforeDeadline deadline elapsed then .nextLine
  else .expired

def boundaryResult (state : Completed) : Boundary → Option (Option Snapshot)
  | .nextLine => none
  | .observedAbort | .expired => some (finish 0 state .interrupted)

def BoundaryContract : Prop :=
  ∀ (state : Completed) (deadline : Option Nat) (elapsed : Nat),
    afterLine true deadline elapsed = .observedAbort ∧
    afterLine false none elapsed = .nextLine ∧
    (∀ limit, afterLine false (some limit) elapsed = .expired ↔ limit ≤ elapsed) ∧
    (∀ aborted, let decision := afterLine aborted deadline elapsed
      boundaryResult state decision =
        if decision = .nextLine then none else some state.last)

theorem boundaryContract_proven : BoundaryContract := by
  intro state deadline elapsed
  refine ⟨rfl, rfl, fun limit => ?_, fun aborted => ?_⟩
  · simp [afterLine, beforeDeadline, Nat.not_lt]
  · cases afterLine aborted deadline elapsed <;> rfl

def SetOnly (history : History) (slot : Nat) : Prop :=
  ∀ i : Fin (history.flags slot).length, 0 < i.val → ((history.flags slot).get i).value = true

private theorem read_set (history : History) (coherent : Coherent history)
    (read : FlagRead history.flags) (present : read ∈ history.flagReads)
    (mark : Fin (history.flags read.slot).length) (positive : 0 < mark.val)
    (setOnly : SetOnly history read.slot)
    (visible : HappensBefore history ((history.flags read.slot).get mark).event read.event) :
    read.value = true := by
  exact setOnly read.source (Nat.lt_of_lt_of_le positive (coherent.flagWriteRead read present mark visible))

structure TimerSignal (history : History) (slot : Nat) where
  mark : Fin (history.flags slot).length
  abort : Fin history.epochs.length
  prior : Fin history.epochs.length
  target : BitVec 32
  markPositive : 0 < mark.val
  beforeAbort : SequencedBefore ((history.flags slot).get mark).event (history.epochs.get abort).event
  released : (history.epochs.get abort).release = true
  previous : prior.val + 1 = abort.val
  requested : requestFor (history.epochs.get prior).value target = some (history.epochs.get abort).value

private theorem TimerSignal.stops (signal : TimerSignal history slot) :
    Stopped (history.epochs.get signal.abort).value ∧
      generation (history.epochs.get signal.abort).value = signal.target := by
  obtain ⟨_, _, generationKept, stopped, requestAt, _, _⟩ :=
    Threading.SearchAbort.generationWordContract_proven (history.epochs.get signal.prior).value
  have targeted := requestAt signal.target
  split at targeted
  next sameTarget =>
    rw [signal.requested] at targeted
    rw [Option.some.inj targeted]
    exact ⟨stopped, generationKept.trans sameTarget⟩
  · rw [signal.requested] at targeted
    cases targeted

structure Dropped (history : History) (slot : Nat) (signal : TimerSignal history slot)
    (observed : Read history.epochs) (deadline : FlagRead history.flags) : Prop where
  observedMember : observed ∈ history.epochReads
  acquire : observed.acquire = true
  reachesAbort : signal.abort.val ≤ observed.source.val
  laterRequests : ∀ i : Fin history.epochs.length, signal.abort.val < i.val → i.val ≤ observed.source.val →
    (history.epochs.get i).rmw = true ∧ ∃ prior : Fin history.epochs.length,
      prior.val + 1 = i.val ∧ (history.epochs.get i).value = request (history.epochs.get prior).value
  deadlineMember : deadline ∈ history.flagReads
  deadlineSlot : deadline.slot = slot
  beforeRecord : SequencedBefore observed.event deadline.event

private theorem stopped_through (signal : TimerSignal history slot)
    (drop : Dropped history slot signal observed deadline) (i : Fin history.epochs.length)
    (start : signal.abort.val ≤ i.val) (through : i.val ≤ observed.source.val) :
    Stopped (history.epochs.get i).value ∧ generation (history.epochs.get i).value = signal.target := by
  by_cases first : signal.abort.val = i.val
  · rw [← Fin.ext first]
    exact signal.stops
  · obtain ⟨_, prior, adjacent, value⟩ := drop.laterRequests i (by omega) through
    obtain ⟨_, _, generationKept, stopped, _⟩ :=
      Threading.SearchAbort.generationWordContract_proven (history.epochs.get prior).value
    rw [value, generationKept]
    exact ⟨stopped, (stopped_through signal drop prior (by omega) (by omega)).2⟩
termination_by i.val
decreasing_by omega

def recordAbort (already : Bool) (deadline : Option Bool) : Bool :=
  if deadline.getD false then true else already

def DropContract : Prop :=
  ∀ (history : History) (_ : Coherent history) (slot : Nat) (signal : TimerSignal history slot)
    (observed : Read history.epochs) (deadline : FlagRead history.flags),
    SetOnly history slot → Dropped history slot signal observed deadline →
    Stopped (history.epochs.get observed.source).value ∧
      generation (history.epochs.get observed.source).value = signal.target ∧
      deadline.value = true ∧
      ∀ (already : Bool) (minimum : Nat) (state : Completed) (attempt : Attempt),
        attempt.index < state.marks.length →
        recordAbort already (some deadline.value) = true ∧
          finish minimum (step minimum state { attempt with aborted := true }) .interrupted = state.last

theorem dropContract_proven : DropContract := by
  intro history coherent slot signal observed deadline setOnly drop
  obtain ⟨stopped, matched⟩ := stopped_through signal drop observed.source drop.reachesAbort (Nat.le_refl _)
  have synchronized : HappensBefore history (history.epochs.get signal.abort).event observed.event :=
    .synchronized signal.abort observed drop.observedMember drop.acquire
      ⟨signal.released, drop.reachesAbort, fun i after read => (drop.laterRequests i after read).1⟩
  have visible := HappensBefore.trans (.sequenced signal.beforeAbort)
    (.trans synchronized (.sequenced drop.beforeRecord))
  have marked : deadline.value = true := by
    cases deadline with
    | mk deadlineSlot event source =>
      have same : deadlineSlot = slot := drop.deadlineSlot
      subst deadlineSlot
      exact read_set history coherent _ drop.deadlineMember signal.mark signal.markPositive setOnly visible
  refine ⟨stopped, matched, marked, fun already minimum state attempt valid => ⟨by simp [recordAbort, marked], ?_⟩⟩
  have progress := SearchPublication.progressContract_proven minimum state { attempt with aborted := true } valid
  rw [progress.2.1 (Or.inl rfl)]
  rfl

def ResultTimeoutContract : Prop :=
  ∀ (history : History) (_ : Coherent history) (messages : List Write) (sent : Fin messages.length)
    (received : Read messages) (load : FlagRead history.flags)
    (recorded : Fin (history.flags load.slot).length),
    SetOnly history load.slot → 0 < recorded.val → load ∈ history.flagReads →
    SequencedBefore ((history.flags load.slot).get recorded).event (messages.get sent).event →
    MessageRead history sent received → SequencedBefore received.event load.event → load.value = true

theorem resultTimeoutContract_proven : ResultTimeoutContract := by
  intro history coherent messages sent received load recorded setOnly positive present beforeSend matchResult afterReceive
  have message := (Threading.SplitShutdown.messageHandoffContract_proven history messages sent received matchResult).2
  exact read_set history coherent load present recorded positive setOnly
    (.trans (.sequenced beforeSend) (.trans message (.sequenced afterReceive)))

example : afterLine false (some 0) 0 = .expired ∧ afterLine true (some 0) 0 = .observedAbort ∧
    recordAbort false (some false) = false := by decide

end NeuralReversi.EndgameDeadline

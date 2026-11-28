module
public import NeuralReversi.Threading.SplitPublication
public import NeuralReversi.Threading.ArcDrop

@[expose] public section

namespace NeuralReversi.Threading.SplitShutdown

open AtomicWord (Read Write SequencedBefore Event InReleaseSequence)
open CutoffVisibility (History HappensBefore)
open SplitCompletion (RetainedThrough retainedAccessContract_proven)

inductive PoolState where
  | live (otherRefs : Nat) (rootRef : Bool)
  | exiting

inductive PoolAction where
  | acquireOther
  | releaseOther
  | enterRoot
  | leaveRoot
  | startExit
  deriving DecidableEq

inductive PoolStep : PoolState → PoolAction → PoolState → Prop
  | acquireOther : (0 < otherRefs ∨ rootRef = true) →
      PoolStep (.live otherRefs rootRef) .acquireOther (.live (otherRefs + 1) rootRef)
  | releaseOther :
      PoolStep (.live (otherRefs + 1) rootRef) .releaseOther (.live otherRefs rootRef)
  | enterRoot :
      PoolStep (.live (otherRefs + 1) false) .enterRoot (.live (otherRefs + 1) true)
  | leaveRoot :
      PoolStep (.live otherRefs true) .leaveRoot (.live otherRefs false)
  | startExit : PoolStep (.live 0 false) .startExit .exiting

inductive PoolRun : PoolState → List PoolAction → PoolState → Prop
  | done : PoolRun state [] state
  | step : PoolStep state action next → PoolRun next remaining final →
      PoolRun state (action :: remaining) final

private theorem root_kept (run : PoolRun state changes final)
    (held : ∃ refs, state = .live refs true) (retained : .leaveRoot ∉ changes) :
    ∃ refs, final = .live refs true := by
  induction run with
  | done => exact held
  | step first _ ih =>
    obtain ⟨refs, rfl⟩ := held
    have later := fun member => retained (List.mem_cons_of_mem _ member)
    cases first with
    | acquireOther _ => exact ih ⟨_, rfl⟩ later
    | releaseOther => exact ih ⟨_, rfl⟩ later
    | leaveRoot => exact (retained (List.mem_cons_self ..)).elim

def PoolRetentionContract : Prop :=
  ∀ (refs : Nat) (changes : List PoolAction) (final : PoolState),
    PoolRun (.live refs true) changes final → .leaveRoot ∉ changes →
    ∃ refs, final = .live refs true ∧ ¬ PoolStep final .startExit .exiting

theorem poolRetentionContract_proven : PoolRetentionContract := by
  intro refs changes final run retained
  obtain ⟨remaining, rfl⟩ := root_kept run ⟨refs, rfl⟩ retained
  exact ⟨remaining, rfl, fun exit => by cases exit⟩

def ExitReadContract : Prop :=
  ∀ (history : History) (writes : List Write) (read : Read writes),
    (∀ i : Fin writes.length, (writes.get i).value ≠ 0 →
      ∃ wait release : Event, RetainedThrough history read.event wait ∧
        SequencedBefore wait release ∧ HappensBefore history release (writes.get i).event) →
    (∀ i : Fin writes.length, HappensBefore history read.event (writes.get i).event →
      read.source.val < i.val) →
    (writes.get read.source).value = 0

theorem exitReadContract_proven : ExitReadContract := by
  intro history writes read afterRelease readWrite
  by_cases zero : (writes.get read.source).value = 0
  · exact zero
  · obtain ⟨wait, release, kept, before, shutdown⟩ := afterRelease read.source zero
    have ordered := retainedAccessContract_proven history read.event wait release kept before
    exact (Nat.lt_irrefl _ (readWrite read.source (.trans ordered shutdown))).elim

/-- One slot in the unbounded channel. The receiver's successful acquire
    observes WRITE, bit zero of the slot's state word. -/
structure MessageRead (history : History) {writes : List Write}
    (published : Fin writes.length) (read : Read writes) : Prop where
  released : (writes.get published).release = true
  acquired : read.acquire = true
  ready : (writes.get read.source).value.getLsbD 0 = true
  beforePublication : ∀ i : Fin writes.length, i.val < published.val →
    (writes.get i).value.getLsbD 0 = false
  afterPublication : ∀ i : Fin writes.length, published.val < i.val → (writes.get i).rmw = true
  synchronize : read.acquire = true → InReleaseSequence writes published read.source →
    history.otherSync (writes.get published).event read.event

def MessageHandoffContract : Prop :=
  ∀ (history : History) (writes : List Write) (published : Fin writes.length) (read : Read writes),
    MessageRead history published read →
    published.val ≤ read.source.val ∧ HappensBefore history (writes.get published).event read.event

theorem messageHandoffContract_proven : MessageHandoffContract := by
  intro history writes published read delivery
  have source : published.val ≤ read.source.val := by
    apply Nat.le_of_not_gt
    intro earlier
    have absent := delivery.beforePublication read.source earlier
    rw [delivery.ready] at absent
    cases absent
  have sequence : InReleaseSequence writes published read.source :=
    ⟨delivery.released, source, fun i later _ => delivery.afterPublication i later⟩
  exact ⟨source, .other (delivery.synchronize delivery.acquired sequence)⟩

inductive ExitWrite (history : History) (destroy : Event) : Event → Prop
  | direct : SequencedBefore destroy store → ExitWrite history destroy store
  | message {writes : List Write} (published : Fin writes.length) (read : Read writes) :
      SequencedBefore destroy (writes.get published).event → MessageRead history published read →
      SequencedBefore read.event store → ExitWrite history destroy store

theorem ExitWrite.orders (cause : ExitWrite history destroy store) :
    HappensBefore history destroy store := by
  cases cause with
  | direct localOrder => exact .sequenced localOrder
  | message published read before delivery after =>
    have ordered := (messageHandoffContract_proven history _ published read delivery).2
    exact .trans (.sequenced before) (.trans ordered (.sequenced after))

def ArcExitReadContract : Prop :=
  ∀ (history : History) (counts exits : List Write) (strong : ArcDrop.StrongHistory counts)
    (last : ArcDrop.LastRelease strong) (rootRelease : Fin counts.length)
    (wait destroy : Event) (read : Read exits),
    strong.publication.val < rootRelease.val → (counts.get rootRelease).release = true →
    ArcDrop.DestructorAcquire history last.read destroy →
    RetainedThrough history read.event wait →
    SequencedBefore wait (counts.get rootRelease).event →
    (∀ i : Fin exits.length, (exits.get i).value ≠ 0 →
      ExitWrite history destroy (exits.get i).event) →
    (∀ i : Fin exits.length, HappensBefore history read.event (exits.get i).event →
      read.source.val < i.val) →
    (exits.get read.source).value = 0

theorem arcExitReadContract_proven : ArcExitReadContract := by
  intro history counts exits strong last rootRelease wait destroy read published released
    acquire retained before exitWrite coherent
  have toDestructor := ArcDrop.releaseToDestructorContract_proven history counts strong last
    destroy rootRelease published released acquire
  exact exitReadContract_proven history exits read
    (fun i marked => ⟨wait, (counts.get rootRelease).event, retained, before,
      .trans toDestructor (exitWrite i marked).orders⟩) coherent

def FailedBookingContract : Prop :=
  ∀ (history : History) (writes : List Write) (failed : Fin writes.length) (read : Read writes),
    SpinLock.OrderedAccess history (writes.get failed).event read.event →
    (∀ i : Fin writes.length, failed.val ≤ i.val → (writes.get i).value = 1) →
    (∀ i : Fin writes.length, HappensBefore history (writes.get i).event read.event →
      i.val ≤ read.source.val) →
    (writes.get read.source).value = 1

theorem failedBookingContract_proven : FailedBookingContract := by
  intro history writes failed read locked sticky writeRead
  exact sticky read.source (writeRead failed locked.orders)

example : PoolRun (.live 1 false) [.enterRoot, .releaseOther] (.live 0 true) :=
  .step .enterRoot (.step .releaseOther .done)

example : PoolRun (.live 1 false) [.enterRoot, .releaseOther, .leaveRoot, .startExit] .exiting :=
  .step .enterRoot (.step .releaseOther (.step .leaveRoot (.step .startExit .done)))

end NeuralReversi.Threading.SplitShutdown

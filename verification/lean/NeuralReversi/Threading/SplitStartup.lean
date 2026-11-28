module
public import NeuralReversi.Threading.SplitPublication

@[expose] public section

namespace NeuralReversi.Threading.SplitStartup

open AtomicWord (Event SequencedBefore Write Read)
open CutoffVisibility (History HappensBefore)
open SplitCompletion (Action Worker MaskState Allowed Legal Activation Retirement initial step run)
open SplitPublication (PublishedThrough LockedFlagRead)

def GuardPublicationContract : Prop :=
  ∀ (history : History) (armed published access : Event),
    SequencedBefore armed published → PublishedThrough history published access →
    HappensBefore history armed access

theorem guardPublicationContract_proven : GuardPublicationContract := by
  intro history armed published access guarded visible
  exact .trans (.sequenced guarded)
    (SplitPublication.initializationContract_proven history published access visible)

private theorem closed_only_finishes (closed : state.accepting = false)
    (legal : Legal state actions) (member : action ∈ actions) : action.release = true := by
  induction actions generalizing state with
  | nil => simp at member
  | cons first rest ih =>
    rcases legal with ⟨allowed, legal⟩
    have finishes : first.release = true := by
      cases first with
      | book _ _ =>
        have contradiction : state.accepting = true := allowed
        rw [closed] at contradiction
        cases contradiction
      | finish _ _ => rfl
    rcases List.mem_cons.mp member with same | later
    · exact same.symm ▸ finishes
    · apply ih (state := step state first) _ legal later
      simp [step, closed]

def StartupAbortContract : Prop :=
  ∀ (history : History) (writes : List Write) (activation : Activation writes)
    (retirement : Retirement history writes activation) (before after : List Action)
    (abandon clear : Event),
    activation.actions = before ++ (.finish activation.owner abandon :: after) →
    SequencedBefore retirement.read.event clear →
    let unwound := step (run (initial activation.owner) before) (.finish activation.owner abandon)
    unwound.has activation.owner = false ∧ unwound.accepting = false ∧
    (∀ worker, worker ≠ activation.owner →
      unwound.has worker = (run (initial activation.owner) before).has worker) ∧
    (∀ action ∈ after, action.release = true) ∧
    (∀ worker event, .book worker event ∈ before →
      ∃ release, .finish worker release ∈ activation.actions) ∧
    (∀ i : Fin activation.actions.length, (activation.actions.get i).release = true →
      HappensBefore history (activation.actions.get i).event clear)

theorem startupAbortContract_proven : StartupAbortContract := by
  intro history writes activation retirement before after abandon clear steps clearing
  have partition := SplitCompletion.legal_append (initial activation.owner) before
    (.finish activation.owner abandon :: after) (steps ▸ activation.legal)
  have remaining := partition.2.2
  obtain ⟨_, _, released⟩ :=
    SplitCompletion.emptyReadContract_proven history writes activation retirement
  have closed : (step (run (initial activation.owner) before) (.finish activation.owner abandon)).accepting =
      false := by simp [step, Action.release]
  refine ⟨?_, closed, ?_, ?_, ?_, ?_⟩
  · simp [SplitCompletion.finish_has]
  · intro worker different
    simp [SplitCompletion.finish_has, different]
  · intro action member
    apply closed_only_finishes (state := step (run (initial activation.owner) before)
      (.finish activation.owner abandon)) closed remaining member
  · intro worker event booked
    apply released
    rw [steps]
    exact List.mem_append_left _ booked
  · intro i released
    exact .trans (SplitCompletion.handoffContract_proven history writes activation retirement i released)
      (.sequenced clearing)

inductive BookingCheck (history : History) {writes : List Write}
    (read : Read writes) (release : Event) : Prop
  | busy (booked : Fin writes.length) : (writes.get booked).value = 1 →
      LockedFlagRead history read booked booked.val → BookingCheck history read release
  | released : SpinLock.OrderedAccess history release read.event → BookingCheck history read release

def RebookingContract : Prop :=
  ∀ (history : History) (writes : List Write) (read : Read writes)
    (oldRelease nextBooking : Event),
    BookingCheck history read oldRelease → (writes.get read.source).value = 0 →
    SequencedBefore read.event nextBooking → HappensBefore history oldRelease nextBooking

theorem rebookingContract_proven : RebookingContract := by
  intro history writes read oldRelease nextBooking checked idle next
  cases checked with
  | busy booked setTrue locked =>
    have busy := (SplitPublication.lockedCheckContract_proven history writes booked read setTrue locked).2
    rw [busy] at idle
    cases idle
  | released locked =>
    exact .trans locked.orders (.sequenced next)

end NeuralReversi.Threading.SplitStartup

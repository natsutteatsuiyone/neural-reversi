module
public import NeuralReversi.Threading.SpinLock
public import NeuralReversi.Threading.SplitCompletion

@[expose] public section

namespace NeuralReversi.Threading.SplitPublication

open AtomicWord (Event Write Read ReadWindow SequencedBefore InReleaseSequence)
open CutoffVisibility (History HappensBefore)
open SplitCompletion (RetainedThrough)

structure SearchWake (history : History) (writes : List Write) where
  idle : Fin writes.length
  booked : Fin writes.length
  consecutive : booked.val = idle.val + 1
  idleValue : (writes.get idle).value = 0
  posted : (writes.get booked).value = 1
  release : (writes.get booked).release = true
  read : Read writes
  acquire : read.acquire = true
  ready : (writes.get read.source).value = 1
  window : ReadWindow read (HappensBefore history) idle booked.val
  sync : ∀ i : Fin writes.length, read.acquire = true →
    InReleaseSequence writes i read.source → history.otherSync (writes.get i).event read.event

def WakeContract : Prop :=
  ∀ (history : History) (writes : List Write) (wake : SearchWake history writes),
    wake.read.source = wake.booked ∧
    HappensBefore history (writes.get wake.booked).event wake.read.event

theorem wakeContract_proven : WakeContract := by
  intro history writes wake
  have ⟨lower, upper⟩ := wake.window.bounds
  have next := wake.consecutive
  have fresh : wake.read.source = wake.booked := by
    have distinct : wake.read.source.val ≠ wake.idle.val := by
      intro same
      have ready := wake.ready
      rw [Fin.ext same, wake.idleValue] at ready
      cases ready
    exact Fin.ext (by omega)
  have source : InReleaseSequence writes wake.booked wake.read.source := by
    rw [fresh]
    exact ⟨wake.release, Nat.le_refl _, fun _ after through => by omega⟩
  exact ⟨fresh, .other (wake.sync wake.booked wake.acquire source)⟩

def BusyCheckContract : Prop :=
  ∀ (history : History) (writes : List Write) (booked : Fin writes.length) (read : Read writes),
    (writes.get booked).value = 1 →
    ReadWindow read (HappensBefore history) booked booked.val →
    read.source = booked ∧ (writes.get read.source).value = 1

theorem busyCheckContract_proven : BusyCheckContract := by
  intro history writes booked read busy window
  have ⟨lower, upper⟩ := window.bounds
  have source := Fin.ext (Nat.le_antisymm upper lower)
  exact ⟨source, source ▸ busy⟩

structure LockedFlagRead (history : History) {writes : List Write} (read : Read writes)
    (first : Fin writes.length) (last : Nat) where
  afterFirst : SpinLock.OrderedAccess history (writes.get first).event read.event
  beforeNext : ∀ i : Fin writes.length, i.val = last + 1 →
    SpinLock.OrderedAccess history read.event (writes.get i).event
  writeRead : ∀ i : Fin writes.length, HappensBefore history (writes.get i).event read.event →
    i.val ≤ read.source.val
  readWrite : ∀ i : Fin writes.length, HappensBefore history read.event (writes.get i).event →
    read.source.val < i.val

theorem LockedFlagRead.window {writes : List Write} {read : Read writes}
    {first : Fin writes.length} {last : Nat} (checked : LockedFlagRead history read first last) :
    ReadWindow read (HappensBefore history) first last :=
  ⟨checked.afterFirst.orders, fun i eq => (checked.beforeNext i eq).orders, checked.writeRead, checked.readWrite⟩

def LockedCheckContract : Prop :=
  ∀ (history : History) (writes : List Write) (booked : Fin writes.length) (read : Read writes),
    (writes.get booked).value = 1 →
    LockedFlagRead history read booked booked.val →
    read.source = booked ∧ (writes.get read.source).value = 1

theorem lockedCheckContract_proven : LockedCheckContract := by
  intro history writes booked read busy checked
  exact busyCheckContract_proven history writes booked read busy checked.window

def ClosedBookingContract : Prop :=
  ∀ (history : History) (writes : List Write) (closed : Fin writes.length)
    (last : Nat) (read : Read writes),
    LockedFlagRead history read closed last →
    (∀ i : Fin writes.length, closed.val ≤ i.val → i.val ≤ last → (writes.get i).value = 0) →
    (writes.get read.source).value = 0

theorem closedBookingContract_proven : ClosedBookingContract := by
  intro history writes closed last read checked closedFlags
  obtain ⟨afterClose, beforeReopen⟩ := checked.window.bounds
  exact closedFlags read.source afterClose beforeReopen

inductive BeforeBooking (history : History) (init : Event) : Event → Prop
  | direct : SequencedBefore init booked → BeforeBooking history init booked
  | late : SpinLock.OrderedAccess history init booked → BeforeBooking history init booked

private theorem before_booking (evidence : BeforeBooking history init booked) :
    HappensBefore history init booked := by
  cases evidence with
  | direct before => exact .sequenced before
  | late access => exact access.orders

inductive PublishedThrough (history : History) : Event → Event → Prop
  | own : SequencedBefore init access → PublishedThrough history init access
  | wake {writes : List Write} (wake : SearchWake history writes) :
      BeforeBooking history init (writes.get wake.booked).event →
      SequencedBefore wake.read.event access → PublishedThrough history init access
  | stateLock {booked : Event} :
      BeforeBooking history init booked → SpinLock.OrderedAccess history booked access →
      PublishedThrough history init access
  | ancestor {capture childInit : Event} :
      PublishedThrough history init capture → SequencedBefore capture childInit →
      PublishedThrough history childInit access → PublishedThrough history init access

def InitializationContract : Prop :=
  ∀ (history : History) (init access : Event),
    PublishedThrough history init access → HappensBefore history init access

theorem initializationContract_proven : InitializationContract := by
  intro history init access published
  induction published with
  | own before => exact .sequenced before
  | wake wake booked after =>
    exact .trans (before_booking booked)
      (.trans (wakeContract_proven _ _ wake).2 (.sequenced after))
  | stateLock booked access => exact .trans (before_booking booked) access.orders
  | ancestor _ copied _ parentReady childReady =>
    exact .trans parentReady (.trans (.sequenced copied) childReady)

def TaskAccessContract : Prop :=
  ∀ (history : History) (init access wait clear : Event),
    PublishedThrough history init access → RetainedThrough history access wait →
    SequencedBefore wait clear →
    HappensBefore history init access ∧ HappensBefore history access clear

theorem taskAccessContract_proven : TaskAccessContract := by
  intro history init access wait clear published retained after
  exact ⟨initializationContract_proven history init access published,
    SplitCompletion.retainedAccessContract_proven history access wait clear retained after⟩

end NeuralReversi.Threading.SplitPublication

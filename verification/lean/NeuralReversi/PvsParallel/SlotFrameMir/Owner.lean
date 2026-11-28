module
public import NeuralReversi.PvsParallel.SlotFrameMir.Mask

@[expose] public section

namespace NeuralReversi.PvsParallel.SlotFrameMir

open Threading
open AtomicWord (Event Read Write ReadWindow SequencedBefore)
open CutoffVisibility (History HappensBefore)
open SplitCompletion (Action Activation Retirement)

/-- Events before the physical split invocation, inside its idle loop, and after
    it returns belong to one worker clock. The initial task-slot cells are unknown.
    Chronology of an external booking uses the RMW in its slot critical section. -/
structure OwnerSegment (u : Worker) (b : Activation bMask) (bslot : Slot)
    (helpful aEnd : Event) where
  level : Nat
  old : Nat → Option Slot
  events : List Occurrence
  before : List (Slot × Event)
  during : List (Slot × Event)
  after : List (Slot × Event)
  executed : Run u (level+1) (entry u level old) events during
  pre : ∀ slot e, (slot,e) ∈ before → SequencedBefore e helpful
  post : ∀ slot e, (slot,e) ∈ after → SequencedBefore aEnd e
  complete : ∀ j : Fin b.actions.length, ∀ e, b.actions.get j = .finish u e →
    (bslot,e) ∈ before ++ during ++ after

/-- Slot and worker identities in the segment have distinct thread-local banks.
    The actual pointer write follows its mask set; the clock may have arbitrary
    events between them. -/
structure Posting (hist : History) (raw : MaskEvents ws b) (u : Worker)
    (guard : Event) (stamp : Event) where
  index : Fin ws.length
  maskSet : raw.tag index = some (.book u (ws.get index).event)
  assigned : raw.time (ws.get index).event ≤ raw.time stamp
  guardBefore : HappensBefore hist guard (ws.get index).event

structure PostedWithin (hist : History) (raw : MaskEvents ws b) (u : Worker) (bslot : Slot)
    (guard : Event) (ops : List Occurrence) where
  source : ∀ item ∈ ops, item.op = .book bslot → Posting hist raw u guard item.atTime
  chronology : ∀ left book middle leave right, ops = left ++ book :: (middle ++ leave :: right) →
    raw.time book.atTime < raw.time leave.atTime

/-- If the prior waiter could skip its initial assignment and both slots complete,
    a duplicate B booking precedes an actual B reset. A membership release that
    happened before A, or a saved B work frame that returns after A, is discharged
    using the two retirements, without guessing the source of any reset. -/
theorem crossed_wait_gives_prior_duplicate (hist : History) (a : Activation aMask)
    (b : Activation bMask) (aEmpty : Retirement hist aMask a) (bEmpty : Retirement hist bMask b)
    (helpful : Read bMask) (enroll : Event) (postedA : .book b.owner enroll ∈ a.actions)
    (range : ReadWindow helpful (HappensBefore hist) b.start (b.start.val+b.actions.length))
    (accepted : WorkerFrameMir.canJoin searching failed true deepest (AtomicBitSet.test (bMask.get helpful.source).value a.owner) = true)
    (waiting : ∀ e, Action.finish b.owner e ∈ a.actions → SequencedBefore bEmpty.read.event e)
    (other : a.owner ≠ b.owner) (slot : Nat)
    (segment : OwnerSegment a.owner b (b.owner,slot) helpful.event aEmpty.read.event)
    (raw : MaskEvents bMask b) (physical : PostedWithin hist raw a.owner (b.owner,slot) helpful.event segment.events)
    (causalClock : ∀ x y, HappensBefore hist x y → raw.time x < raw.time y) :
    ∃ first later leave : Fin b.actions.length, ∃ e1 e2 resetAt,
      b.actions.get first = .book a.owner e1 ∧ b.actions.get later = .book a.owner e2 ∧
      b.actions.get leave = .finish a.owner resetAt ∧ first.val < later.val ∧ later.val < leave.val ∧
      raw.time e2 < raw.time bEmpty.read.event := by
  classical
  apply Classical.byContradiction
  intro absent
  apply WorkerFrameMir.completed_crossed_wait hist aMask bMask a b aEmpty bEmpty helpful enroll postedA
    range accepted waiting
  intro j e reset
  have found := segment.complete j e reset
  rcases List.mem_append.mp found with earlier | later
  · rcases List.mem_append.mp earlier with prior | inside
    · exact Or.inl (.sequenced (segment.pre _ e prior))
    · obtain ⟨left,book,mid,leave,right,eq,assigned,when⟩ :=
        segment.executed.before_reset (entry_clean _ _ _ _ (other_owner _ 0 _ other _ (by omega)))
          (other_owner _ _ _ other) inside
      have member : book ∈ segment.events := eq ▸ List.mem_append_right _ (List.mem_cons_self ..)
      let booking := physical.source book member assigned
      have step := physical.chronology left book mid leave right eq
      have target : raw.time (bMask.get booking.index).event < raw.time (b.actions.get j).event := by
        rw [reset]; change _ < raw.time e; rw [← when]; exact Nat.lt_of_le_of_lt booking.assigned step
      obtain ⟨earlier,later,initial,one,two,lt,upto⟩ := raw.earlier_duplicate hist helpful range other accepted
        booking.index _ booking.maskSet booking.guardBefore j target
      have released := SplitCompletion.handoffContract_proven hist bMask b bEmpty j (by rw [reset]; rfl)
      exact (absent ⟨earlier,later,j,initial,_,e,one,two,reset,lt,upto,
        Nat.lt_trans target (causalClock _ _ released)⟩).elim
  · exact Or.inr (segment.post _ e later)

end NeuralReversi.PvsParallel.SlotFrameMir

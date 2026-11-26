import NeuralReversi.PvsParallel.WorkerFrameMir.Reads

namespace NeuralReversi.PvsParallel.WorkerFrameMir

open Threading
open AtomicWord (Read Write Event SequencedBefore ReadWindow InReleaseSequence)
open CutoffVisibility (History HappensBefore)
open SplitCompletion (Worker MaskState Action)
open SplitRebookingMir (Coherence)

structure Section where
  lock : Event
  action : Action

/-- Slot critical sections are ordered by the physical slot mutex. The proposed booking
    is between the two lists. The slot view contains the actions before that booking. -/
structure SlotOrder (history : History) (acts : List Action) (book : Event) where
  prior : List Section
  after : List Section
  completed : acts = prior.map Section.action
  ahead : ∀ s ∈ after, SpinLock.OrderedAccess history book s.lock

structure Bindings (history : History) (root : Frame) (span : Span root ws) (s : MaskState)
    (acts : List Action) (w : Worker) (book : Event) (publishedFailures : List Event)
    (ordering : SlotOrder history acts book) where
  splitCall : ∀ st, Record.localWrite st ∈ span.records → ∀ f, FirstInner st.frames f →
    GuardSample history s acts book f
  ownDetach : ∀ st, Record.localWrite st ∈ span.records → ∀ d, st.frames = [d] → d.kind = .detach →
    ∃ sec ∈ ordering.prior ++ ordering.after, ∃ e, sec.action = .finish w e ∧ SequencedBefore sec.lock st.time
  abandon : ∀ st, Record.localWrite st ∈ span.records → ∀ d parent rest, st.frames = d :: parent :: rest →
    d.kind = .detach → parent.kind = .abandon →
    ∃ stamp : Fin publishedFailures.length, SequencedBefore (publishedFailures.get stamp) st.time

/-- A successful (non-busy, non-failed) recheck cannot source its zero from a worker's
    normally entered A-frame interval. A poisoned cleanup cannot pass the recheck either. -/
theorem no_booking_from_frame (history : History) (s : MaskState) (acts : List Action) (w : Worker)
    (root : Frame) (ws : List Write) (span : Span root ws) (read : Read ws) (book : Event)
    (range : ReadWindow read (HappensBefore history) span.first span.last.val)
    (memory : Coherence history ws read) (acquire : read.acquire = true)
    (sync : ∀ i : Fin ws.length, read.acquire = true → InReleaseSequence ws i read.source →
      history.otherSync (ws.get i).event read.event)
    (beforeBook : SequencedBefore read.event book)
    (birth : Event) (faults : List Event) (failed : Read (failures birth faults))
    (failMemory : Coherence history (failures birth faults) failed) (afterRead : SequencedBefore read.event failed.event)
    (order : SlotOrder history acts book) (mapped : Bindings history root span s acts w book faults order)
    (late : Bool) (opened : (SplitCompletion.run s acts).accepting = true)
    (capacity : SplitRebookingMir.room late (AtomicBitSet.count (SplitCompletion.run s acts).bits) = true) :
    joinCheck (decide ((ws.get read.source).value = 1))
      (decide (((failures birth faults).get failed.source).value = 1)) = false := by
  cases granted : joinCheck (decide ((ws.get read.source).value = 1))
      (decide (((failures birth faults).get failed.source).value = 1)) with
  | false => rfl
  | true =>
    exfalso
    obtain ⟨idle,healthy⟩ := Bool.and_eq_true_iff.mp granted
    obtain ⟨st,member,off,when,published⟩ := span.selected read.source range.bounds (by simpa using idle)
    have releaseSeq : InReleaseSequence ws read.source read.source :=
      ⟨published,Nat.le_refl _,fun _ strictly upper => by omega⟩
    have fromSource : HappensBefore history st.time read.event := when ▸
      (HappensBefore.other (sync read.source acquire releaseSeq))
    have atBook := HappensBefore.trans fromSource (.sequenced beforeBook)
    have wrote : ((top root st.frames).kind,false) ∈ searchingStores := off ▸ st.emitted
    have why := origin st.stack wrote
    generalize eq : st.frames = fs at why
    cases why with
    | nested first =>
      have marked := mapped.splitCall st member _ (eq ▸ first)
      have localOrder := st.stack.before_write (eq ▸ first.mem) st.afterEntry
      have inside := marked.entered (.trans (.sequenced localOrder) atBook)
      exact SplitRebookingMir.open_and_room_excludes_inner s acts late opened capacity inside.sample
    | own d kind =>
      obtain ⟨sec,loc,e,act,prior⟩ := mapped.ownDetach st member d eq kind
      rcases List.mem_append.mp loc with before | after
      · have found : .finish w e ∈ acts := order.completed ▸ (act ▸ List.mem_map_of_mem before)
        exact SplitRebookingMir.detach_cannot_overlap history s acts read memory w opened (.earlier e found)
      · have waiting := (order.ahead sec after).orders
        have future := HappensBefore.trans (.sequenced beforeBook) (waiting.trans (.sequenced prior))
        exact Nat.lt_irrefl _ (memory.readWrite read.source (when ▸ future))
    | unwind d parent kind abandon =>
      obtain ⟨stamp,prior⟩ := mapped.abandon st member d parent _ eq kind abandon
      have caught := failed_after_unwind history birth faults failed stamp
        (.trans (.sequenced prior) (fromSource.trans (.sequenced afterRead))) failMemory
      exact (by simpa using healthy : ¬ ((failures birth faults).get failed.source).value = 1) caught

end NeuralReversi.PvsParallel.WorkerFrameMir

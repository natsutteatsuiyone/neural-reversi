import NeuralReversi.PvsParallel.SplitSchedulerMir.Mutex.Transfer
import NeuralReversi.PvsParallel.SplitSchedulerMir.WorkerFields

namespace NeuralReversi.PvsParallel.SplitSchedulerMir.Startup

open Threading
open AtomicWord (Event SequencedBefore)
open CutoffVisibility (History)

abbrev StateLocks (hist : History) := List (Nat × Mutex.Capture hist)

def stateOps (states : StateLocks hist) : List (Nat × List SpinLock.Operation) :=
  states.map fun (worker,mutex) => (worker,mutex.ops)

/-- At initial booking the owner holds the target thread's state lock. A late joiner
    acquires the slot lock itself. The owning thread can proceed in program order. -/
def ready (slot : List SpinLock.Operation) (states : List (Nat × List SpinLock.Operation)) (assignment entry : Event) : Bool :=
  decide (assignment.worker = entry.worker ∧ assignment.step < entry.step) ||
  (Mutex.carries slot assignment entry ||
    states.any fun (worker, lock) => worker == entry.worker && Mutex.carries lock assignment entry)

theorem published (slot : Mutex.Capture hist) (states : StateLocks hist) {seed assignment entry access : Event}
    (starts : ready slot.ops (stateOps states) assignment entry) (init : SequencedBefore seed assignment)
    (entered : SequencedBefore entry access) : SplitPublication.PublishedThrough hist seed access := by
  rcases Bool.or_eq_true_iff.mp starts with direct | cross
  · have before : assignment.worker = entry.worker ∧ assignment.step < entry.step := of_decide_eq_true direct
    exact .own ⟨(init.1.trans before.1).trans entered.1, Nat.lt_trans init.2 (Nat.lt_trans before.2 entered.2)⟩
  · have linked : Nonempty (SpinLock.OrderedAccess hist assignment entry) := by
      rcases Bool.or_eq_true_iff.mp cross with throughSlot | throughThread
      · exact slot.transfer throughSlot
      · obtain ⟨_, member, joined⟩ := List.any_eq_true.mp throughThread
        obtain ⟨⟨_,mutex⟩, _, rfl⟩ := List.mem_map.mp member
        exact mutex.transfer (Bool.and_eq_true_iff.mp joined).2
    obtain ⟨path⟩ := linked
    exact .stateLock (.direct init) ⟨path.release,path.acquire,path.beforeRelease,path.handoff,
      path.afterAcquire.1.trans entered.1,Nat.lt_trans path.afterAcquire.2 entered.2⟩

end NeuralReversi.PvsParallel.SplitSchedulerMir.Startup

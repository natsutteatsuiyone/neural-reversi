module
public import NeuralReversi.PvsParallel.SplitSchedulerMir.WorkerFields
public import NeuralReversi.PvsParallel.NativeFrames.Returns

@[expose] public section

namespace NeuralReversi.PvsParallel.SplitSchedulerMir.WorkerFields

open NativeFrames (Event Record)

def afterSearch (left reset : Nat) : List Event → Bool
  | .leave dispatch _ .dispatch none none :: tail =>
      decide (left < dispatch ∧ dispatch < reset) && tail.head?.all (fun e => reset < e.step)
  | _ => false

/-- On the worker's own call stack, the split-search and dispatch returns
    are adjacent. A mask reset executes before the next call-stack event. -/
def resetBetween (opening : Nat) (stamps : List Nat) (reset : Nat) : List Event → Bool
  | [] => false
  | e :: rest =>
    (match e with
      | .leave step entry .splitSearch none none => (entry == opening && opening < step) && stamps.all (fun t => t < step) &&
        afterSearch step reset rest
      | _ => false) || resetBetween opening stamps reset rest

def Dispatch.retired (d : Dispatch) (rs : List Record) :=
  d.maskReset.any fun stamp =>
    resetBetween d.opening d.stamps stamp (NativeFrames.Returns.localEvents d.worker rs)

theorem Dispatch.traced_reset (d : Dispatch) (valid : d.valid) (seen : d.retired rs) :
    resetBetween d.opening d.stamps d.reset.step (NativeFrames.Returns.localEvents d.worker rs) := by
  cases stamp : d.maskReset with
  | none => simp [retired, stamp] at seen
  | some t =>
    have reset := d.at_reset valid stamp
    rw [reset]
    simpa [retired, stamp] using seen

theorem return_window (accepted : resetBetween opening stamps reset es) :
    ∃ pre left dispatched first tail, es = pre ++ .leave left opening .splitSearch none none ::
      .leave dispatched first .dispatch none none :: tail ∧
      opening < left ∧ left < dispatched ∧ dispatched < reset ∧
      (∀ t ∈ stamps, t < left) ∧ ∀ e ∈ tail.head?, reset < e.step := by
  induction es with
  | nil => contradiction
  | cons e rest ih =>
    rcases Bool.or_eq_true_iff.mp accepted with hit | later
    · split at hit <;> try contradiction
      rename_i step entry
      obtain ⟨checks, next⟩ := Bool.and_eq_true_iff.mp hit
      obtain ⟨checks, contained⟩ := Bool.and_eq_true_iff.mp checks
      obtain ⟨rfl, start⟩ := (show entry = opening ∧ opening < step by simpa using checks)
      unfold afterSearch at next
      split at next <;> try contradiction
      rename_i dispatched first tail
      obtain ⟨middle, upper⟩ := Bool.and_eq_true_iff.mp next
      exact ⟨[], step, dispatched, first, tail, rfl, start, (of_decide_eq_true middle).1,
        (of_decide_eq_true middle).2,
        fun _ mem => of_decide_eq_true (List.all_eq_true.mp contained _ mem), by
          intro e member
          rw [show tail.head? = some e from member] at upper
          exact of_decide_eq_true upper⟩
    · obtain ⟨pre, l, d, f, tail, eq, bounds⟩ := ih later
      exact ⟨e::pre, l, d, f, tail, by simp [eq], bounds⟩

end NeuralReversi.PvsParallel.SplitSchedulerMir.WorkerFields

module
public import NeuralReversi.PvsParallel.SplitRebookingMir.Fields

@[expose] public section

namespace NeuralReversi.PvsParallel.SplitRebookingMir

open Threading
open AtomicWord (Event Read Write)
open CutoffVisibility (History HappensBefore)
open SplitCompletion (Worker MaskState Action step run)

private theorem a_finish_closes (s : MaskState) (xs : List Action) (mem : Action.finish w e ∈ xs) :
    (run s xs).accepting = false := by
  cases openState : (run s xs).accepting with
  | false => rfl
  | true =>
    induction xs generalizing s with
    | nil => cases mem
    | cons first rest ih =>
      rcases List.mem_cons.mp mem with here | later
      · subst first
        have prior := (open_history (step s (.finish w e)) rest openState).1
        simp [step,Action.release] at prior
      · exact ih (step s first) later openState

/-- The two orders of the outer slot's critical sections. The candidate cannot
    interleave with the worker's false searching store, close, and mask reset. -/
inductive OuterDetach (history : History) (read : Read words) (acts : List Action) (w : Worker) : Prop
  | earlier (atRelease : Event) : Action.finish w atRelease ∈ acts → OuterDetach history read acts w
  | later : HappensBefore history read.event (words.get read.source).event → OuterDetach history read acts w

theorem detach_cannot_overlap (history : History) (s : MaskState) (acts : List Action)
    (read : Read words) (memory : Coherence history words read) (w : Worker) (opened : (run s acts).accepting = true) :
    ¬OuterDetach history read acts w := by
  intro detach
  cases detach with
  | earlier atRelease member =>
    have closed := a_finish_closes s acts member
    rw [opened] at closed; contradiction
  | later future =>
    have impossible := memory.readWrite read.source future
    exact (Nat.lt_irrefl _) impossible

/-- Every candidate is checked under the thread lock. When the original assignment is still
    retained, a false searching value must be related to an inner split or the outer detach.
    The dynamic call stack and the read-from source supply that origin classification. -/
theorem no_duplicate_book (history : History) (s : MaskState) (acts : List Action) (w : Worker)
    (words : List Write) (read : Read words) (memory : Coherence history words read)
    (book : Event) (late : Bool) (opened : (run s acts).accepting = true)
    (capacity : room late (AtomicBitSet.count (run s acts).bits) = true)
    (canJoin : SplitActivationMir.gate late true (decide ((words.get read.source).value = 0)) = true)
    (assignments : SplitActivationMir.Fields) (receipt : Event)
    (origins : assignments.assignment w = some receipt →
      (∃ stamp : Fin words.length, (words.get stamp).value = 1 ∧
        Nonempty (SplitPublication.LockedFlagRead history read stamp stamp.val)) ∨
      FirstNested history s acts book ∨ OuterDetach history read acts w) : assignments.assignment w ≠ some receipt := by
  intro before
  rcases origins before with ⟨stamp,busy,⟨protectedRead⟩⟩ | inner | detach
  · have stillBusy := (SplitPublication.lockedCheckContract_proven history words stamp read busy protectedRead).2
    rw [stillBusy] at canJoin
    cases late <;> simp [SplitActivationMir.gate] at canJoin
  · exact open_and_room_excludes_inner s acts late opened capacity inner.sample
  · exact detach_cannot_overlap history s acts read memory w opened detach

end NeuralReversi.PvsParallel.SplitRebookingMir

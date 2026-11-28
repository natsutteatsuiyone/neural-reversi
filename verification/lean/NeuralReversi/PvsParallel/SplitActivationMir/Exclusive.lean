module
public import NeuralReversi.PvsParallel.SplitActivationMir.Workers

@[expose] public section

namespace NeuralReversi.PvsParallel.SplitActivationMir

open Threading
open AtomicWord (Event Write Read)
open CutoffVisibility (History)
open SplitCompletion (Action Worker MaskState Legal step)

structure Recheck (history : History) (fields : Fields) (worker : Worker) where
  words : List Write
  load : Read words
  late : Bool
  available : Bool
  passed : gate late available (decide ((words.get load.source).value = 0)) = true
  busyWhileAssigned : ∀ receipt, fields.assignment worker = some receipt →
    ∃ stamp : Fin words.length, (words.get stamp).value = 1 ∧
      Nonempty (SplitPublication.LockedFlagRead history load stamp stamp.val)

theorem Recheck.vacant (check : Recheck history fields w) : fields.assignment w = none := by
  have idle : (check.words.get check.load.source).value = 0 := by
    have ok := check.passed
    cases stage : check.late <;> simp only [gate,stage,Bool.false_eq_true,ite_false,ite_true] at ok
    · exact of_decide_eq_true ok
    · exact of_decide_eq_true (Bool.and_eq_true_iff.mp ok).2
  cases eq : fields.assignment w with
  | none => rfl
  | some receipt =>
    obtain ⟨index,busy,⟨order⟩⟩ := check.busyWhileAssigned receipt eq
    have wasBusy := (SplitPublication.lockedCheckContract_proven history check.words index check.load busy order).2
    rw [idle] at wasBusy; contradiction

def Rechecks (history : History) : Fields → List Action → Prop
  | _, [] => True
  | f, .book w e :: xs => Nonempty (Recheck history f w) ∧ Rechecks history (onEnroll f w e) xs
  | f, .finish w _ :: xs => Rechecks history (leave f w) xs

private theorem closed_no_book (m : MaskState) (xs : List Action) (legal : Legal m xs)
    (closed : m.accepting = false) : ∀ w e, Action.book w e ∉ xs := by
  induction xs generalizing m with
  | nil => simp
  | cons x xs ih =>
    obtain ⟨ok,tail⟩ := legal
    cases x with
    | book w e => change m.accepting = true at ok; rw [closed] at ok; contradiction
    | finish w e => intro who when; simpa using ih _ tail (by simp [step,Action.release]) who when

private theorem held_no_book (m : MaskState) (xs : List Action) (legal : Legal m xs)
    (checks : Rechecks history f xs) (held : f.assignment w ≠ none) :
    ∀ e, Action.book w e ∉ xs := by
  induction xs generalizing m f with
  | nil => simp
  | cons x xs ih =>
    obtain ⟨ok,tail⟩ := legal
    cases x with
    | book v e =>
      obtain ⟨⟨check⟩,rest⟩ := checks
      have other : w ≠ v := by intro same; subst v; exact held check.vacant
      intro when mem
      rcases List.mem_cons.mp mem with eq | later
      · injection eq with same _; exact other same
      · apply ih _ tail rest _ when later
        simpa [onEnroll_eq,setAssignment,other] using held
    | finish v e =>
      intro when
      simpa using closed_no_book _ xs tail (by simp [step,Action.release]) w when

def enrolled : Action → Option Worker
  | .book w _ => some w | .finish .. => none

private theorem distinct_books (m : MaskState) (xs : List Action) (legal : Legal m xs)
    (checks : Rechecks history f xs) : (xs.filterMap enrolled).Nodup := by
  induction xs generalizing m f with
  | nil => simp
  | cons x xs ih =>
    cases x with
    | finish v e => exact ih _ legal.2 checks
    | book v e =>
      obtain ⟨_,rest⟩ := checks
      change (v :: xs.filterMap enrolled).Nodup
      refine List.nodup_cons.mpr ⟨?_, ih _ legal.2 rest⟩
      intro mem
      obtain ⟨act,member,eq⟩ := List.mem_filterMap.mp mem
      cases act with
      | finish _ _ => cases eq
      | book w time =>
        have same : w = v := Option.some.inj eq
        subst w
        exact held_no_book _ xs legal.2 rest (by simp [onEnroll_eq,setAssignment]) time member

theorem exclusive_enrollment (owner : Worker) (mark : Event) (early : Startup (onInit owner mark) first f)
    (later : Run f rest) (checks : Rechecks history (onInit owner mark) (first ++ rest)) :
    ((first ++ rest).filterMap enrolled).Nodup ∧ ∀ e, Action.book owner e ∉ first ++ rest := by
  have legal := actions_legal owner mark early later
  exact ⟨distinct_books _ _ legal checks, held_no_book _ _ legal checks (by simp [onInit_eq,begin])⟩

theorem returned_from_activation (history : History) (mask scores : List Write)
    (owner : Worker) (clear mark : Event) (early : Startup (onInit owner mark) first f) (later : Run f rest)
    (mo : ModificationOrder mask (code owner clear mark (first ++ rest)))
    (clock : SplitSchedulerMir.Clock) (log : List (SplitSchedulerMir.Guard mask))
    (source last : Fin scores.length) (run : SplitSchedulerMir.Returned SplitSchedulerMir.idle mask 0 log)
    (openPool : SplitSchedulerMir.Guard.shutdown true ∉ log)
    (memory : SplitSchedulerMir.OwnerMemory history mask scores (activation owner clear mark early later mo)
      clock log source)
    (origin : (scores.get last).event = clock.atStep 2 ∨ ∃ i : Fin (first ++ rest).length,
      ((first ++ rest).get i).release = true ∧
      AtomicWord.SequencedBefore (scores.get last).event ((first ++ rest).get i).event)
    (noReuse : ∀ i : Fin scores.length, i.val = last.val + 1 →
      AtomicWord.SequencedBefore (SplitSchedulerMir.scoreLoad scores source clock log).event (scores.get i).event)
    (pv : Bool) (beta : Int) (original : SplitWindow.State) (reports : List Int)
    (stores : SplitWindow.BestStoreWindow scores last original.best reports) :
    AtomicWord.signed32At scores source = (SplitWindow.scan pv beta original reports).best :=
  (SplitSchedulerMir.returned_scan history mask scores (activation owner clear mark early later mo) clock log
    source last run openPool memory origin noReuse pv beta original reports stores).2

end NeuralReversi.PvsParallel.SplitActivationMir

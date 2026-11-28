module
public import NeuralReversi.PvsParallel.SplitActivationMir.Program
public import NeuralReversi.Threading.SplitCompletion

@[expose] public section

namespace NeuralReversi.PvsParallel.SplitActivationMir

open Threading
open AtomicWord (Event Write)
open SplitCompletion (Worker Action MaskState Allowed Legal initial step)

structure Fields where
  accepting : Bool
  assignment : Worker → Option Event

def setAssignment (fields : Fields) (worker : Worker) (receipt : Option Event) : Fields :=
  { fields with assignment := fun w => if w = worker then receipt else fields.assignment w }

def begin (owner : Worker) (mark : Event) : Fields :=
  { accepting := true, assignment := fun w => if w = owner then some mark else none }

def leave (fields : Fields) (worker : Worker) : Fields :=
  { setAssignment fields worker none with accepting := false }

def applyField (w : Worker) (receipt : Event) (f : Fields) : Op → Fields
  | .assign => setAssignment f w (some receipt)
  | .take => setAssignment f w none
  | .accept => { f with accepting := true }
  | .close => { f with accepting := false }
  | _ => f

def onInit (w : Worker) (receipt : Event) : Fields :=
  initialOps.foldl (applyField w receipt) ⟨false, fun _ => none⟩

def onEnroll (f : Fields) (w : Worker) (receipt : Event) : Fields :=
  enrollOps.foldl (applyField w receipt) f

def onLeave (f : Fields) (w : Worker) (receipt : Event) : Fields :=
  (releaseOps.flatMap (fun op => if op = .detach then detachOps else [op])).foldl (applyField w receipt) f

theorem onInit_eq (w : Worker) (e : Event) : onInit w e = begin w e := rfl

theorem onEnroll_eq (f : Fields) (w : Worker) (e : Event) : onEnroll f w e = setAssignment f w (some e) := rfl

theorem onLeave_eq (f : Fields) (w : Worker) (e : Event) : onLeave f w e = leave f w := rfl

/-- A receipt identifies the assignment captured by this invocation, including the owner's
    initial assignment. Each worker's projected register must agree with its captured Arc
    through nested calls. Such a projection is a condition on an execution of the scheduler. -/
inductive Run : Fields → List Action → Prop
  | done : Run fields []
  | enroll {fields : Fields} (w : Worker) (event : Event) {idle : Bool} :
      gate true fields.accepting idle = true →
      Run (onEnroll fields w event) rest → Run fields (.book w event :: rest)
  | finish {fields : Fields} (w : Worker) (receipt atRelease : Event) :
      fields.assignment w = some receipt →
      Run (onLeave fields w receipt) rest → Run fields (.finish w atRelease :: rest)

structure Agrees (fields : Fields) (m : MaskState) : Prop where
  flag : fields.accepting = m.accepting
  held : ∀ w, (fields.assignment w).isSome = m.has w

private theorem begin_agrees (owner : Worker) (mark : Event) : Agrees (begin owner mark) (initial owner) := by
  refine ⟨rfl, fun w => ?_⟩
  by_cases h : w = owner
  · subst w; simp [begin, initial, MaskState.has, Bitboard.squareMask_bit]
  · change (if w = owner then some mark else none).isSome = (Bitboard.squareMask owner).getLsbD w.val
    rw [Bitboard.squareMask_bit, ite_eq_right h]; simp [Fin.val_inj, h]

private theorem agree_enroll (h : Agrees f m) (w : Worker) (event : Event) :
    Agrees (setAssignment f w (some event)) (step m (.book w event)) := by
  refine ⟨by simpa [setAssignment, step, Action.release] using h.flag, fun j => ?_⟩
  rw [SplitCompletion.book_has]
  by_cases same : j = w
  · simp [setAssignment, same]
  · simp [setAssignment, same, h.held]

private theorem agree_leave (h : Agrees f m) (w : Worker) (event : Event) :
    Agrees (leave f w) (step m (.finish w event)) := by
  refine ⟨by simp [leave, step, Action.release], fun j => ?_⟩
  rw [SplitCompletion.finish_has]
  by_cases same : j = w
  · simp [leave, setAssignment, same]
  · simp [leave, setAssignment, same, h.held]

private theorem run_legal (r : Run f xs) (a : Agrees f m) : Legal m xs := by
  induction r generalizing m with
  | done => trivial
  | @enroll f rest w event idle guard _ ih =>
    constructor
    · change m.accepting = true
      rw [← a.flag]
      exact (Bool.and_eq_true_iff.mp guard).1
    · exact ih (agree_enroll a w event)
  | @finish f rest w id event captured _ ih =>
    constructor
    · change m.has w = true
      rw [← a.held, captured]; rfl
    · exact ih (agree_leave a w event)

inductive Startup : Fields → List Action → Fields → Prop
  | done : Startup f [] f
  | enroll {f : Fields} (w : Worker) (event : Event) {idle : Bool} :
      gate false f.accepting idle = true → Startup (onEnroll f w event) rest after →
      Startup f (.book w event :: rest) after

private theorem startup_legal (r : Startup f xs last) (a : Agrees f m) (flag : f.accepting = true)
    (tailRun : Run last tail) : Legal m (xs ++ tail) := by
  induction r generalizing m with
  | done => exact run_legal tailRun a
  | enroll w event _ _ ih =>
    exact ⟨a.flag.symm.trans flag, ih (agree_enroll a w event) flag tailRun⟩

theorem actions_legal (owner : Worker) (mark : Event) (early : Startup (onInit owner mark) first f)
    (later : Run f rest) : Legal (initial owner) (first ++ rest) :=
  startup_legal early (begin_agrees owner mark) rfl later

/-- The caller supplies a per-invocation register projection and an ordered list of slot-lock
    critical sections. No legal-mask predicate or bit-presence premise is used. -/
theorem retired_bit_was_set {owner : Worker} {mark : Event}
    (early : Startup (onInit owner mark) first f) (later : Run f rest)
    (pos : first ++ rest = past ++ .finish w event :: after) :
    ((SplitCompletion.run (initial owner) past).bits.getLsbD w.val) = true := by
  have legal := actions_legal owner mark early later
  rw [pos] at legal
  exact (SplitCompletion.legal_append _ _ _ legal).2.1

end NeuralReversi.PvsParallel.SplitActivationMir

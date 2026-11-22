import NeuralReversi.Threading.CutoffVisibility

namespace NeuralReversi.Threading.SpinLock

open AtomicWord (Event SequencedBefore Write Read InReleaseSequence)
open CutoffVisibility (History HappensBefore)

inductive Operation where
  | acquire (event : Event)
  | release (event : Event)
  deriving DecidableEq

inductive Legal : Option Event → List Operation → Prop
  | stop : Legal holder []
  | acquired : Legal (some event) rest → Legal none (.acquire event :: rest)
  | released : SequencedBefore held event → Legal none rest → Legal (some held) (.release event :: rest)

private theorem suffix_legal (left right : List Operation) (legal : Legal holder (left ++ right)) :
    ∃ current, Legal current right := by
  induction left generalizing holder with
  | nil => exact ⟨holder, legal⟩
  | cons op left ih =>
    cases legal with
    | acquired remaining => exact ih remaining
    | released _ remaining => exact ih remaining

structure Modifications (history : History) where
  operations : List Operation
  writes : List Write
  length : writes.length = operations.length + 1
  initial : (writes.get ⟨0, by omega⟩).value = 0
  encode : ∀ i : Fin operations.length,
    let write := writes.get ⟨i.val + 1, by have := i.isLt; omega⟩
    match operations.get i with
    | .acquire event => write = ⟨event, 1, false, true⟩
    | .release event => write = ⟨event, 0, true, false⟩
  casRead : ∀ i : Fin operations.length, ∀ event,
    operations.get i = .acquire event → ∃ read : Read writes,
      read.event = event ∧ read.acquire = true ∧ read.source.val = i.val ∧
        (writes.get read.source).value = 0
  sync : ∀ (head : Fin writes.length) (read : Read writes),
    (∃ i : Fin operations.length, operations.get i = .acquire read.event ∧
      read.source.val = i.val) →
    read.acquire = true → InReleaseSequence writes head read.source →
    history.otherSync (writes.get head).event read.event

structure Trace (history : History) extends Modifications history where
  legal : Legal none operations

def ExclusiveContract : Prop :=
  ∀ (history : History) (trace : Modifications history) (a b : Event)
    (left right : List Operation),
    trace.operations ≠ left ++ (.acquire a :: .acquire b :: right)

theorem exclusiveContract_proven : ExclusiveContract := by
  intro history trace a b left right twice
  let i : Fin trace.operations.length := ⟨left.length, by rw [twice]; simp⟩
  let j : Fin trace.operations.length := ⟨left.length + 1, by rw [twice]; simp⟩
  have first : trace.operations.get i = .acquire a := by
    change trace.operations[left.length]'i.isLt = _
    simp [twice, List.getElem_append_right]
  have second : trace.operations.get j = .acquire b := by
    change trace.operations[left.length + 1]'j.isLt = _
    simp [twice, List.getElem_append_right]
  have held := trace.encode i
  simp only [first] at held
  obtain ⟨read, _, _, previous, free⟩ := trace.casRead j b second
  have same : (⟨i.val + 1, by have := trace.length; have := i.isLt; omega⟩ :
      Fin trace.writes.length) = read.source := Fin.ext previous.symm
  rw [← same, held] at free
  cases free

private theorem after_unlock (trace : Trace history) (left right : List Operation) (u a : Event)
    (adjacent : trace.operations = left ++ (.release u :: .acquire a :: right)) :
    HappensBefore history u a := by
  let i : Fin trace.operations.length := ⟨left.length, by rw [adjacent]; simp⟩
  let j : Fin trace.operations.length := ⟨left.length + 1, by rw [adjacent]; simp⟩
  have unlockAt : trace.operations.get i = .release u := by
    change trace.operations[left.length]'i.isLt = _
    simp [adjacent, List.getElem_append_right]
  have lockAt : trace.operations.get j = .acquire a := by
    change trace.operations[left.length + 1]'j.isLt = _
    simp [adjacent, List.getElem_append_right]
  have unlock := trace.encode i
  simp only [unlockAt] at unlock
  obtain ⟨read, event, acquired, previous, _⟩ := trace.casRead j a lockAt
  have head : (⟨i.val + 1, by have := trace.length; have := i.isLt; omega⟩ :
      Fin trace.writes.length) = read.source := by
    apply Fin.ext
    exact previous.symm
  have relaxedWord := unlock
  rw [head] at relaxedWord
  have same : InReleaseSequence trace.writes read.source read.source :=
    ⟨by rw [relaxedWord], Nat.le_refl _, fun k after before => by omega⟩
  have edge := trace.sync read.source read ⟨j, event.symm ▸ lockAt, previous⟩ acquired same
  rw [relaxedWord, event] at edge
  exact .other edge

private theorem lock_chain (trace : Trace history) (left middle right : List Operation) (u a : Event)
    (split : trace.operations = left ++ (.release u :: (middle ++ .acquire a :: right)))
    (valid : Legal none (middle ++ .acquire a :: right)) : HappensBefore history u a := by
  match middle with
  | [] => exact after_unlock trace left right u a split
  | .acquire entering :: rest =>
    cases valid with
    | acquired held =>
      cases rest with
      | nil => cases held
      | cons next remaining =>
        cases held with
        | released inside tailLegal =>
          rename_i leaving
          have first := after_unlock trace left (.release leaving :: remaining ++ .acquire a :: right)
            u entering (by simpa only [List.cons_append] using split)
          have later := lock_chain trace (left ++ [.release u, .acquire entering]) remaining right
            leaving a (by simpa only [List.append_assoc, List.cons_append, List.nil_append] using split) tailLegal
          exact .trans first (.trans (.sequenced inside) later)
  | .release _ :: _ => cases valid
termination_by middle.length

def HandoffContract : Prop :=
  ∀ (history : History) (trace : Trace history) (u a : Event)
    (left middle right : List Operation),
    trace.operations = left ++ (.release u :: (middle ++ .acquire a :: right)) →
    HappensBefore history u a

theorem handoffContract_proven : HandoffContract := by
  intro history trace u a left middle right split
  have full := split ▸ trace.legal
  obtain ⟨_, after⟩ := suffix_legal left _ full
  cases after with
  | released _ tail => exact lock_chain trace left middle right u a split tail

structure Handoff (history : History) (u a : Event) where
  trace : Trace history
  left : List Operation
  middle : List Operation
  right : List Operation
  ordered : trace.operations = left ++ (.release u :: (middle ++ .acquire a :: right))

theorem Handoff.orders (handoff : Handoff history u a) : HappensBefore history u a :=
  handoffContract_proven history handoff.trace u a handoff.left handoff.middle handoff.right handoff.ordered

structure OrderedAccess (history : History) (first later : Event) where
  release : Event
  acquire : Event
  beforeRelease : SequencedBefore first release
  handoff : Handoff history release acquire
  afterAcquire : SequencedBefore acquire later

theorem OrderedAccess.orders (ordered : OrderedAccess history first later) :
    HappensBefore history first later :=
  .trans (.sequenced ordered.beforeRelease) (.trans ordered.handoff.orders (.sequenced ordered.afterAcquire))

end NeuralReversi.Threading.SpinLock

module
public import NeuralReversi.PvsParallel.SlotFrameMir.Program

@[expose] public section

namespace NeuralReversi.PvsParallel.SlotFrameMir

open Threading.AtomicWord (Event SequencedBefore)

/-- Only this worker can publish or clear tasks in its own bank. The unobserved
    contents of every slot on entry may be arbitrary. Materialize a saved parent in
    its actual bank cell, so popping a nested call reads the former publication. -/
structure State where
  active : Option Slot
  capturedParent : Option Slot
  taskParents : Nat → Option Slot
  nested : Nat
  work : List (Option Slot)
  taken : Option Slot

inductive Op
  | book (s : Slot) | captureParent | enter | exit | captureWork | take | finishWork | finishUnwind
  deriving DecidableEq

structure Occurrence where
  op : Op
  atTime : Event

def reset (st : State) : Op → Option Slot
  | .finishWork => workerReset (st.work.headD none) st.taken
  | .finishUnwind => unwindReset none st.taken
  | _ => none

/-- The two local stack boundaries are Rust call/return. An invocation that was
    already suspended outside the owning split cannot execute its epilogue inside it.
    Other threads may assign this worker between any two projected operations. -/
def advance (u : Worker) (base : Nat) (st : State) : Op → Option State
  | .book s => some {st with active := installed s}
  | .captureParent => some {st with capturedParent := savedParent st.active}
  | .enter =>
    let i := base + st.nested
    some {st with
      taskParents := fun j => if j = i then st.capturedParent else st.taskParents j
      active := installed (ownedSlot u i)
      nested := st.nested + 1}
  | .exit =>
    match st.nested with
    | 0 => none
    | n+1 =>
      let i := base + n
      some {st with
        active := restored (st.taskParents i)
        nested := n
        taskParents := fun j => if j = i then none else st.taskParents j}
  | .captureWork => some {st with work := st.active :: st.work}
  | .take => some {st with active := none, taken := st.active}
  | .finishWork => match st.work with
    | [] => none
    | _ :: rest => some {st with work := rest}
  | .finishUnwind => some st

inductive Run (u : Worker) (base : Nat) : State → List Occurrence → List (Slot × Event) → Prop
  | done : Run u base s [] []
  | next (ok : advance u base s item.op = some t) (rest : Run u base t items resets) :
      Run u base s (item :: items) (((reset s item.op).map (fun slot => (slot,item.atTime))).toList ++ resets)

def Clean (b : Slot) (base : Nat) (s : State) : Prop :=
  s.active ≠ some b ∧ s.capturedParent ≠ some b ∧ s.taken ≠ some b ∧
  (∀ q ∈ s.work, q ≠ some b) ∧
  ∀ j, j < s.nested → s.taskParents (base + j) ≠ some b

def DisjointBank (u : Worker) (base : Nat) (b : Slot) : Prop :=
  ∀ i, base ≤ i → ownedSlot u i ≠ b

theorem other_owner (u : Worker) (base : Nat) (b : Slot) (other : u ≠ b.1) : DisjointBank u base b := by
  intro i _ eq; exact other (congrArg Prod.fst eq)

theorem lower_slot (u : Worker) (base : Nat) (b : Slot) (lower : b.2 < base) : DisjointBank u base b := by
  intro i le eq
  have equal : i = b.2 := congrArg Prod.snd eq
  omega

theorem Clean.advance {s t : State} (c : Clean b base s) (disjoint : DisjointBank u base b)
    (op : Op) (different : op ≠ .book b) (next : advance u base s op = some t) :
    Clean b base t ∧ reset s op ≠ some b := by
  obtain ⟨active,parent,taken,work,heap⟩ := c
  cases op with
  | book v =>
    have ne : v ≠ b := fun eq => different (eq ▸ rfl)
    cases next
    exact ⟨⟨by simpa [installed] using ne,parent,taken,work,heap⟩,by simp [reset]⟩
  | captureParent => cases next; exact ⟨⟨active,active,taken,work,heap⟩,by simp [reset]⟩
  | enter =>
    cases next
    refine ⟨⟨?_,parent,taken,work,?_⟩,by simp [reset]⟩
    · intro eq
      exact disjoint _ (by omega) (Option.some.inj eq)
    · intro j bound
      change (if base + j = base + s.nested then s.capturedParent else s.taskParents (base + j)) ≠ some b
      split <;> rename_i branch
      · exact parent
      · apply heap; change j < s.nested + 1 at bound; omega
  | exit =>
    cases h : s.nested with
    | zero => simp [SlotFrameMir.advance,h] at next
    | succ n =>
      simp [SlotFrameMir.advance,h] at next
      subst t
      refine ⟨⟨heap n (by omega),parent,taken,work,?_⟩,by simp [reset]⟩
      intro j lj
      change j < n at lj
      change (if base + j = base + n then none else _) ≠ some b
      rw [ite_eq_right (by omega)]
      exact heap j (by omega)
  | captureWork =>
    cases next
    exact ⟨⟨active,parent,taken,fun q mem => (List.mem_cons.mp mem).elim (fun eq => eq ▸ active) (work q),heap⟩,
      by simp [reset]⟩
  | take => cases next; exact ⟨⟨by simp,parent,active,work,heap⟩,by simp [reset]⟩
  | finishWork =>
    cases eq : s.work with
    | nil => simp [SlotFrameMir.advance,eq] at next
    | cons q qs =>
      simp [SlotFrameMir.advance,eq] at next
      subst t
      have kept : ∀ x ∈ q :: qs, x ≠ some b := eq ▸ work
      exact ⟨⟨active,parent,taken,fun x h => kept x (List.mem_cons_of_mem _ h),heap⟩,
        by simpa [reset,workerReset,eq] using kept q (List.mem_cons_self ..)⟩
  | finishUnwind => cases next; exact ⟨⟨active,parent,taken,work,heap⟩,taken⟩

theorem Run.no_foreign_reset (run : Run u base s events resets) (clean : Clean b base s)
    (other : DisjointBank u base b) (unbooked : ∀ item ∈ events, item.op ≠ .book b) :
    ∀ e, (b,e) ∉ resets := by
  induction run with
  | done => simp
  | @next s t items resets item next _ ih =>
    obtain ⟨safe,foreign⟩ := clean.advance other item.op (unbooked item (List.mem_cons_self ..)) next
    intro e mem
    rcases List.mem_append.mp mem with here | later
    · cases eq : reset s item.op with
      | none => simp [eq] at here
      | some slot =>
        have same : b = slot := congrArg Prod.fst (by simpa [eq] using here : (b,e) = (slot,item.atTime))
        exact foreign (same ▸ eq)
    · exact ih safe (fun i h => unbooked i (List.mem_cons_of_mem _ h)) e later

private theorem Run.stamped (run : Run u base s events resets) (found : (b,e) ∈ resets) :
    ∃ pre item post, events = pre ++ item :: post ∧ item.atTime = e := by
  induction run with
  | done => cases found
  | @next s t xs rs item _ _ ih =>
    rcases List.mem_append.mp found with here | later
    · cases eq : reset s item.op with
      | none => simp [eq] at here
      | some slot =>
        have same : (b,e) = (slot,item.atTime) := by simpa [eq] using here
        exact ⟨[],item,xs,rfl,(congrArg Prod.snd same).symm⟩
    · obtain ⟨left,atE,right,loc,when⟩ := ih later
      exact ⟨item::left,atE,right,by simp [loc],when⟩

/-- A reset in this scope has an earlier booking of the same physical foreign
    slot. The former worker invocation may have been suspended for arbitrarily
    many nested calls and the outer parent stored before entry may be that slot. -/
theorem Run.before_reset (run : Run u base s events resets) (c : Clean b base s)
    (other : DisjointBank u base b) (found : (b,e) ∈ resets) :
    ∃ left book middle leave right, events = left ++ book :: (middle ++ leave :: right) ∧
      book.op = .book b ∧ leave.atTime = e := by
  induction run with
  | done => cases found
  | @next s t xs rs item next tail ih =>
    by_cases booked : item.op = .book b
    · have later : (b,e) ∈ rs := by simpa [booked,reset] using found
      obtain ⟨mid,leave,end_,eq,when⟩ := tail.stamped later
      exact ⟨[],item,mid,leave,end_,by simp [eq],booked,when⟩
    · obtain ⟨safe,diff⟩ := c.advance other item.op booked next
      rcases List.mem_append.mp found with here | later
      · cases eq : reset s item.op with
        | none => simp [eq] at here
        | some slot =>
          have same : (b,e) = (slot,item.atTime) := by simpa [eq] using here
          have val : b = slot := congrArg Prod.fst same
          exact (diff (val ▸ eq)).elim
      · obtain ⟨left,book,mid,leave,right,loc,posted,when⟩ := ih safe later
        exact ⟨item::left,book,mid,leave,right,by simp [loc],posted,when⟩

def entry (u : Worker) (level : Nat) (old : Nat → Option Slot) : State :=
  ⟨installed (ownedSlot u level), none, old, 0, [], none⟩

theorem entry_clean (u : Worker) (level : Nat) (old : Nat → Option Slot) (b : Slot) (other : ownedSlot u level ≠ b) :
    Clean b (level+1) (entry u level old) := by
  refine ⟨?_,by simp [entry],by simp [entry],by simp [entry],by simp [entry]⟩
  intro eq
  exact other (Option.some.inj eq)

theorem suspended_slot_not_reset (run : Run u (level+1) (entry u level old) events resets)
    (apart : DisjointBank u level b) (unbooked : ∀ item ∈ events, item.op ≠ .book b) :
    ∀ e, (b,e) ∉ resets :=
  run.no_foreign_reset (entry_clean _ _ _ _ (apart level (by omega)))
    (fun i hi => apart i (by omega)) unbooked

end NeuralReversi.PvsParallel.SlotFrameMir

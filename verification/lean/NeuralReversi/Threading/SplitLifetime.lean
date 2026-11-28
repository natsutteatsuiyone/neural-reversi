module
public import NeuralReversi.Threading.SplitCompletion

@[expose] public section

namespace NeuralReversi.Threading.SplitLifetime

open SplitCompletion (Worker MaskState Action)

structure Slot where
  owner : Worker
  index : Fin 8
  deriving DecidableEq

structure Task where
  slot : Slot
  parent : Option Nat
  deriving DecidableEq

structure Frame where
  task : Task
  mask : MaskState

structure State where
  next : Nat
  active : Nat → Option Frame

def empty : State :=
  ⟨0, fun _ => none⟩

def Held (state : State) (id : Nat) (worker : Worker) : Prop :=
  ∃ frame, state.active id = some frame ∧ frame.mask.has worker = true

def ParentHeld (state : State) (task : Task) : Prop :=
  ∀ parent, task.parent = some parent → Held state parent task.slot.owner

def Free (state : State) (slot : Slot) : Prop :=
  ∀ id frame, state.active id = some frame → frame.task.slot ≠ slot

structure Valid (state : State) : Prop where
  allocated : ∀ id frame, state.active id = some frame → id < state.next
  uniqueSlot : ∀ id other frame second, state.active id = some frame →
    state.active other = some second → frame.task.slot = second.task.slot → id = other
  ancestry : ∀ id frame parent, state.active id = some frame → frame.task.parent = some parent →
    parent < id ∧ Held state parent frame.task.slot.owner

def State.store (state : State) (id : Nat) (frame : Option Frame) : State :=
  { state with active := fun key => if key = id then frame else state.active key }

def State.create (state : State) (task : Task) : State :=
  { (state.store state.next (some ⟨task, SplitCompletion.initial task.slot.owner⟩)) with
    next := state.next + 1 }

def Frame.apply (frame : Frame) (action : Action) : Frame :=
  { frame with mask := SplitCompletion.step frame.mask action }

def MayLeave (state : State) (id : Nat) (action : Action) : Prop :=
  match action with
  | .book .. => True
  | .finish worker _ =>
    ∀ child frame, state.active child = some frame → frame.task.parent = some id →
      frame.task.slot.owner ≠ worker

inductive Change where
  | create (task : Task)
  | update (id : Nat) (action : Action)
  | retire (id : Nat)

inductive Step : State → Change → State → Prop
  | create {state : State} {task : Task} : Free state task.slot → ParentHeld state task →
      Step state (.create task) (state.create task)
  | update {state : State} {id : Nat} {frame : Frame} {action : Action} : state.active id = some frame → SplitCompletion.Allowed frame.mask action →
      MayLeave state id action →
      Step state (.update id action) (state.store id (some (frame.apply action)))
  | retire {state : State} {id : Nat} {frame : Frame} : state.active id = some frame → frame.mask.bits = 0 →
      Step state (.retire id) (state.store id none)

inductive Run : State → List Change → State → Prop
  | done : Run state [] state
  | step : Step state action next → Run next rest final → Run state (action :: rest) final

theorem empty_valid : Valid empty := by
  constructor <;> simp [empty]

private theorem create_valid (state : State) (task : Task) (valid : Valid state)
    (free : Free state task.slot) (parentHeld : ParentHeld state task) :
    Valid (state.create task) := by
  constructor
  · intro id frame stored
    by_cases fresh : id = state.next
    · simp [State.create, fresh]
    · have old : state.active id = some frame := by
        simpa [State.create, State.store, fresh] using stored
      exact Nat.lt_succ_of_lt (valid.allocated id frame old)
  · intro id other frame second firstAt secondAt sameSlot
    by_cases fresh : id = state.next <;> by_cases otherFresh : other = state.next
    · exact fresh.trans otherFresh.symm
    · have eq : (⟨task, SplitCompletion.initial task.slot.owner⟩ : Frame) = frame :=
        Option.some.inj (by simpa [State.create, State.store, fresh] using firstAt)
      have old : state.active other = some second := by
        simpa [State.create, State.store, otherFresh] using secondAt
      exact (free other second old (by simpa [← eq] using sameSlot.symm)).elim
    · have eq : (⟨task, SplitCompletion.initial task.slot.owner⟩ : Frame) = second :=
        Option.some.inj (by simpa [State.create, State.store, otherFresh] using secondAt)
      have old : state.active id = some frame := by
        simpa [State.create, State.store, fresh] using firstAt
      exact (free id frame old (by simpa [← eq] using sameSlot)).elim
    · exact valid.uniqueSlot _ _ _ _
        (by simpa [State.create, State.store, fresh] using firstAt)
        (by simpa [State.create, State.store, otherFresh] using secondAt) sameSlot
  · intro id frame parent stored link
    by_cases fresh : id = state.next
    · have eq : (⟨task, SplitCompletion.initial task.slot.owner⟩ : Frame) = frame :=
        Option.some.inj (by simpa [State.create, State.store, fresh] using stored)
      subst frame
      obtain ⟨held, live, bit⟩ := parentHeld parent link
      have older := valid.allocated parent held live
      exact ⟨fresh ▸ older, held, by simpa [State.create, State.store, Nat.ne_of_lt older] using live, bit⟩
    · have old : state.active id = some frame := by
        simpa [State.create, State.store, fresh] using stored
      obtain ⟨before, held, live, bit⟩ := valid.ancestry id frame parent old link
      have older := valid.allocated parent held live
      exact ⟨before, held, by simpa [State.create, State.store, Nat.ne_of_lt older] using live, bit⟩

private theorem update_origin {state : State} {id : Nat} {before : Frame} {action : Action}
    (present : state.active id = some before)
    (stored : (state.store id (some (before.apply action))).active key = some frame) :
    ∃ old, state.active key = some old ∧ old.task = frame.task := by
  by_cases same : key = id
  · have eq : before.apply action = frame := Option.some.inj (by simpa [State.store, same] using stored)
    exact ⟨before, same.symm ▸ present, by cases eq; rfl⟩
  · exact ⟨frame, by simpa [State.store, same] using stored, rfl⟩

private theorem held_after_update (before : Frame) (action : Action)
    (worker : Worker) (ownerBit : before.mask.has worker = true)
    (keep : ∀ finished event, action = .finish finished event → worker ≠ finished) :
    (before.apply action).mask.has worker = true := by
  cases action with
  | book booked event =>
    simp [Frame.apply, SplitCompletion.book_has, ownerBit]
  | finish finished event =>
    simp [Frame.apply, SplitCompletion.finish_has, ownerBit, keep finished event rfl]

private theorem update_valid (state : State) (id : Nat) (before : Frame) (action : Action)
    (valid : Valid state) (present : state.active id = some before) (mayLeave : MayLeave state id action) :
    Valid (state.store id (some (before.apply action))) := by
  constructor
  · intro key frame stored
    obtain ⟨old, live, _⟩ := update_origin present stored
    exact valid.allocated key old live
  · intro key other frame second stored secondAt sameSlot
    obtain ⟨old, live, task⟩ := update_origin present stored
    obtain ⟨oldSecond, liveSecond, otherTask⟩ := update_origin present secondAt
    exact valid.uniqueSlot key other old oldSecond live liveSecond (by rw [task, otherTask, sameSlot])
  · intro key frame parent stored link
    obtain ⟨older, live, task⟩ := update_origin present stored
    obtain ⟨prior, parentFrame, parentLive, ownerBit⟩ := valid.ancestry key older parent live (task ▸ link)
    refine ⟨prior, ?_⟩
    by_cases parentSame : parent = id
    · subst parent
      have same := Option.some.inj (present.symm.trans parentLive)
      subst parentFrame
      refine ⟨before.apply action, by simp [State.store], ?_⟩
      rw [← task]
      exact held_after_update before action _ ownerBit (by
        intro finished event eq
        subst action
        exact mayLeave key older live (task ▸ link))
    · exact ⟨parentFrame, by simpa [State.store, parentSame] using parentLive, task ▸ ownerBit⟩

private theorem retired_origin {state : State} {id key : Nat}
    (stored : (state.store id none).active key = some frame) :
    key ≠ id ∧ state.active key = some frame := by
  by_cases same : key = id
  · simp [State.store, same] at stored
  · exact ⟨same, by simpa [State.store, same] using stored⟩

private theorem retire_valid (state : State) (id : Nat) (before : Frame)
    (valid : Valid state) (present : state.active id = some before) (clear : before.mask.bits = 0) :
    Valid (state.store id none) := by
  constructor
  · intro key frame stored
    exact valid.allocated key frame (retired_origin stored).2
  · intro key other frame second stored secondAt sameSlot
    exact valid.uniqueSlot key other frame second (retired_origin stored).2 (retired_origin secondAt).2 sameSlot
  · intro key frame parent stored link
    obtain ⟨prior, held, parentLive, bit⟩ := valid.ancestry key frame parent (retired_origin stored).2 link
    have distinct : parent ≠ id := by
      intro same
      subst parent
      have eq := Option.some.inj (parentLive.symm.trans present)
      subst held
      simp [MaskState.has, clear] at bit
    exact ⟨prior, held, by simpa [State.store, distinct] using parentLive, bit⟩

theorem step_valid (valid : Valid state) (step : Step state change next) : Valid next := by
  cases step with
  | create free parent => exact create_valid _ _ valid free parent
  | update live allowed mayLeave => exact update_valid _ _ _ _ valid live mayLeave
  | retire live clear => exact retire_valid _ _ _ valid live clear

theorem run_valid (valid : Valid state) (run : Run state changes final) : Valid final := by
  induction run with
  | done => exact valid
  | step first rest ih => exact ih (step_valid valid first)

def LifetimeContract : Prop :=
  ∀ (changes : List Change) (state : State), Run empty changes state → Valid state

theorem lifetimeContract_proven : LifetimeContract :=
  fun _ _ run => run_valid empty_valid run

inductive Path (state : State) : Nat → List (Nat × Task) → Prop
  | root {id : Nat} : state.active id = some frame → frame.task.parent = none →
      Path state id [(id, frame.task)]
  | parent {id : Nat} : state.active id = some frame → frame.task.parent = some earlier →
      Path state earlier tail → Path state id ((id, frame.task) :: tail)

private theorem path_exists (valid : Valid state) (key : Nat) (live : state.active key = some frame) :
    ∃ tasks, Path state key tasks := by
  cases link : frame.task.parent with
  | none => exact ⟨_, .root live link⟩
  | some parent =>
    obtain ⟨rank, held, parentLive, _⟩ := valid.ancestry key frame parent live link
    obtain ⟨tasks, path⟩ := path_exists valid parent parentLive
    exact ⟨_, .parent live link path⟩
termination_by key

private theorem path_entry_live (path : Path state key tasks) (member : entry ∈ tasks) :
    ∃ frame, state.active entry.1 = some frame ∧ frame.task = entry.2 := by
  induction path with
  | root atRoot _ =>
    have first := List.mem_singleton.mp member
    cases first
    exact ⟨_, atRoot, rfl⟩
  | parent atHere _ _ ih =>
    rcases List.mem_cons.mp member with first | rest
    · cases first
      exact ⟨_, atHere, rfl⟩
    · exact ih rest

private theorem path_rank (valid : Valid state) (path : Path state key tasks) (member : entry ∈ tasks) :
    entry.1 ≤ key := by
  induction path with
  | root _ _ =>
    rw [List.mem_singleton.mp member]
    exact Nat.le_refl _
  | parent atHere link _ ih =>
    rcases List.mem_cons.mp member with first | rest
    · rw [first]
      exact Nat.le_refl _
    · exact Nat.le_of_lt (Nat.lt_of_le_of_lt (ih rest) (valid.ancestry _ _ _ atHere link).1)

private theorem path_distinct (valid : Valid state) (path : Path state key tasks) :
    tasks.Pairwise (fun current earlier => earlier.1 < current.1 ∧ current.2.slot ≠ earlier.2.slot) := by
  induction path with
  | root atRoot root => simp
  | parent here link tail ih =>
    refine List.pairwise_cons.mpr ⟨?_, ih⟩
    intro entry member
    have prior := Nat.lt_of_le_of_lt (path_rank valid tail member) (valid.ancestry _ _ _ here link).1
    refine ⟨prior, ?_⟩
    intro sameSlot
    obtain ⟨older, live, sameTask⟩ := path_entry_live tail member
    have alias := valid.uniqueSlot _ _ _ _ here live (by rw [sameTask]; exact sameSlot)
    omega

def AncestorContract : Prop :=
  ∀ (state : State) (key : Nat) (worker : Worker), Valid state → Held state key worker →
    ∃ tasks, Path state key tasks ∧
      tasks.Pairwise (fun current earlier => earlier.1 < current.1 ∧ current.2.slot ≠ earlier.2.slot)

theorem ancestorContract_proven : AncestorContract := by
  intro state key worker valid held
  obtain ⟨frame, live, _⟩ := held
  obtain ⟨tasks, path⟩ := path_exists valid key live
  exact ⟨tasks, path, path_distinct valid path⟩

def Leaves (key : Nat) (worker : Worker) : Change → Prop
  | .update id (.finish finished _) => id = key ∧ finished = worker
  | _ => False

private theorem step_held {key : Nat} {worker : Worker}
    (valid : Valid state) (change : Step state action next) (held : Held state key worker)
    (keep : ¬ Leaves key worker action) : Held next key worker := by
  obtain ⟨before, live, bit⟩ := held
  cases change with
  | create free parent =>
    have older := valid.allocated key before live
    exact ⟨before, by simpa [State.create, State.store, Nat.ne_of_lt older] using live, bit⟩
  | @update affected frame edited present allowed mayLeave =>
    by_cases here : key = affected
    · subst key
      have same := Option.some.inj (live.symm.trans present)
      subst before
      refine ⟨frame.apply edited, by simp [State.store], held_after_update frame edited worker bit ?_⟩
      intro finished event eq alias
      subst edited
      exact keep ⟨rfl, alias.symm⟩
    · exact ⟨before, by simpa [State.store, here] using live, bit⟩
  | @retire affected frame present empty =>
    have distinct : key ≠ affected := by
      intro here
      subst key
      have same := Option.some.inj (live.symm.trans present)
      subst before
      simp [MaskState.has, empty] at bit
    exact ⟨before, by simpa [State.store, distinct] using live, bit⟩

private theorem same_task {key : Nat} {original later : Frame}
    (valid : Valid state) (step : Step state change next)
    (before : state.active key = some original) (after : next.active key = some later) :
    original.task = later.task := by
  cases step with
  | create free parent =>
    have older := valid.allocated key original before
    have still : state.active key = some later := by
      simpa [State.create, State.store, Nat.ne_of_lt older] using after
    exact congrArg Frame.task (Option.some.inj (before.symm.trans still))
  | update live allowed mayLeave =>
    obtain ⟨was, present, unchanged⟩ := update_origin live after
    exact (congrArg Frame.task (Option.some.inj (before.symm.trans present))).trans unchanged
  | retire live clear =>
    exact congrArg Frame.task (Option.some.inj (before.symm.trans (retired_origin after).2))

private theorem same_path (valid : Valid state) (changed : Step state action next)
    (afterValid : Valid next) (path : Path state key tasks) :
    (∃ frame, next.active key = some frame) → Path next key tasks := by
  induction path with
  | root old link =>
    rintro ⟨frame, live⟩
    have same := same_task valid changed old live
    exact same.symm ▸ Path.root live (same ▸ link)
  | parent old link _ ih =>
    rintro ⟨frame, live⟩
    have same := same_task valid changed old live
    have nextLink : frame.task.parent = _ := same ▸ link
    obtain ⟨_, outer, outerLive, _⟩ := afterValid.ancestry _ _ _ live nextLink
    have followed := Path.parent live nextLink (ih ⟨outer, outerLive⟩)
    exact same.symm ▸ followed

def ScanRetentionContract : Prop :=
  ∀ (state final : State) (changes : List Change) (key : Nat) (worker : Worker) (tasks : List (Nat × Task)),
    Valid state → Held state key worker → Path state key tasks →
    Run state changes final → (∀ change ∈ changes, ¬ Leaves key worker change) →
    Held final key worker ∧ Path final key tasks

theorem scanRetentionContract_proven : ScanRetentionContract := by
  intro state final changes key worker tasks valid held path run retained
  induction run with
  | done => exact ⟨held, path⟩
  | step first rest ih =>
    have ready := step_valid valid first
    have still := step_held valid first held (retained _ (by simp))
    have pathNext := same_path valid first ready path ⟨_, still.choose_spec.1⟩
    exact ih ready still pathNext (fun action mem => retained action (List.mem_cons_of_mem _ mem))

def ParentRetentionContract : Prop :=
  ∀ (state : State) (child parent : Nat) (frame : Frame), Valid state →
    state.active child = some frame → frame.task.parent = some parent →
    (∀ event after, ¬ Step state (.update parent (.finish frame.task.slot.owner event)) after) ∧
    (∀ after, ¬ Step state (.retire parent) after)

theorem parentRetentionContract_proven : ParentRetentionContract := by
  intro state child parent frame valid live link
  constructor
  · intro event after finish
    cases finish with
    | update stored allowed leaves =>
      exact leaves child frame live link rfl
  · intro after retire
    cases retire with
    | retire stored clear =>
      obtain ⟨_, held, parentLive, bit⟩ := valid.ancestry child frame parent live link
      have same := Option.some.inj (parentLive.symm.trans stored)
      subst held
      simp [MaskState.has, clear] at bit

private theorem next_monotone (step : Step state change after) : state.next ≤ after.next := by
  cases step with
  | create _ _ => exact Nat.le_succ _
  | update _ _ _ => exact Nat.le_refl _
  | retire _ _ => exact Nat.le_refl _

private theorem old_missing {key : Nat} (born : key < state.next) (missing : state.active key = none)
    (step : Step state change after) : after.active key = none := by
  cases step with
  | create _ _ => simpa [State.create, State.store, Nat.ne_of_lt born] using missing
  | @update affected frame edit live allowed mayLeave =>
    by_cases here : key = affected
    · subst key
      rw [missing] at live
      cases live
    · simpa [State.store, here] using missing
  | retire _ _ =>
    simp [State.store, missing]

def ReuseContract : Prop :=
  ∀ (state after : State) (key : Nat) (changes : List Change),
    key < state.next → state.active key = none → Run state changes after →
    key < after.next ∧ after.active key = none

theorem reuseContract_proven : ReuseContract := by
  intro state after key changes born missing run
  induction run with
  | done => exact ⟨born, missing⟩
  | step first _ ih =>
    exact ih (Nat.lt_of_lt_of_le born (next_monotone first)) (old_missing born missing first)

end NeuralReversi.Threading.SplitLifetime

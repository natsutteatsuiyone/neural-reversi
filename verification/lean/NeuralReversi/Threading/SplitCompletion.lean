import NeuralReversi.Threading.CutoffVisibility
import NeuralReversi.Bitboard.SquareProperties

namespace NeuralReversi.Threading.SplitCompletion

open AtomicWord (Event Write Read SequencedBefore InReleaseSequence)
open CutoffVisibility (History HappensBefore)

abbrev Worker := Fin 64

inductive Action where
  | book (worker : Worker) (event : Event)
  | finish (worker : Worker) (event : Event)

def Action.event : Action → Event
  | .book _ event | .finish _ event => event

def Action.release : Action → Bool
  | .book .. => false
  | .finish .. => true

structure MaskState where
  bits : BitVec 64
  accepting : Bool

def MaskState.has (state : MaskState) (worker : Worker) : Bool :=
  state.bits.getLsbD worker.val

def initial (owner : Worker) : MaskState :=
  ⟨Bitboard.squareMask owner, true⟩

def Action.updateBits (action : Action) (bits : BitVec 64) : BitVec 64 :=
  match action with
  | .book worker _ => bits ||| Bitboard.squareMask worker
  | .finish worker _ => bits &&& ~~~(Bitboard.squareMask worker)

def step (state : MaskState) (action : Action) : MaskState :=
  ⟨action.updateBits state.bits, state.accepting && !action.release⟩

def Allowed (state : MaskState) : Action → Prop
  | .book .. => state.accepting = true
  | .finish worker _ => state.has worker = true

def run (state : MaskState) : List Action → MaskState
  | [] => state
  | action :: rest => run (step state action) rest

def Legal (state : MaskState) : List Action → Prop
  | [] => True
  | action :: rest => Allowed state action ∧ Legal (step state action) rest

private theorem run_append (state : MaskState) (left right : List Action) :
    run state (left ++ right) = run (run state left) right := by
  induction left generalizing state with
  | nil => rfl
  | cons action rest ih => exact ih (step state action)

theorem legal_append (state : MaskState) (left right : List Action)
    (legal : Legal state (left ++ right)) :
    Legal state left ∧ Legal (run state left) right := by
  induction left generalizing state with
  | nil => exact ⟨trivial, legal⟩
  | cons action rest ih =>
    obtain ⟨start, tailLegal⟩ := legal
    obtain ⟨first, later⟩ := ih (step state action) tailLegal
    exact ⟨⟨start, first⟩, later⟩

theorem book_has (state : MaskState) (booked worker : Worker) (event : Event) :
    (step state (.book booked event)).has worker = (state.has worker || decide (worker = booked)) := by
  simp only [step, Action.updateBits, MaskState.has, BitVec.getLsbD_or, Bitboard.squareMask_bit, Fin.val_inj]

theorem finish_has (state : MaskState) (finished worker : Worker) (event : Event) :
    (step state (.finish finished event)).has worker = (state.has worker && !(decide (worker = finished))) := by
  simp only [step, Action.updateBits, MaskState.has, BitVec.getLsbD_and, BitVec.getLsbD_not,
    Bitboard.squareMask_bit, worker.isLt, decide_true, Bool.true_and, Fin.val_inj]

private theorem open_keeps_owner (state : MaskState) (owner : Worker) (actions : List Action)
    (keeps : state.accepting = true → state.has owner = true) :
    (run state actions).accepting = true → (run state actions).has owner = true := by
  induction actions generalizing state with
  | nil => exact keeps
  | cons action rest ih =>
    apply ih (step state action)
    cases action with
    | book worker event =>
      intro openPhase
      have wasOpen : state.accepting = true := by simpa [step, Action.release] using openPhase
      rw [book_has, keeps wasOpen]
      rfl
    | finish worker event => simp [step, Action.release]

private theorem initially_has (owner : Worker) : (initial owner).has owner = true := by
  simp [initial, MaskState.has, Bitboard.squareMask_bit]

private theorem no_action_at_zero (state : MaskState) (actions : List Action)
    (empty : state.bits = 0) (closed : state.accepting = false) (legal : Legal state actions) :
    actions = [] := by
  cases actions with
  | nil => rfl
  | cons action rest =>
    have permitted := legal.1
    cases action with
    | book worker event =>
      have openPhase : state.accepting = true := permitted
      simp [closed] at openPhase
    | finish worker event =>
      have has : state.has worker = true := permitted
      simp [MaskState.has, empty] at has

private theorem release_found (state : MaskState) (worker : Worker) (actions : List Action)
    (active : state.has worker = true) (done : (run state actions).has worker = false) :
    ∃ event, .finish worker event ∈ actions := by
  induction actions generalizing state with
  | nil => simp [run, active] at done
  | cons action rest ih =>
    cases action with
    | book booked event =>
      have activeNext : (step state (.book booked event)).has worker = true := by
        rw [book_has, active]; rfl
      obtain ⟨release, member⟩ := ih _ activeNext done
      exact ⟨release, List.mem_cons_of_mem _ member⟩
    | finish finished event =>
      by_cases same : worker = finished
      · subst finished
        exact ⟨event, List.mem_cons_self ..⟩
      · have activeNext : (step state (.finish finished event)).has worker = true := by
          simp [finish_has, active, same]
        obtain ⟨release, member⟩ := ih _ activeNext done
        exact ⟨release, List.mem_cons_of_mem _ member⟩

private theorem booked_released (state : MaskState) (worker : Worker) (event : Event)
    (actions : List Action) (booked : .book worker event ∈ actions)
    (done : (run state actions).has worker = false) :
    ∃ release, .finish worker release ∈ actions := by
  induction actions generalizing state with
  | nil => cases booked
  | cons action rest ih =>
    rcases List.mem_cons.mp booked with same | later
    · subst action
      have activeNext : (step state (.book worker event)).has worker = true := by
        simp [book_has]
      obtain ⟨released, member⟩ := release_found _ worker rest activeNext done
      exact ⟨released, List.mem_cons_of_mem _ member⟩
    · obtain ⟨released, member⟩ := ih _ later done
      exact ⟨released, List.mem_cons_of_mem _ member⟩

def ZeroContract : Prop :=
  ∀ (owner : Worker) (actions : List Action) (observed : Nat),
    Legal (initial owner) actions → observed ≤ actions.length →
    (run (initial owner) (actions.take observed)).bits = 0 →
    observed = actions.length ∧
    (∃ event, .finish owner event ∈ actions) ∧
    (∀ worker event, .book worker event ∈ actions → ∃ release, .finish worker release ∈ actions)

theorem zeroContract_proven : ZeroContract := by
  intro owner actions observed legal within empty
  let first := actions.take observed
  let later := actions.drop observed
  have parts : first ++ later = actions := List.take_append_drop _ _
  have laterLegal := (legal_append (initial owner) first later (parts.symm ▸ legal)).2
  have closed : (run (initial owner) first).accepting = false := by
    cases flag : (run (initial owner) first).accepting
    · rfl
    · have ownerBit := open_keeps_owner (initial owner) owner first (fun _ => initially_has owner) flag
      change (run (initial owner) first).bits = 0 at empty
      simp [MaskState.has, empty] at ownerBit
  have ended := no_action_at_zero _ later empty closed laterLegal
  have endLength := congrArg List.length ended
  change (actions.drop observed).length = 0 at endLength
  rw [List.length_drop] at endLength
  have endAt : observed = actions.length := by omega
  rw [endAt, List.take_length] at empty
  have gone : ∀ worker, (run (initial owner) actions).has worker = false := by
    intro worker
    simp [MaskState.has, empty]
  exact ⟨endAt, release_found (initial owner) owner actions (initially_has owner) (gone owner),
    fun worker event enrolled => booked_released (initial owner) worker event actions enrolled (gone worker)⟩

structure Activation (writes : List Write) where
  owner : Worker
  start : Fin writes.length
  actions : List Action
  endInBounds : start.val + actions.length < writes.length
  legal : Legal (initial owner) actions
  ownerBit : (writes.get start).value = (initial owner).bits
  updates : ∀ i : Fin actions.length,
    let write := writes.get ⟨start.val + i.val + 1, by have := i.isLt; omega⟩
    let action := actions.get i
    let previous := writes.get ⟨start.val + i.val, by have := i.isLt; omega⟩
    write.event = action.event ∧ write.release = action.release ∧ write.rmw = true ∧
      write.value = action.updateBits previous.value

def Activation.updateIndex (activation : Activation writes)
    (i : Fin activation.actions.length) : Fin writes.length :=
  ⟨activation.start.val + i.val + 1, by have := activation.endInBounds; have := i.isLt; omega⟩

private theorem prefix_value {writes : List Write} (activation : Activation writes)
    (n : Nat) (within : n ≤ activation.actions.length) :
    (writes.get ⟨activation.start.val + n, by have := activation.endInBounds; omega⟩).value =
      (run (initial activation.owner) (activation.actions.take n)).bits := by
  induction n with
  | zero => exact activation.ownerBit
  | succ n ih =>
    have lt : n < activation.actions.length := by omega
    have wrote := (activation.updates ⟨n, lt⟩).2.2.2
    simp only [Nat.add_assoc] at wrote
    rw [wrote, ih (by omega), List.take_succ_eq_append_getElem lt, run_append]
    rfl

structure Retirement (history : History) (writes : List Write) (activation : Activation writes) where
  read : Read writes
  acquire : read.acquire = true
  zero : (writes.get read.source).value = 0
  afterOwnerBit : SequencedBefore (writes.get activation.start).event read.event
  beforeReuse : ∀ i : Fin writes.length, i.val = activation.start.val + activation.actions.length + 1 →
    SequencedBefore read.event (writes.get i).event
  writeRead : ∀ i : Fin writes.length,
    HappensBefore history (writes.get i).event read.event → i.val ≤ read.source.val
  readWrite : ∀ i : Fin writes.length,
    HappensBefore history read.event (writes.get i).event → read.source.val < i.val
  synchronize : ∀ i : Fin writes.length, read.acquire = true →
    InReleaseSequence writes i read.source → history.otherSync (writes.get i).event read.event

private theorem observation_inside {history : History} {writes : List Write}
    {activation : Activation writes} (retirement : Retirement history writes activation) :
    activation.start.val ≤ retirement.read.source.val ∧
    retirement.read.source.val ≤ activation.start.val + activation.actions.length := by
  exact AtomicWord.ReadWindow.bounds
    ⟨.sequenced retirement.afterOwnerBit, fun i eq => .sequenced (retirement.beforeReuse i eq),
      retirement.writeRead, retirement.readWrite⟩

def EmptyReadContract : Prop :=
  ∀ (history : History) (writes : List Write) (activation : Activation writes)
    (retirement : Retirement history writes activation),
    retirement.read.source.val = activation.start.val + activation.actions.length ∧
    (∃ event, .finish activation.owner event ∈ activation.actions) ∧
    (∀ worker event, .book worker event ∈ activation.actions →
      ∃ released, .finish worker released ∈ activation.actions)

theorem emptyReadContract_proven : EmptyReadContract := by
  intro history writes activation retirement
  obtain ⟨start, stop⟩ := observation_inside retirement
  let count := retirement.read.source.val - activation.start.val
  have within : count ≤ activation.actions.length := by omega
  have same : (⟨activation.start.val + count, by have := activation.endInBounds; omega⟩ :
      Fin writes.length) = retirement.read.source := Fin.ext (Nat.add_sub_of_le start)
  have zero := (prefix_value activation count within).symm.trans (same ▸ retirement.zero)
  obtain ⟨last, ownerReleased, released⟩ :=
    zeroContract_proven activation.owner activation.actions count activation.legal within zero
  exact ⟨by change retirement.read.source.val - activation.start.val = _ at last; omega,
    ownerReleased, released⟩

private theorem all_rmw_between {writes : List Write} (activation : Activation writes)
    (i : Fin writes.length) (afterStart : activation.start.val < i.val)
    (beforeEnd : i.val ≤ activation.start.val + activation.actions.length) :
    (writes.get i).rmw = true := by
  let update : Fin activation.actions.length := ⟨i.val - activation.start.val - 1, by omega⟩
  have index : activation.updateIndex update = i := by
    apply Fin.ext
    change activation.start.val + (i.val - activation.start.val - 1) + 1 = i.val
    omega
  exact index ▸ (activation.updates update).2.2.1

def HandoffContract : Prop :=
  ∀ (history : History) (writes : List Write) (activation : Activation writes)
    (retirement : Retirement history writes activation)
    (i : Fin activation.actions.length),
    (activation.actions.get i).release = true →
    HappensBefore history (activation.actions.get i).event retirement.read.event

theorem handoffContract_proven : HandoffContract := by
  intro history writes activation retirement i released
  have last := (emptyReadContract_proven history writes activation retirement).1
  have inSequence : InReleaseSequence writes (activation.updateIndex i) retirement.read.source := by
    have update := activation.updates i
    refine ⟨update.2.1.trans released, ?_, ?_⟩
    · change activation.start.val + i.val + 1 ≤ retirement.read.source.val
      have := i.isLt
      omega
    · intro stamp after before
      change activation.start.val + i.val + 1 < stamp.val at after
      exact all_rmw_between activation stamp (by omega) (by omega)
  have edge := retirement.synchronize (activation.updateIndex i) retirement.acquire inSequence
  change history.otherSync (writes.get (activation.updateIndex i)).event retirement.read.event at edge
  have eventEq : (writes.get (activation.updateIndex i)).event = (activation.actions.get i).event :=
    (activation.updates i).1
  exact .other (eventEq ▸ edge)

inductive RetainedThrough (history : History) (access : Event) : Event → Prop
  | here : RetainedThrough history access access
  | wait {writes : List Write} {activation : Activation writes}
      (retirement : Retirement history writes activation) (i : Fin activation.actions.length) :
      (activation.actions.get i).release = true →
      RetainedThrough history access inner →
      SequencedBefore inner (activation.actions.get i).event →
      RetainedThrough history access retirement.read.event

def RetainedAccessContract : Prop :=
  ∀ (history : History) (access wait retire : Event),
    RetainedThrough history access wait → SequencedBefore wait retire →
    HappensBefore history access retire

private theorem retained_or_same (kept : RetainedThrough history access wait) :
    access = wait ∨ HappensBefore history access wait := by
  induction kept with
  | here => exact Or.inl rfl
  | wait retirement i released _ before ih =>
    have finish := handoffContract_proven _ _ _ retirement i released
    have ordered := HappensBefore.trans (.sequenced before) finish
    cases ih with
    | inl same => exact Or.inr (same.symm ▸ ordered)
    | inr prior => exact Or.inr (.trans prior ordered)

theorem retainedAccessContract_proven : RetainedAccessContract := by
  intro history access wait retire retained after
  cases retained_or_same retained with
  | inl same => exact .sequenced (same.symm ▸ after)
  | inr prior => exact .trans prior (.sequenced after)

def FlagRetirementContract : Prop :=
  ∀ (history : History) (publication : CutoffVisibility.Publication history)
    (load : CutoffVisibility.EpochRead history.epochs) (scan : CutoffVisibility.FlagReads history load),
    (∀ r ∈ scan.reads, r.slot = publication.slot →
      ∀ i : Fin (history.flags r.slot).length, publication.mark.val < i.val →
        ((history.flags r.slot).get i).value = false →
        ∃ wait, RetainedThrough history r.event wait ∧
          SequencedBefore wait ((history.flags r.slot).get i).event) →
    CutoffVisibility.OrdersClearsAfter publication scan

theorem flagRetirementContract_proven : FlagRetirementContract := by
  intro history publication load scan kept r member slot i after cleared
  obtain ⟨wait, retained, ordered⟩ := kept r member slot i after cleared
  exact retainedAccessContract_proven history r.event wait _ retained ordered

end NeuralReversi.Threading.SplitCompletion

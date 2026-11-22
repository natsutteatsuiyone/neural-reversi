import NeuralReversi.Threading.SplitLifetime

namespace NeuralReversi.Threading.SplitOwnerStack

open SplitLifetime (Slot Task Frame ParentHeld Free Valid Change)
open SplitCompletion (Worker Action)
open AtomicWord (Event Write Read ReadWindow)

structure State where
  lifetime : SplitLifetime.State
  depth : Worker → Fin 9

def empty : State :=
  ⟨SplitLifetime.empty, fun _ => 0⟩

def Occupied (lifetime : SplitLifetime.State) (slot : Slot) : Prop :=
  ∃ activation frame, lifetime.active activation = some frame ∧ frame.task.slot = slot

def Prefix (state : State) : Prop :=
  ∀ slot : Slot, Occupied state.lifetime slot ↔ slot.index.val < (state.depth slot.owner).val

private def State.setDepth (state : State) (worker : Worker) (depth : Fin 9) : Worker → Fin 9 :=
  fun other => if other = worker then depth else state.depth other

private def State.push (state : State) (task : Task) : State :=
  ⟨state.lifetime.create task,
   state.setDepth task.slot.owner ⟨task.slot.index.val + 1, Nat.succ_lt_succ task.slot.index.isLt⟩⟩

private def State.pop (state : State) (activation : Nat) (slot : Slot) : State :=
  ⟨state.lifetime.store activation none,
   state.setDepth slot.owner ⟨slot.index.val, Nat.lt_succ_of_lt slot.index.isLt⟩⟩

inductive Step : State → Change → State → Prop
  | push {state : State} {task : Task} :
      task.slot.index.val = (state.depth task.slot.owner).val →
      ParentHeld state.lifetime task → Step state (.create task) (state.push task)
  | mask {state : State} {lifetime : SplitLifetime.State} {activation : Nat} {action : Action} :
      SplitLifetime.Step state.lifetime (.update activation action) lifetime →
      Step state (.update activation action) { state with lifetime }
  | pop {state : State} {activation : Nat} {frame : Frame} :
      state.lifetime.active activation = some frame → frame.mask.bits = 0 →
      frame.task.slot.index.val + 1 = (state.depth frame.task.slot.owner).val →
      Step state (.retire activation) (state.pop activation frame.task.slot)

inductive Run : State → List Change → State → Prop
  | done : Run state [] state
  | step : Step state change next → Run next remaining final → Run state (change :: remaining) final

private theorem empty_prefix : Prefix empty := by
  intro slot
  simp [empty, SplitLifetime.empty, Occupied]

theorem free_above_prefix (packed : Prefix state) (index : (state.depth slot.owner).val ≤ slot.index.val) :
    Free state.lifetime slot := by
  intro activation frame active same
  exact Nat.not_lt_of_ge index ((packed slot).1 ⟨activation, frame, active, same⟩)

private theorem occupied_create (valid : Valid lifetime) (task : Task) (slot : Slot) :
    Occupied (lifetime.create task) slot ↔ slot = task.slot ∨ Occupied lifetime slot := by
  constructor
  · rintro ⟨activation, frame, active, same⟩
    by_cases fresh : activation = lifetime.next
    · simp [SplitLifetime.State.create, SplitLifetime.State.store, fresh] at active
      exact Or.inl (same.symm.trans (congrArg (fun f : Frame => f.task.slot) active.symm))
    · exact Or.inr ⟨activation, frame, by simpa [SplitLifetime.State.create, SplitLifetime.State.store, fresh] using active, same⟩
  · rintro (rfl | ⟨activation, frame, active, same⟩)
    · exact ⟨lifetime.next, ⟨task, SplitCompletion.initial task.slot.owner⟩,
        by simp [SplitLifetime.State.create, SplitLifetime.State.store], rfl⟩
    · have different := Nat.ne_of_lt (valid.allocated _ _ active)
      exact ⟨activation, frame, by simpa [SplitLifetime.State.create, SplitLifetime.State.store, different] using active, same⟩

private theorem occupied_update (step : SplitLifetime.Step lifetime (.update activation action) after) :
    Occupied after slot ↔ Occupied lifetime slot := by
  cases step with
  | update active allowed mayLeave =>
    rename_i frame
    constructor
    · rintro ⟨key, old, live, same⟩
      by_cases here : key = activation
      · subst key
        simp [SplitLifetime.State.store] at live
        subst old
        exact ⟨activation, frame, active, same⟩
      · exact ⟨key, old, by simpa [SplitLifetime.State.store, here] using live, same⟩
    · rintro ⟨key, old, live, same⟩
      by_cases here : key = activation
      · subst key
        have equal := Option.some.inj (active.symm.trans live)
        subst old
        exact ⟨activation, frame.apply action, by simp [SplitLifetime.State.store], same⟩
      · exact ⟨key, old, by simpa [SplitLifetime.State.store, here] using live, same⟩

private theorem occupied_retire (valid : Valid lifetime) (active : lifetime.active activation = some frame) :
    Occupied (lifetime.store activation none) slot ↔ Occupied lifetime slot ∧ slot ≠ frame.task.slot := by
  constructor
  · rintro ⟨key, other, live, same⟩
    have different : key ≠ activation := by
      rintro rfl
      simp [SplitLifetime.State.store] at live
    have prior : lifetime.active key = some other := by
      simpa [SplitLifetime.State.store, different] using live
    refine ⟨⟨key, other, prior, same⟩, ?_⟩
    intro equal
    exact different (valid.uniqueSlot _ _ _ _ prior active (same.trans equal))
  · rintro ⟨⟨key, other, live, same⟩, different⟩
    have keyDifferent : key ≠ activation := by
      rintro rfl
      exact different (same.symm.trans (congrArg (fun f : Frame => f.task.slot) (Option.some.inj (live.symm.trans active))))
    exact ⟨key, other, by simpa [SplitLifetime.State.store, keyDifferent] using live, same⟩

private theorem same_owner_index (a b : Slot) (same : a.owner = b.owner) :
    a = b ↔ a.index.val = b.index.val := by
  cases a
  cases b
  simp_all [Slot.mk.injEq, Fin.ext_iff]

private theorem push_prefix (valid : Valid state.lifetime) (packed : Prefix state)
    (atDepth : task.slot.index.val = (state.depth task.slot.owner).val) : Prefix (state.push task) := by
  intro slot
  change Occupied (state.lifetime.create task) slot ↔ _
  rw [occupied_create valid, packed]
  cases slot with
  | mk owner index =>
    by_cases same : owner = task.slot.owner
    · subst owner
      rw [same_owner_index ⟨task.slot.owner, index⟩ task.slot rfl]
      simp [State.push, State.setDepth]
      omega
    · have different : (⟨owner, index⟩ : Slot) ≠ task.slot := fun eq => same (congrArg Slot.owner eq)
      simp [State.push, State.setDepth, same, different]

private theorem pop_prefix (valid : Valid state.lifetime) (packed : Prefix state)
    (active : state.lifetime.active activation = some frame)
    (atTop : frame.task.slot.index.val + 1 = (state.depth frame.task.slot.owner).val) :
    Prefix (state.pop activation frame.task.slot) := by
  intro slot
  change Occupied (state.lifetime.store activation none) slot ↔ _
  rw [occupied_retire valid active, packed]
  cases slot with
  | mk owner index =>
    by_cases same : owner = frame.task.slot.owner
    · subst owner
      simp only [ne_eq, same_owner_index ⟨frame.task.slot.owner, index⟩ frame.task.slot rfl]
      simp [State.pop, State.setDepth]
      omega
    · have different : (⟨owner, index⟩ : Slot) ≠ frame.task.slot := fun eq => same (congrArg Slot.owner eq)
      simp [State.pop, State.setDepth, same, different]

theorem Step.refines (step : Step state change next) (valid : Valid state.lifetime) (packed : Prefix state) :
    SplitLifetime.Step state.lifetime change next.lifetime ∧ Prefix next := by
  cases step with
  | push depth parent =>
    exact ⟨.create (free_above_prefix packed (Nat.le_of_eq depth.symm)) parent, push_prefix valid packed depth⟩
  | mask edited =>
    exact ⟨edited, fun slot => (occupied_update edited).trans (packed slot)⟩
  | pop active clear depth =>
    exact ⟨.retire active clear, pop_prefix valid packed active depth⟩

theorem Run.refines (run : Run state changes final) (valid : Valid state.lifetime) (packed : Prefix state) :
    SplitLifetime.Run state.lifetime changes final.lifetime ∧ Prefix final := by
  induction run with
  | done => exact ⟨.done, packed⟩
  | step first _ ih =>
    obtain ⟨refined, filled⟩ := first.refines valid packed
    obtain ⟨remaining, finalPrefix⟩ := ih (SplitLifetime.step_valid valid refined) filled
    exact ⟨.step refined remaining, finalPrefix⟩

def StackContract : Prop :=
  ∀ (changes : List Change) (state : State), Run empty changes state →
    SplitLifetime.Run SplitLifetime.empty changes state.lifetime ∧ Prefix state

theorem stackContract_proven : StackContract :=
  fun _ _ run => run.refines SplitLifetime.empty_valid empty_prefix

def counterWord (width : Nat) (state : State) (owner : Worker) : BitVec width :=
  BitVec.ofNat width (state.depth owner).val

theorem counterWord_toNat (fits : 8 < 2 ^ width) (state : State) (owner : Worker) :
    (counterWord width state owner).toNat = (state.depth owner).val := by
  rw [counterWord, BitVec.toNat_ofNat]
  exact Nat.mod_eq_of_lt (Nat.lt_of_le_of_lt (Nat.le_of_lt_succ (state.depth owner).isLt) fits)

def CounterUpdateContract : Prop :=
  ∀ (width : Nat) (state next : State) (change : Change), Step state change next →
    match change with
    | .create task => ∀ owner,
        counterWord width next owner =
          if owner = task.slot.owner then counterWord width state owner + 1 else counterWord width state owner
    | .update .. => ∀ owner, counterWord width next owner = counterWord width state owner
    | .retire activation => ∃ frame, state.lifetime.active activation = some frame ∧
        ∀ owner, counterWord width next owner =
          if owner = frame.task.slot.owner then counterWord width state owner - 1 else counterWord width state owner

theorem counterUpdateContract_proven : CounterUpdateContract := by
  intro width state next change step
  cases step with
  | push depth parent =>
    rename_i task
    intro owner
    by_cases same : owner = task.slot.owner
    · subst owner
      simp [counterWord, State.push, State.setDepth, depth, BitVec.ofNat_add]
    · simp [counterWord, State.push, State.setDepth, same]
  | mask edited => intro owner; rfl
  | pop active clear depth =>
    rename_i frame
    refine ⟨frame, active, ?_⟩
    intro owner
    by_cases same : owner = frame.task.slot.owner
    · subst owner
      simp [counterWord, State.pop, State.setDepth, ← depth, BitVec.ofNat_add, BitVec.add_sub_cancel]
    · simp [counterWord, State.pop, State.setDepth, same]

private theorem window_source (window : ReadWindow load before current current.val) :
    load.source = current :=
  Fin.ext (Nat.le_antisymm window.bounds.2 window.bounds.1)

def OwnerSelectionContract : Prop :=
  ∀ (writes : List Write) (current : Fin writes.length) (check select : Read writes)
    (before : Event → Event → Prop) (state : State) (owner : Worker),
    Prefix state →
    ReadWindow check before current current.val → ReadWindow select before current current.val →
    (writes.get current).value.toNat = (state.depth owner).val →
    (writes.get check.source).value.toNat < 8 →
    ∃ index : Fin 8, (writes.get select.source).value.toNat = index.val ∧
      index.val = (state.depth owner).val ∧ Free state.lifetime ⟨owner, index⟩

theorem ownerSelectionContract_proven : OwnerSelectionContract := by
  intro _ current check select _ state owner packed checked selected width pass
  have checkSource := window_source checked
  have selectSource := window_source selected
  rw [checkSource, width] at pass
  exact ⟨⟨(state.depth owner).val, pass⟩, selectSource.symm ▸ width, rfl, free_above_prefix packed (Nat.le_refl _)⟩

def ObserverIndexContract : Prop :=
  ∀ (writes : List Write) (read : Read writes),
    (∀ i : Fin writes.length, (writes.get i).value.toNat ≤ 8) →
    let count := (writes.get read.source).value.toNat
    count = 0 ∨ ∃ index : Fin 8, index.val + 1 = count

theorem observerIndexContract_proven : ObserverIndexContract := by
  intro writes read bounded
  let count := (writes.get read.source).value.toNat
  change count = 0 ∨ ∃ index : Fin 8, index.val + 1 = count
  have limit := bounded read.source
  cases known : count with
  | zero => exact Or.inl rfl
  | succ index =>
    change count ≤ 8 at limit
    rw [known] at limit
    exact Or.inr ⟨⟨index, by omega⟩, rfl⟩

theorem full_has_all_slots (packed : Prefix state) (full : (state.depth owner).val = 8) :
    ∀ index : Fin 8, Occupied state.lifetime ⟨owner, index⟩ := by
  intro index
  exact (packed ⟨owner, index⟩).2 (full.symm ▸ index.isLt)

theorem no_push_at_full (full : (state.depth task.slot.owner).val = 8) :
    ¬ Step state (.create task) next := by
  intro push
  cases push with
  | push depth _ => exact Nat.ne_of_lt task.slot.index.isLt (depth.trans full)

private abbrev sampleWrites : List Write :=
  [⟨⟨0, 0⟩, 0, false, false⟩, ⟨⟨0, 2⟩, 1, true, true⟩, ⟨⟨0, 5⟩, 0, true, true⟩]

private def sampleRead (step : Nat) (source : Fin sampleWrites.length) : Read sampleWrites :=
  ⟨⟨0, step⟩, source, false⟩

private theorem sampleWindow (step : Nat) (early : 2 < step) (late : step < 5) :
    ReadWindow (sampleRead step 1) AtomicWord.SequencedBefore 1 1 := by
  constructor
  · exact ⟨rfl, early⟩
  · intro i next
    have last : i = 2 := Fin.ext next
    subst i
    exact ⟨rfl, late⟩
  · intro i before
    change i.val ≤ 1
    by_cases sourceBound : i.val ≤ 1
    · exact sourceBound
    · have last : i = 2 := Fin.ext (by
        change i.val = 2
        have bound : i.val < 3 := i.isLt
        omega)
      subst i
      have impossible : 5 < step := before.2
      omega
  · intro i before
    change 1 < i.val
    by_cases sourceBound : 1 < i.val
    · exact sourceBound
    · have earlyWrite : i = 0 ∨ i = 1 := by
        by_cases first : i.val = 0
        · exact Or.inl (Fin.ext first)
        · exact Or.inr (Fin.ext (by change i.val = 1; omega))
      rcases earlyWrite with rfl | rfl
      · exact (Nat.not_lt_zero _ before.2).elim
      · exact (Nat.not_lt_of_ge (Nat.le_of_lt early) before.2).elim

example : ∃ index : Fin 8, index.val = 1 ∧ Free
    (empty.push ⟨⟨0, 0⟩, none⟩).lifetime ⟨0, index⟩ := by
  let task : Task := ⟨⟨0, 0⟩, none⟩
  have step : Step empty (.create task) (empty.push task) := .push rfl (by simp [task, ParentHeld])
  have packed := (stackContract_proven _ _ (.step step .done)).2
  obtain ⟨index, observed, _, free⟩ := ownerSelectionContract_proven
    sampleWrites 1 (sampleRead 3 1) (sampleRead 4 1) AtomicWord.SequencedBefore (empty.push task) 0
    packed (sampleWindow 3 (by decide) (by decide)) (sampleWindow 4 (by decide) (by decide)) rfl (by decide)
  exact ⟨index, observed.symm, free⟩

example :
    (∀ i : Fin sampleWrites.length,
      AtomicWord.SequencedBefore (sampleRead 4 0).event (sampleWrites.get i).event → 0 < i.val) ∧
    ¬ ReadWindow (sampleRead 4 0) AtomicWord.SequencedBefore 1 1 := by
  constructor
  · intro i before
    have lower : 1 < i.val := (sampleWindow 4 (by decide) (by decide)).readWrite i before
    omega
  · intro window
    have bound := window.bounds.1
    change 1 ≤ 0 at bound
    omega

example :
    (∀ i : Fin sampleWrites.length,
      AtomicWord.SequencedBefore (sampleWrites.get i).event (sampleRead 4 2).event → i.val ≤ 2) ∧
    ¬ ReadWindow (sampleRead 4 2) AtomicWord.SequencedBefore 1 1 := by
  constructor
  · intro i _
    exact Nat.le_of_lt_succ i.isLt
  · intro window
    have bound := window.bounds.2
    change 2 ≤ 1 at bound
    omega

end NeuralReversi.Threading.SplitOwnerStack

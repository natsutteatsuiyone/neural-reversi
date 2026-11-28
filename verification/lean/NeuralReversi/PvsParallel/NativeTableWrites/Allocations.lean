module
public import NeuralReversi.PvsParallel.NativeTableWrites.Schedule
public import NeuralReversi.TranspositionTable.Duplicate.View
public import NeuralReversi.TranspositionTable.Duplicate.Image

@[expose] public section

namespace NeuralReversi.PvsParallel.NativeTableWrites.Journal

open TranspositionTable
open SeqLock (Location Memory Payload)
open FrameTableReads (Address Birth)

variable {x : Execution c endian rs}

theorem extent (x : Execution c endian rs) (a : Address) (init : Even (x.memory a)) (loc : Location) :
    Duplicate.modifications (x.slot a init) loc = (mods c rs a loc).length := by
  cases loc <;> simp [Duplicate.modifications,Execution.slot,Execution.toSequence,Execution.toFields,sequence,fields]

private theorem claimSource (x : Execution c endian rs) (a : Address) (init : Even (x.memory a))
    (member : WriterTrace.Action.claim call ∈ (x.slot a init).history) :
    ∃ k, Record.writer ⟨k,.cas⟩ ∈ rs ∧ (c k).address = a ∧ (c k).success = true ∧ call = (c k).call endian := by
  obtain ⟨⟨k,pc⟩,inSlot,same⟩ := List.mem_map.mp member
  obtain ⟨hit,addr,success,_⟩ := mem_mods.mp inSlot
  simp only [Site.action] at same
  split at same <;> try contradiction
  rename_i claim
  subst pc
  exact ⟨k,hit,addr,success,(WriterTrace.Action.claim.inj same).symm⟩

/-- The writer schedule up to a synchronized copy. Preceding source writes
    happen before the sample and subsequent writes happen after it. -/
structure CopyPoint (x : Execution c endian rs) (a : Address) (sample : Nat) where
  turns : List Turn
  included : (schedule c endian turns).IsPrefix rs
  idle : (stateAfter c endian turns).idleAt c endian a = true
  prior : ∀ site, Record.writer site ∈ schedule c endian turns → (c site.invocation).address = a →
    site.pc.location.isSome → (c site.invocation).success = true → (x.memory a).hb (site.event c) sample
  later : ∀ site, Record.writer site ∈ rs → Record.writer site ∉ schedule c endian turns →
    (c site.invocation).address = a → site.pc.location.isSome → (c site.invocation).success = true →
      (x.memory a).hb sample (site.event c)

private theorem CopyPoint.joined (point : CopyPoint x a sample) (k : Nat)
    (cas : Record.writer ⟨k,.cas⟩ ∈ rs) (addr : (c k).address = a) (ok : (c k).success = true)
    (hb : (x.memory a).hb ((c k).event 0) sample) :
    Record.writer ⟨k,.finish⟩ ∈ rs ∧ (x.memory a).hb ((c k).event 5) sample := by
  have before : Record.writer ⟨k,.cas⟩ ∈ schedule c endian point.turns := by
    by_cases seen : Record.writer ⟨k,.cas⟩ ∈ schedule c endian point.turns
    · exact seen
    · have after := point.later _ cas seen addr rfl ok
      exact ((x.memory a).hb_irrefl _ ((x.memory a).hb_trans after hb)).elim
  have released := releaseBeforeCopy c endian point.turns k (addr ▸ point.idle) before ok
  exact ⟨point.included.subset released,point.prior _ released addr rfl ok⟩

noncomputable def Execution.wordsAt (x : Execution c endian rs) (allocation sample index : Nat) :
    Location → SeqLock.Word :=
  (Duplicate.Visible.atCopy (x.memory ⟨allocation,index⟩)
    (fun loc => (mods c rs ⟨allocation,index⟩ loc).length) sample).word

/-- Immediate inputs to one copied slot. Allocation ordinals distinguish
    lifetimes, including reuse of the same numerical pointer. -/
structure Clone (x : Execution c endian rs) (before : Nat → Nat → Prop) (ready : Address → Nat) (a : Address) where
  parent : Nat
  older : parent < a.allocation
  buffers : Duplicate.Buffers
  inBounds : a.index < buffers.clusters*2
  sample : Nat
  allocated : before (ready ⟨parent,a.index⟩) sample
  returned : before sample (ready a)
  destImage : ∀ loc, (x.memory a).writeValue loc 0 = Duplicate.readWord endian
    (Duplicate.image (buffers.after (Duplicate.materialize endian buffers.source (x.wordsAt parent sample)))
      buffers.dest a.index) loc
  point : CopyPoint x ⟨parent,a.index⟩ sample

noncomputable def Clone.visible (op : Clone x before ready a) :=
  Duplicate.Visible.atCopy (x.memory ⟨op.parent,a.index⟩) (fun l => (mods c rs ⟨op.parent,a.index⟩ l).length) op.sample

noncomputable def Clone.view (op : Clone x before ready a) (init : Even (x.memory ⟨op.parent,a.index⟩)) :
    Duplicate.View (x.slot ⟨op.parent,a.index⟩ init) op.sample where
  index := op.visible.index
  bounded loc := (extent x _ init loc).symm ▸ op.visible.bounded loc
  before := op.visible.before
  latest loc i bound := op.visible.latest loc i ((extent x _ init loc) ▸ bound)

private theorem Clone.copied (op : Clone x before ready a) (loc : Location) :
    (x.memory a).writeValue loc 0 = op.visible.word loc := by
  rw [op.destImage,op.buffers.image_eq _ op.inBounds,Duplicate.readMaterialized]
  rfl

theorem Clone.stable (op : Clone x before ready a) (init : Even (x.memory ⟨op.parent,a.index⟩)) :
    (op.view init).word .sequence &&& 1 = 0 := by
  apply Duplicate.View.evenOfQuiescent
  intro call member hb
  obtain ⟨k,issued,addr,success,rfl⟩ := claimSource x _ init member
  obtain ⟨released,earlier⟩ := op.point.joined k issued addr success hb
  exact ⟨List.mem_map.mpr ⟨⟨k,.finish⟩,mem_mods.mpr ⟨released,addr,success,rfl⟩,rfl⟩,earlier⟩

inductive Start (x : Execution c endian rs) (before : Nat → Nat → Prop) (ready : Address → Nat) (a : Address) : Prop
  | zeroed (blank : Duplicate.Blank endian a.index (fun loc => (x.memory a).writeValue loc 0))
  | duplicate (op : Clone x before ready a)

theorem Start.even (starts : ∀ a, Start x before ready a) (a : Address) : Even (x.memory a) := by
  obtain ⟨id,index⟩ := a
  induction id using Nat.strongRecOn with
  | ind id ih =>
    cases starts ⟨id,index⟩ with
    | zeroed blank =>
      unfold Even
      rw [blank.zero .sequence]
      rfl
    | duplicate op =>
      change (x.memory _).writeValue .sequence 0 &&& 1 = 0
      rw [op.copied]
      exact op.stable (ih op.parent op.older)

theorem Clone.payload (op : Clone x before ready a) (init : Even (x.memory ⟨op.parent,a.index⟩))
    (dest : Even (x.memory a)) :
    (x.slot a dest).payloadAt 0 =
      (x.slot ⟨op.parent,a.index⟩ init).payloadAt (op.visible.index .sequence / 2) := by
  let v := op.view init
  have even := Duplicate.View.evenIndex v (op.stable init)
  have fields (f : SeqLock.Field) : (x.memory a).writeValue (.payload f) 0 =
      (x.memory ⟨op.parent,a.index⟩).writeValue (.payload f) (op.visible.index .sequence / 2) := by
    rw [op.copied]
    change _ = (x.memory ⟨op.parent,a.index⟩).writeValue (.payload f) (v.index .sequence / 2)
    rw [← Duplicate.View.fieldIndex v _ even f]
    rfl
  simp only [WriterTrace.Slot.payloadAt,fields]

/-- A slot copied through any finite chain of allocations inherits a concrete
    earlier publication, or the vacant initialization. Later writes to each
    source allocation remain in the same global execution. -/
def Execution.table (x : Execution c endian rs) (log : FrameReturns.Log mc fc) (ready : Address → Nat)
    (starts : ∀ a, Start x log.before ready a)
    (recorded : log.writes = NativeFrames.tableEvents (rs.filterMap (frameRecord c endian)))
    (agrees : ∀ a i j, (x.memory a).hb i j → log.before i j)
    (trans : ∀ {i j k}, log.before i j → log.before j k → log.before i k) : FrameTableReads.Table log := by
  let j := x.journal log (Start.even starts) recorded agrees trans
  refine ⟨j,ready,?_⟩
  rintro ⟨id,index⟩
  induction id using Nat.strongRecOn with
  | ind id ih =>
    cases starts ⟨id,index⟩ with
    | zeroed blank => exact .vacant _ _ (blank.zero (.payload .metadata))
    | duplicate op =>
      let parent : Address := ⟨op.parent,index⟩
      let init := Start.even starts parent
      let v := op.view init
      let epoch := v.index .sequence / 2
      have even : v.index .sequence = 2*epoch := Duplicate.View.evenIndex v (op.stable init)
      have bounded : 2*epoch ≤ (j.slot parent).history.length := even ▸ v.bounded .sequence
      apply Birth.later j _ op.returned
      apply j.duplicate parent (ih op.parent op.older) op.allocated epoch bounded _ (op.payload init _)
      intro positive call actual
      have release := ((j.slot parent).modified _ _ actual).1
      rw [show 2*epoch-1+1 = 2*epoch by omega] at release
      change (j.memory parent).write .sequence (2*epoch) = call.finish at release
      rw [← release,← even]
      exact v.before .sequence (by omega)

end NeuralReversi.PvsParallel.NativeTableWrites.Journal

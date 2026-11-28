module
public import NeuralReversi.PvsParallel.FrameReturns
public import NeuralReversi.TranspositionTable.ReaderMir.Publication
public import NeuralReversi.TranspositionTable.ReaderMir.Lookup

@[expose] public section

namespace NeuralReversi.PvsParallel.FrameTableReads

open TranspositionTable
open SeqLock (Word Memory Payload)
open ConcurrentReplay (Event)
open FrameReturns (Log Frame Samples)
open Cutoffs (RawSample)

/-- The allocation is part of the address when the source and its duplicate are both live. -/
structure Address where
  allocation : Nat
  index : Nat
  deriving DecidableEq

def slotAt (allocation : Nat) (clusters key : Word) (i : Fin 2) : Address :=
  ⟨allocation,Clear.getClusterIdx key clusters + i.val⟩

theorem slotAt_in_bounds (allocation : Nat) (clusters key : Word) (positive : 0 < clusters.toNat) (i : Fin 2) :
    (slotAt allocation clusters key i).index < clusters.toNat * 2 :=
  Clear.probe_slot_in_bounds key clusters i positive i.isLt

inductive Mark where
  | claim (address : Address) (call : WriterTrace.Call Nat)
  | finish (address : Address) (event : Event)
  | other (event : Event)

def Mark.event? : Mark → Option Event
  | .claim .. => none
  | .finish _ e | .other e => some e

def Mark.actionAt (a : Address) : Mark → Option (WriterTrace.Action Nat)
  | .claim atSlot call => if atSlot = a then some (.claim call) else none
  | .finish atSlot e => do
    if atSlot ≠ a then none else
      let call ← e.call
      if e.id = call.finish then some (.publish call) else none
  | .other _ => none

structure Journal (log : Log mc fc) where
  trace : List Mark
  recorded : trace.filterMap Mark.event? = log.writes
  memory : Address → Memory Nat
  slot : (a : Address) → WriterTrace.Slot (memory a)
  journal : ∀ a, (slot a).history = trace.filterMap (Mark.actionAt a)
  agrees : ∀ a x y, (memory a).hb x y → log.before x y
  trans : ∀ {i j k}, log.before i j → log.before j k → log.before i k

theorem Journal.published {log : Log mc fc} (t : Journal log) (a : Address) {i : Nat} {call : WriterTrace.Call Nat}
    (seen : (t.slot a).history[i]? = some (.publish call)) :
    ∃ e ∈ log.writes, e.call = some call ∧ e.id = call.finish := by
  have hm := List.mem_of_getElem? seen
  rw [t.journal] at hm
  obtain ⟨mark,mem,hit⟩ := List.mem_filterMap.mp hm
  cases mark with
  | other e => contradiction
  | claim address c =>
    simp only [Mark.actionAt] at hit
    split at hit <;> cases hit
  | finish address e =>
    simp only [Mark.actionAt] at hit
    split at hit <;> try contradiction
    obtain ⟨c,published,eq⟩ := Option.bind_eq_some_iff.mp hit
    split at eq <;> try contradiction
    rename_i id
    cases eq
    refine ⟨e,?_,published,id⟩
    rw [← t.recorded]
    exact List.mem_filterMap.mpr ⟨_,mem,rfl⟩

inductive Birth (log : Log mc fc) : Nat → Payload → Prop
  | vacant (ready : Nat) (words : Payload) : words.metadata = 0 → Birth log ready words
  | copied (ready : Nat) (e : Event) (call : WriterTrace.Call Nat) :
      e ∈ log.writes → e.call = some call → log.before e.id ready → Birth log ready call.payload

theorem Birth.later {log : Log mc fc} (t : Journal log) (born : Birth log old words) (hb : log.before old new) :
    Birth log new words := by
  cases born with
  | vacant _ zero => exact .vacant _ _ zero
  | copied e c mem written earlier => exact .copied _ e c mem written (t.trans earlier hb)

/-- Copying a quiescent source slot at a completed epoch, including a duplicate of a duplicate. -/
theorem Journal.duplicate {log : Log mc fc} (t : Journal log) (a : Address)
    (source : Birth log old ((t.slot a).payloadAt 0)) (ready : log.before old new)
    (epoch : Nat) (complete : 2 * epoch ≤ (t.slot a).history.length)
    (prior : 0 < epoch → ∀ call, (t.slot a).history[2*epoch-1]? = some (.publish call) → (t.memory a).hb call.finish new)
    (copied : dest = (t.slot a).payloadAt epoch) : Birth log new dest := by
  subst dest
  cases epoch with
  | zero => exact source.later t ready
  | succ epoch =>
    obtain ⟨call,index,words⟩ := (t.slot a).publishedPayload _ (by omega) complete
    obtain ⟨e,mem,write,id⟩ := t.published a index
    rw [words]
    exact .copied _ e call mem write (id ▸ t.agrees a _ _ (prior (by omega) call index))

structure Table (log : Log mc fc) extends Journal log where
  ready : Address → Nat
  initial : ∀ a, Birth log (ready a) ((slot a).payloadAt 0)

structure Read (log : Log mc fc) (t : Table log) (a : Address) (consumer : Nat) where
  ev : Nat → Nat
  source : Nat → Nat
  allocated : log.before (t.ready a) (ev 0)
  validated : (t.memory a).sb (ev 5) consumer
  noWrap : source 5 < source 0 + SeqLock.modulus

private theorem ordered {m : Memory Nat} (run : ReaderMir.Realizes m actions ev source)
    {i j : Nat} (lt : i < j) (inside : j < actions.length) : m.sb (ev i) (ev j) := by
  induction j with
  | zero => omega
  | succ j ih =>
    by_cases last : i = j
    · subst j; exact run.adjacent i inside
    · exact m.sb_trans (ih (by omega) (by omega)) (run.adjacent j inside)

def Read.words {log : Log mc fc} {t : Table log} (r : Read log t a f) := (t.memory a).readValue ∘ r.ev

private structure Six (acts : List ReaderMir.AtomicAction) where
  size : acts.length = 6
  entry : acts[0]? = some (.load .sequence .acquire)
  fence : acts[4]? = some (.fence .acquire)
  exit : acts[5]? = some (.load .sequence .relaxed)
  field : SeqLock.Field → Fin 3
  load : ∀ fld, acts[(field fld).val+1]? = some (.load (.payload fld) .relaxed)

private def rawSample {m : Memory Nat} (slot : WriterTrace.Slot m) (ev source : Nat → Nat)
    (noWrap : source 5 < source 0 + SeqLock.modulus) (endian : Bool) (layout : Six acts)
    (steps : ReaderMir.Realizes m acts ev source) : RawSample := by
  let reader : SeqLock.RawReader m (source 0) (source 5) :=
    ⟨ev 0,ev 4,ev 5,steps.acquire 0 _ layout.entry,steps.fence 4 layout.fence,
      ordered steps (by decide) (by rw [layout.size]; decide),steps.adjacent 4 (by rw [layout.size]; decide),
      steps.reads 0 _ _ layout.entry,steps.reads 5 _ _ layout.exit⟩
  let rd (fld : SeqLock.Field) :
      SeqLock.FieldRead reader fld (source ((layout.field fld).val+1)) :=
    ⟨ev _,ordered steps (by omega) (by have := (layout.field fld).isLt; rw [layout.size]; omega),
      ordered steps (by have := (layout.field fld).isLt; omega) (by rw [layout.size]; decide),
      steps.reads _ _ _ (layout.load fld)⟩
  exact ⟨m,source 0,source 5,slot,reader,_,_,_,rd .player,rd .opponent,rd .metadata,
    noWrap,endian⟩

private def Read.raw {log : Log mc fc} {t : Table log} (r : Read log t addr f) (endian : Bool)
    (layout : Six acts) (steps : ReaderMir.Realizes (t.memory addr) acts r.ev r.source) : RawSample :=
  rawSample (t.slot addr) r.ev r.source r.noWrap endian layout steps

private theorem Read.evidence {log : Log mc fc} {t : Table log} (r : Read log t addr consumer) (endian : Bool)
    (layout : Six acts) (steps : ReaderMir.Realizes (t.memory addr) acts r.ev r.source)
    (even : r.words 0 &&& 1 = 0) :
    ConcurrentReplay.ReadAt log.before log.events consumer endian (r.raw endian layout steps) := by
  refine ⟨rfl,even,?_,r.validated,t.agrees addr,?_⟩
  · intro _
    generalize seed : (t.slot addr).payloadAt 0 = initial at *
    have birth := Eq.mp (congrArg (Birth log (t.ready addr)) seed) (t.initial addr)
    cases birth with
    | vacant _ zero =>
      apply ConcurrentReplay.SlotOrigin.empty
      change (Packing.unpack endian ((t.slot addr).payloadAt 0).metadata).bound = 0
      rw [seed,zero]
      cases endian <;> rfl
    | copied e call mem write before =>
      exact .copied e call (List.mem_append_right _ mem) write (t.trans before r.allocated) seed
  · intro call index
    obtain ⟨e,mem,write,_⟩ := t.toJournal.published addr index
    exact List.mem_filterMap.mpr ⟨e,List.mem_append_right _ mem,write⟩

structure InvocationAt {log : Log mc fc} (t : Table log) (addr : Address) (consumer : Nat)
    (code : (Nat → Word) → List ReaderMir.AtomicAction) where
  ev : Nat → Nat
  source : Nat → Nat
  allocated : log.before (t.ready addr) (ev 0)
  steps : ReaderMir.Realizes (t.memory addr) (code ((t.memory addr).readValue ∘ ev)) ev source
  validated : (code ((t.memory addr).readValue ∘ ev)).length = 6 → (t.memory addr).sb (ev 5) consumer
  noWrap : (code ((t.memory addr).readValue ∘ ev)).length = 6 → source 5 < source 0 + SeqLock.modulus

abbrev Invocation {log : Log mc fc} (t : Table log) (addr : Address) (f : Frame) :=
  InvocationAt t addr (log.identify f.use)

def InvocationAt.words {log : Log mc fc} {t : Table log} (r : InvocationAt t addr consumer code) :=
  (t.memory addr).readValue ∘ r.ev

private def InvocationAt.full {log : Log mc fc} {t : Table log} (r : InvocationAt t addr consumer code)
    (size : (code r.words).length = 6) : Read log t addr consumer :=
  ⟨r.ev,r.source,r.allocated,r.validated size,r.noWrap size⟩

def snapCode (endian : Bool) (w : Nat → Word) := (ReaderMir.run endian w).actions

def snapshot (endian : Bool) (w : Nat → Word) : Option (ProbeRetry.Key × EntryData) :=
  (ReaderMir.run endian w).returned.map (fun s => ((s.player,s.opponent),Packing.unpack endian s.metadata))

private def snapActions : List ReaderMir.AtomicAction :=
  [.load .sequence .acquire,.load (.payload .metadata) .relaxed,.load (.payload .player) .relaxed,
   .load (.payload .opponent) .relaxed,.fence .acquire,.load .sequence .relaxed]

private def snapSix : Six snapActions :=
  ⟨rfl,rfl,rfl,rfl,(fun | .player => 1 | .opponent => 2 | .metadata => 0),by intro f; cases f <;> rfl⟩

private theorem snap_path (w : Nat → Word) (endian : Bool) (even : ¬w 0 &&& 1 ≠ 0)
    (occupied : (Packing.unpack endian (w 1)).bound ≠ 0#8) (equal : ¬w 0 ≠ w 5) :
    snapCode endian w = snapActions := by
  unfold snapCode ReaderMir.run
  rw [ite_eq_right even,ite_eq_right occupied,ite_eq_right equal]
  rfl

/-- On the vacant branch the final sequence load is action 3. The keys returned there are zero. -/
def saveObservation (endian : Bool) (w : Nat → Word) (current : Word) : Save.Observation :=
  if (Packing.unpack endian (w 1)).bound = 0 then ⟨w 0,w 3,⟨0,0,w 1⟩,current⟩
  else ⟨w 0,w 5,⟨w 2,w 3,w 1⟩,current⟩

theorem save_read {log : Log mc fc} (t : Table log) (endian : Bool)
    (r : InvocationAt t addr consumer (snapCode endian)) (current : Word) :
    let obs := saveObservation endian r.words current
    ∀ s, Snapshot.read endian obs.first obs.last obs.sample = some s → (Packing.unpack endian s.metadata).bound ≠ 0 →
      ∃ raw, ConcurrentReplay.ReadAt log.before log.events consumer endian raw ∧ raw.bigEndian = endian ∧
        raw.first = obs.first ∧ raw.last = obs.last ∧ raw.payload = obs.sample := by
  intro obs s accepted occupied
  obtain ⟨even,same,metadata,_,_⟩ := (Snapshot.readContract_proven _ _ _ _ _).mp accepted
  by_cases empty : (Packing.unpack endian (r.words 1)).bound = 0
  · have key : s.metadata = r.words 1 := by simpa [obs,saveObservation,empty] using metadata
    exact (occupied (key ▸ empty)).elim
  · have full : obs = ⟨r.words 0,r.words 5,⟨r.words 2,r.words 3,r.words 1⟩,current⟩ := by
      simp [obs,saveObservation]
      intro vacant
      exact (empty vacant).elim
    rw [full] at even same ⊢
    have path := snap_path r.words endian (fun h => h even) empty (fun h => h same)
    have size : (snapCode endian r.words).length = 6 := by rw [path]; rfl
    let load := r.full size
    have steps : ReaderMir.Realizes (t.memory addr) snapActions r.ev r.source := path ▸ r.steps
    exact ⟨load.raw endian snapSix steps,load.evidence endian snapSix steps even,rfl,rfl,rfl,rfl⟩

theorem snapshot_read_at {log : Log mc fc} (t : Table log) (endian : Bool)
    (r : InvocationAt t addr consumer (snapCode endian)) :
    TableReads.SnapshotReadFrom (ConcurrentReplay.ReadAt log.before log.events consumer endian) (snapshot endian r.words) := by
  dsimp only [snapshot]
  unfold ReaderMir.run
  split
  · exact .unavailable
  next even =>
    split
    next vacant =>
      split
      · exact .unavailable
      · exact .vacant _ vacant
    next occupied =>
      split
      · exact .unavailable
      next equal =>
        have path := snap_path r.words endian even occupied equal
        have size : (snapCode endian r.words).length = 6 := by rw [path]; rfl
        let full := r.full size
        have steps : ReaderMir.Realizes (t.memory addr) snapActions r.ev r.source := path ▸ r.steps
        let raw := full.raw endian snapSix steps
        have source := full.evidence endian snapSix steps (Classical.not_not.mp even)
        apply TableReads.SnapshotReadFrom.sampled raw ⟨r.words 2,r.words 3,r.words 0,r.words 1⟩ source
        apply (Snapshot.readContract_proven _ _ _ _ _).mpr
        change r.words 0 &&& 1 = 0 ∧ r.words 0 = r.words 5 ∧ _ ∧ _ ∧ _
        refine ⟨Classical.not_not.mp even,Classical.not_not.mp equal,rfl,rfl,?_⟩
        change (if Snapshot.empty endian (r.words 1) then _ else _)
        have notEmpty : Snapshot.empty endian (r.words 1) = false := by simpa [Snapshot.empty] using occupied
        rw [notEmpty]
        exact ⟨rfl,rfl⟩

theorem snapshot_read {log : Log mc fc} (t : Table log) (endian : Bool)
    (r : Invocation t addr f (snapCode endian)) :
    TableReads.SnapshotReadFrom (Samples log endian f) (snapshot endian r.words) :=
  snapshot_read_at t endian r

private theorem Birth.occupied {log : Log mc fc} (born : Birth log old words)
    (nz : Snapshot.empty endian words.metadata = false) :
    ∃ e ∈ log.writes, ∃ call, e.call = some call ∧ log.before e.id old ∧ words = call.payload := by
  cases born with
  | vacant _ zero => rw [zero] at nz; cases endian <;> contradiction
  | copied e call mem write prior => exact ⟨e,mem,call,write,prior,rfl⟩

private theorem occupied_size (endian : Bool) (words : Nat → Word) (result : SeqLock.Snapshot)
    (hit : (ReaderMir.run endian words).returned = some result)
    (occupied : Snapshot.empty endian result.metadata = false) : (snapCode endian words).length = 6 := by
  unfold snapCode ReaderMir.run at *
  repeat' split at hit <;> try contradiction
  all_goals simp_all [Snapshot.empty]
  subst result
  contradiction

/-- The acquired publication in the native table trace; a duplicate may inherit an older call. -/
theorem occupied_event {log : Log mc fc} (t : Table log) (endian : Bool)
    (r : Invocation t addr f (snapCode endian)) (result : SeqLock.Snapshot)
    (hit : (ReaderMir.run endian r.words).returned = some result)
    (occupied : Snapshot.empty endian result.metadata = false) :
    ∃ e ∈ log.writes, ∃ call, e.call = some call ∧ log.before e.id (r.ev 0) ∧
      (⟨result.player,result.opponent,result.metadata⟩ : Payload) = call.payload := by
  have size := occupied_size endian r.words result hit occupied
  rcases ReaderMir.origin_of_occupied_return (t.memory addr) (t.slot addr) endian r.ev r.source result
    r.steps hit occupied (r.noWrap size) with ⟨_,initial⟩ | ⟨call,index,hb,words⟩
  · have metadata := congrArg Payload.metadata initial
    change result.metadata = _ at metadata
    obtain ⟨e,member,call,write,prior,stored⟩ := (t.initial addr).occupied (metadata ▸ occupied)
    exact ⟨e,member,call,write,t.trans prior r.allocated,initial.trans stored⟩
  · obtain ⟨e,mem,write,id⟩ := t.toJournal.published addr index
    exact ⟨e,mem,call,write,id ▸ t.agrees addr _ _ hb,words⟩

structure ProbeRound {log : Log mc fc} (t : Table log) (allocation : Nat) (clusters key : Word)
    (f : Frame) (endian : Bool) (query : ProbeRetry.Key) where
  front : Invocation t ⟨allocation,(ReaderMir.probeSlots clusters key).1⟩ f (snapCode endian)
  back : Option (Invocation t ⟨allocation,(ReaderMir.probeSlots clusters key).2⟩ f (snapCode endian))
  branched : back.isSome = ReaderMir.probeSecond query (snapshot endian front.words)
  generation : BitVec 8

def ProbeRound.raw {log : Log mc fc} {t : Table log} (round : ProbeRound t allocation clusters key f endian query) :
    ProbeRetry.RawAttempt :=
  ⟨fun i => if i = 0 then snapshot endian round.front.words else round.back.bind (snapshot endian ∘ InvocationAt.words),
    round.generation⟩

theorem probe_reads {log : Log mc fc} (t : Table log) (allocation : Nat) (clusters key : Word)
    (f : Frame) (endian : Bool) (attempts : List (ProbeRound t allocation clusters key f endian query)) :
    TableReads.ProbeReadsFrom (Samples log endian f) (attempts.map ProbeRound.raw) := by
  intro a mem i
  obtain ⟨round,_,rfl⟩ := List.mem_map.mp (List.mem_of_mem_take mem)
  dsimp only [ProbeRound.raw]
  split
  · exact snapshot_read t endian round.front
  · cases round.back with
    | none => exact .unavailable
    | some r => exact snapshot_read t endian r

theorem skipped_probe_slot {log : Log mc fc} {t : Table log} (r : ProbeRound t allocation clusters key f endian query)
    (skipped : r.back = none) :
    ∃ stored data, r.raw.readSlot 0 = some (stored,data) ∧
      (data.bound = 0 ∨ stored = query) := by
  have called := r.branched
  rw [skipped] at called
  cases seen : snapshot endian r.front.words with
  | none => simp [ReaderMir.probeSecond,seen] at called
  | some pair =>
    obtain ⟨stored,data⟩ := pair
    refine ⟨stored,data,by simpa [ProbeRound.raw] using seen,?_⟩
    by_cases empty : data.bound = 0
    · exact .inl empty
    · have compare : sameBoardBits stored.1 stored.2 query.1 query.2 = true := by
        have guarded : data.bound ≠ 0#8 → sameBoardBits stored.1 stored.2 query.1 query.2 = true := by
          simpa [ReaderMir.probeSecond,seen] using called.symm
        exact guarded empty
      obtain ⟨p,o⟩ := (sameBoardBits_exact _ _ _ _).mp compare
      exact .inr (Prod.ext p o)

def lookupCode (endian : Bool) (b : PvsBoardTree.Position) (w : Nat → Word) :=
  (ReaderMir.lookupRun endian b.val.player b.val.opponent w).1

def lookupValue (endian : Bool) (b : PvsBoardTree.Position) (w : Nat → Word) :=
  (ReaderMir.lookupRun endian b.val.player b.val.opponent w).2

private def lookupActions : List ReaderMir.AtomicAction :=
  [.load .sequence .acquire,.load (.payload .player) .relaxed,.load (.payload .opponent) .relaxed,
   .load (.payload .metadata) .relaxed,.fence .acquire,.load .sequence .relaxed]

private def lookupSix : Six lookupActions :=
  ⟨rfl,rfl,rfl,rfl,(fun | .player => 0 | .opponent => 1 | .metadata => 2),by intro fld; cases fld <;> rfl⟩

private theorem lookup_path (b : PvsBoardTree.Position) (w : Nat → Word)
    (hit : lookupValue endian b w = some entry) :
    lookupCode endian b w = lookupActions ∧ w 0 &&& 1 = 0 ∧ w 1 = b.val.player ∧ w 2 = b.val.opponent ∧
      w 0 = w 5 ∧ Packing.unpack endian (w 3) = entry ∧ entry.bound ≠ 0 := by
  unfold lookupValue ReaderMir.lookupRun at hit
  split at hit <;> try contradiction
  rename_i even
  split at hit <;> try contradiction
  rename_i keys
  split at hit <;> try contradiction
  rename_i equal
  dsimp only at hit
  split at hit <;> try contradiction
  rename_i occupied
  cases hit
  have bits := (sameBoardBits_exact _ _ _ _).mp keys
  refine ⟨?_,Classical.not_not.mp even,bits.1,bits.2,equal,rfl,occupied⟩
  unfold lookupCode ReaderMir.lookupRun
  rw [ite_eq_right even,ite_eq_left keys,ite_eq_left equal]
  rfl

/-- The native six-load layout constructs the physical operand of a child
    lookup even when only the per-slot writer sequence is available. -/
theorem lookup_operand {m : Memory Nat} (slot : WriterTrace.Slot m) (endian : Bool) (b : PvsBoardTree.Position)
    (ev source : Nat → Nat) (steps : ReaderMir.Realizes m (lookupCode endian b (m.readValue ∘ ev)) ev source)
    (nowrap : (lookupCode endian b (m.readValue ∘ ev)).length = 6 → source 5 < source 0 + SeqLock.modulus) :
    TableReads.SlotLookupFrom (fun _ => True) b (lookupValue endian b (m.readValue ∘ ev)) := by
  cases result : lookupValue endian b (m.readValue ∘ ev) with
  | none => exact .miss
  | some entry =>
    obtain ⟨path,even,player,opponent,equal,packed,occupied⟩ := lookup_path b _ result
    let measured := rawSample slot ev source (nowrap (by rw [path]; rfl)) endian lookupSix (path ▸ steps)
    apply TableReads.SlotLookupFrom.hit measured entry trivial
    exact (LookupCut.readContract_proven ..).mpr ⟨even,player,opponent,equal,packed,occupied⟩

theorem lookup_read_at {log : Log mc fc} (t : Table log) (endian : Bool) (b : PvsBoardTree.Position)
    (r : InvocationAt t addr consumer (lookupCode endian b)) :
    TableReads.SlotLookupFrom (ConcurrentReplay.ReadAt log.before log.events consumer endian) b (lookupValue endian b r.words) := by
  cases actual : lookupValue endian b r.words with
  | none => exact .miss
  | some entry =>
    obtain ⟨path,even,player,opponent,equal,packed,occupied⟩ := lookup_path b r.words actual
    have size : (lookupCode endian b r.words).length = 6 := by rw [path]; rfl
    let full := r.full size
    have steps : ReaderMir.Realizes (t.memory addr) lookupActions r.ev r.source := path ▸ r.steps
    let raw := full.raw endian lookupSix steps
    apply TableReads.SlotLookupFrom.hit raw entry (full.evidence endian lookupSix steps even)
    exact (LookupCut.readContract_proven _ _ _ _ _ _ _).mpr ⟨even,player,opponent,equal,packed,occupied⟩

theorem lookup_read {log : Log mc fc} (t : Table log) (endian : Bool) (b : PvsBoardTree.Position)
    (r : Invocation t addr f (lookupCode endian b)) :
    TableReads.SlotLookupFrom (Samples log endian f) b (lookupValue endian b r.words) :=
  lookup_read_at t endian b r

/-- The accepted board and packed entry have a specific publication in the shared
    writer history. A copied slot can name a publication from its source allocation. -/
def Published {mc fc} (log : Log mc fc) (consumer : Nat) (endian : Bool) (query : ProbeRetry.Key) (entry : EntryData) : Prop :=
  entry.bound ≠ 0#8 ∧ ∃ e ∈ log.writes, ∃ call, e.call = some call ∧ log.before e.id consumer ∧
    call.payload = ⟨query.1,query.2,Packing.pack endian entry⟩

private theorem InvocationAt.before_use {log : Log mc fc} {t : Table log} (r : InvocationAt t addr consumer code)
    (size : (code r.words).length = 6) : log.before (r.ev 0) consumer :=
  t.agrees addr _ _ ((t.memory addr).sb_hb ((t.memory addr).sb_trans (ordered r.steps (by decide : 0 < 5)
    (by change 5 < (code r.words).length; omega)) (r.validated size)))

theorem snapshot_published {log : Log mc fc} {t : Table log} (r : Invocation t addr f (snapCode endian))
    (hit : snapshot endian r.words = some (query,d)) (occupied : d.bound ≠ 0#8) :
    Published log (log.identify f.use) endian query d := by
  obtain ⟨result,returned,eq⟩ := Option.map_eq_some_iff.mp hit
  cases eq
  refine ⟨occupied,?_⟩
  have used : Snapshot.empty endian result.metadata = false := by simpa [Snapshot.empty] using occupied
  obtain ⟨e,mem,call,wrote,before,words⟩ := occupied_event t endian r result returned used
  have size := occupied_size endian r.words result returned used
  exact ⟨e,mem,call,wrote,t.trans before (r.before_use size),by simpa [Packing.pack_unpack] using words.symm⟩

theorem lookup_published {log : Log mc fc} {t : Table log} (b : PvsBoardTree.Position)
    (r : InvocationAt t addr consumer (lookupCode endian b)) (hit : lookupValue endian b r.words = some d) :
    Published log consumer endian (Cutoffs.key b) d := by
  obtain ⟨path,even,player,opponent,equal,packed,occupied⟩ := lookup_path b r.words hit
  refine ⟨occupied,?_⟩
  have size : (lookupCode endian b r.words).length = 6 := by rw [path]; rfl
  let full := r.full size
  have steps : ReaderMir.Realizes (t.memory addr) lookupActions r.ev r.source := path ▸ r.steps
  let raw := full.raw endian lookupSix steps
  have physical : LookupCut.readOccupied endian raw.first raw.last raw.payload b.val.player b.val.opponent = some d :=
    (LookupCut.readContract_proven ..).mpr ⟨even,player,opponent,equal,packed,occupied⟩
  let writers := raw.slot.writers
  have twice := raw.reader.evenIndex writers even
  have short : raw.finalIndex < 2 * (raw.firstIndex / 2) + SeqLock.modulus := by rw [←twice]; exact raw.noWrap
  obtain ⟨p,o,metadata,_⟩ := LookupCut.publishedReadContract_proven writers (raw.reader.asStable writers even)
    (raw.player.asStable writers even) (raw.opponent.asStable writers even) (raw.metadata.asStable writers even) _ _ endian d short physical
  have words : raw.slot.payloadAt (raw.firstIndex / 2) = ⟨b.val.player,b.val.opponent,Packing.pack endian d⟩ := by
    simp only [WriterTrace.Slot.payloadAt,p,o,←metadata,Packing.pack_unpack]
  have finish := r.before_use size
  by_cases zero : raw.firstIndex = 0
  · rw [zero,Nat.zero_div] at words
    change (t.slot addr).payloadAt 0 = _ at words
    have used : Snapshot.empty endian ((t.slot addr).payloadAt 0).metadata = false := by
      simpa [words,Snapshot.empty,Packing.unpack_pack] using occupied
    obtain ⟨e,mem,call,wrote,before,old⟩ := (t.initial addr).occupied used
    exact ⟨e,mem,call,wrote,t.trans before (t.trans r.allocated finish),old.symm.trans words⟩
  · obtain ⟨call,atIndex,old,hb⟩ := raw.publishingCall even (Nat.pos_of_ne_zero zero)
    obtain ⟨e,mem,wrote,event⟩ := t.toJournal.published addr atIndex
    exact ⟨e,mem,call,wrote,t.trans (event ▸ t.agrees addr _ _ hb) finish,old.symm.trans words⟩

/-- A normal cluster lookup calls the second slot only after the first missed. -/
inductive Lookup {log : Log mc fc} (t : Table log) (allocation : Nat) (clusters key : Word) (f : Frame)
    (endian : Bool) (b : PvsBoardTree.Position) where
  | front (r : Invocation t (slotAt allocation clusters key 0) f (lookupCode endian b))
      (hit : (lookupValue endian b r.words).isSome = true)
  | back (r : Invocation t (slotAt allocation clusters key 0) f (lookupCode endian b))
      (miss : lookupValue endian b r.words = none)
      (s : Invocation t (slotAt allocation clusters key 1) f (lookupCode endian b))

def Lookup.pair {log : Log mc fc} {t : Table log} :
    Lookup t allocation clusters key f endian b → Option EntryData × Option EntryData
  | .front r _ => (lookupValue endian b r.words,none)
  | .back r _ s => (lookupValue endian b r.words,lookupValue endian b s.words)

structure Packet {log : Log mc fc} (t : Table log) (f : Frame)
    (code : (Nat → Word) → List ReaderMir.AtomicAction) where
  address : Address
  run : Invocation t address f code

def Packet.atSlot {log : Log mc fc} {t : Table log} (p : Packet t f code) (expected : Address) :
    Option (Invocation t expected f code) :=
  if same : p.address = expected then some (same ▸ p.run) else none

def decodeRound {log : Log mc fc} (t : Table log) (allocation : Nat) (clusters key : Word) (f : Frame)
    (endian : Bool) (query : ProbeRetry.Key) (generation : BitVec 8)
    (calls : List (Packet t f (snapCode endian))) : Option (ProbeRound t allocation clusters key f endian query) := do
  let first :: rest := calls | none
  let front ← first.atSlot ⟨allocation,(ReaderMir.probeSlots clusters key).1⟩
  if needed : ReaderMir.probeSecond query (snapshot endian front.words) then
    let [second] := rest | none
    let back ← second.atSlot ⟨allocation,(ReaderMir.probeSlots clusters key).2⟩
    return ⟨front,some back,needed.symm,generation⟩
  else if rest.isEmpty then
    return ⟨front,none,by simpa using needed,generation⟩
  else none

def rounds {log : Log mc fc} (t : Table log) (allocation : Nat) (clusters key : Word) (f : Frame)
    (endian : Bool) (query : ProbeRetry.Key) : Nat →
    List (BitVec 8 × List (Packet t f (snapCode endian))) → Option (List (ProbeRound t allocation clusters key f endian query))
  | 0, [] => some []
  | 0, _ :: _ | _+1, [] => none
  | n+1, (generation,calls) :: rest => do
    let round ← decodeRound t allocation clusters key f endian query generation calls
    if (ProbeRetry.decisive (ProbeRetry.observe query round.raw)).isSome then
      if rest.isEmpty then return [round] else none
    else
      return round :: (← rounds t allocation clusters key f endian query n rest)

def decodeProbe {log : Log mc fc} (t : Table log) (allocation : Nat) (clusters key : Word) (f : Frame)
    (endian : Bool) (query : ProbeRetry.Key) (input : List (BitVec 8 × List (Packet t f (snapCode endian)))) :
    Option (List ProbeRetry.RawAttempt) :=
  (rounds t allocation clusters key f endian query ReaderMir.probeRetries input).map (List.map ProbeRound.raw)

theorem decoded_probe_reads {log : Log mc fc} (t : Table log)
    {input : List (BitVec 8 × List (Packet t f (snapCode endian)))}
    (accepted : decodeProbe t allocation clusters key f endian query input = some attempts) :
    TableReads.ProbeReadsFrom (Samples log endian f) attempts := by
  unfold decodeProbe at accepted
  obtain ⟨actual,_,rfl⟩ := Option.map_eq_some_iff.mp accepted
  exact probe_reads t allocation clusters key f endian actual

/-- Decode precisely the nested read_for_lookup calls of this cluster lookup, in call order. -/
def decodeLookup {log : Log mc fc} (t : Table log) (allocation : Nat) (clusters key : Word)
    (f : Frame) (endian : Bool) (b : PvsBoardTree.Position) :
    List (Packet t f (lookupCode endian b)) → Option (Lookup t allocation clusters key f endian b)
  | [] => none
  | first :: rest => do
    let front ← first.atSlot (slotAt allocation clusters key 0)
    match h : lookupValue endian b front.words with
    | some _ =>
      if rest.isEmpty then return .front front (by rw [h]; rfl) else none
    | none =>
      let [second] := rest | none
      let back ← second.atSlot (slotAt allocation clusters key 1)
      return .back front h back

theorem Lookup.slots {log : Log mc fc} {t : Table log} (k : Lookup t allocation clusters key f endian b) :
    TableReads.SlotLookupFrom (Samples log endian f) b k.pair.1 ∧
      TableReads.SlotLookupFrom (Samples log endian f) b k.pair.2 := by
  cases k with
  | front r _ => exact ⟨lookup_read t endian b r,.miss⟩
  | back r _ s => exact ⟨lookup_read t endian b r,lookup_read t endian b s⟩

theorem Lookup.cluster {log : Log mc fc} {t : Table log} (k : Lookup t allocation clusters key f endian b)
    (read : Nat → Word → Word → Option EntryData)
    (first : read (slotAt allocation clusters key 0).index b.val.player b.val.opponent = k.pair.1)
    (second : k.pair.1 = none → read (slotAt allocation clusters key 1).index b.val.player b.val.opponent = k.pair.2) :
    ReaderMir.lookupCluster clusters key b.val.player b.val.opponent read = k.pair.1.or k.pair.2 := by
  change read (Clear.getClusterIdx key clusters) b.val.player b.val.opponent = _ at first
  change _ → read (Clear.getClusterIdx key clusters + 1) b.val.player b.val.opponent = _ at second
  unfold ReaderMir.lookupCluster
  rw [first]
  cases status : k.pair.1 with
  | none => exact second status
  | some _ => rfl

def childLookup {log : Log mc fc} {t : Table log} (b : PvsBoardTree.Position)
    (move : PvsBoardTree.GeneratedSquare mc b.val)
    (receipt : Lookup t allocation clusters key f endian (PvsBoardTree.play mc fc b move).val) :
    TableReads.ChildLookupFrom mc fc b (Samples log endian f) :=
  ⟨move,receipt.pair.1,receipt.pair.2,receipt.slots.1,receipt.slots.2⟩

def decodedChild {log : Log mc fc} (t : Table log) (allocation : Nat) (clusters key : Word)
    (f : Frame) (endian : Bool) (b : PvsBoardTree.Position) (move : PvsBoardTree.GeneratedSquare mc b.val)
    (records : List (Packet t f (lookupCode endian (PvsBoardTree.play mc fc b move).val))) :
    Option (TableReads.ChildLookupFrom mc fc b (Samples log endian f)) :=
  (decodeLookup t allocation clusters key f endian (PvsBoardTree.play mc fc b move).val records).map
    (childLookup b move)

end NeuralReversi.PvsParallel.FrameTableReads

import NeuralReversi.PvsParallel.NativeTableWrites.Journal

namespace NeuralReversi.PvsParallel.NativeTableWrites.Journal

open TranspositionTable
open SeqLock (Word Location Field Memory)
open FrameTableReads (Address Mark)

/-- The ordering and value facts of an executed atomic instruction. -/
def mode (m : Memory Nat) (e : Nat) (success : Bool) : Option WriterMir.Atom → Prop
  | some (.cas _ expected _ .acqRel _) => success = true → m.acquire e ∧ m.readValue e = expected
  | some (.fence .release) => m.releaseFence e
  | some (.store _ _ .release) => m.release e
  | some _ => True
  | none => False

/-- Atomic execution facts at the Rust boundary. Indices are modification-order
    indices for one location. The per-invocation premise only looks backwards,
    so a prefix can stop after any instruction of a successful writer. -/
structure Atomics (c : Calls) (endian : Bool) (rs : List Record) where
  memory : Address → Memory Nat
  distinct : ∀ a loc, ((mods c rs a loc).map (Site.event c)).Nodup
  source : ∀ a loc i r, (memory a).rf ((memory a).write loc i) r → i ≤ (mods c rs a loc).length
  modified : ∀ a loc i s, (mods c rs a loc)[i]? = some s →
    (memory a).write loc (i+1) = s.event c ∧
      writes (c s.invocation).success ((c s.invocation).program endian).1[s.pc.index]? =
        some (loc,(memory a).writeValue loc (i+1))
  rmw : ∀ a i k, (mods c rs a .sequence)[i]? = some ⟨k,.cas⟩ →
    (memory a).rf ((memory a).write .sequence i) ((c k).event 0)
  ordering : ∀ s, .writer s ∈ rs →
    mode (memory (c s.invocation).address) (s.event c) (c s.invocation).success
      ((c s.invocation).program endian).1[s.pc.index]?

structure Execution (c : Calls) (endian : Bool) (rs : List Record) extends Atomics c endian rs where
  earlier : ∀ s, .writer s ∈ rs → ∀ pc : Pc, pc.index < s.pc.index →
    .writer ⟨s.invocation,pc⟩ ∈ rs ∧
      (memory (c s.invocation).address).sb ((c s.invocation).event pc.index) (s.event c)

variable {c : Calls} {endian : Bool} {rs : List Record}

private theorem Execution.writeAt (x : Execution c endian rs) {s : Site}
    (atIndex : (mods c rs a loc)[i]? = some s) :
    (x.memory a).write loc (i+1) = s.event c ∧ (x.memory a).writeValue loc (i+1) = s.word c endian := by
  obtain ⟨whereAt,word⟩ := x.modified _ _ _ _ atIndex
  obtain ⟨_,_,success,location⟩ := mem_mods.mp (List.mem_of_getElem? atIndex)
  rw [s.modification,success,ite_eq_left rfl,location] at word
  exact ⟨whereAt,(congrArg Prod.snd (Option.some.inj word)).symm⟩

private theorem unmap {i : Nat} {xs : List α} {f : α → β} (atIndex : (xs.map f)[i]? = some v) :
    ∃ x, xs[i]? = some x ∧ f x = v := by
  simpa only [List.getElem?_map,Option.map_eq_some_iff] using atIndex

private theorem seqPc {pc : Pc} (loc : pc.location = some .sequence) : pc = .cas ∨ pc = .finish := by
  cases pc <;> simp_all [Pc.location]

private theorem fldPc {pc : Pc} (loc : pc.location = some (.payload f)) : pc = .field f := by
  cases pc <;> simp_all [Pc.location]

private theorem action_event {s : Site} (mem : s ∈ mods c rs a .sequence) :
    (s.action c endian).event = s.event c := by
  obtain ⟨_,_,_,loc⟩ := mem_mods.mp mem
  rcases s with ⟨k,pc⟩
  rcases seqPc loc with rfl | rfl <;> rfl

private theorem Execution.sequenceUnique (x : Execution c endian rs) (a : Address) :
    ((sequence c endian rs a).map WriterTrace.Action.event).Nodup := by
  rw [sequence,List.map_map]
  simp only [Function.comp_def]
  rw [List.map_congr_left (fun s mem => action_event (c := c) (rs := rs) (a := a) (endian := endian) mem)]
  exact x.distinct a .sequence

private theorem prevClaim {k : Nat} (mem : Record.writer ⟨k,.cas⟩ ∈ rs) (addr : (c k).address = a)
    (success : (c k).success = true) :
    WriterTrace.Action.claim ((c k).call endian) ∈ sequence c endian rs a :=
  List.mem_map.mpr ⟨⟨k,.cas⟩,mem_mods.mpr ⟨mem,addr,success,rfl⟩,rfl⟩

abbrev Even (m : Memory Nat) := m.writeValue .sequence 0 &&& 1 = 0

def Execution.toSequence (x : Execution c endian rs) (a : Address) (init : Even (x.memory a)) : WriterTrace.Sequence (x.memory a) where
  history := sequence c endian rs a
  unique := x.sequenceUnique a
  initialized := init
  source i r rf := by simpa [sequence] using x.source a .sequence i r rf
  modified := by
    intro i action atIndex
    obtain ⟨⟨k,pc⟩,hit,rfl⟩ := unmap atIndex
    obtain ⟨_,_,_,loc⟩ := mem_mods.mp (List.mem_of_getElem? hit)
    rcases seqPc loc with rfl | rfl <;> exact x.writeAt hit
  cas := by
    intro i call atIndex
    obtain ⟨⟨k,pc⟩,hit,same⟩ := unmap atIndex
    obtain ⟨mem,addr,success,loc⟩ := mem_mods.mp (List.mem_of_getElem? hit)
    rcases seqPc loc with rfl | rfl
    · change WriterTrace.Action.claim _ = .claim _ at same
      cases same
      have ordered := x.ordering _ mem
      simp only [mode,Site.event,Pc.index,Attempt.program,WriterMir.writeStable,success,↓reduceIte,
        List.getElem?_cons_zero,addr,forall_const] at ordered
      exact ⟨(c k).expected.property,ordered.1,x.rmw _ _ _ hit,ordered.2⟩
    · contradiction
  publish := by
    intro i call atIndex
    obtain ⟨⟨k,pc⟩,hit,same⟩ := unmap atIndex
    obtain ⟨mem,addr,success,loc⟩ := mem_mods.mp (List.mem_of_getElem? hit)
    rcases seqPc loc with rfl | rfl
    · contradiction
    · change WriterTrace.Action.publish _ = .publish _ at same
      cases same
      have prior := x.earlier _ mem .cas (by simp [Pc.index])
      have ordered := x.ordering _ mem
      simp [Site.event,Pc.index,Attempt.program,WriterMir.writeStable,mode,success,addr] at ordered
      exact ⟨prevClaim prior.1 addr success,addr ▸ prior.2,ordered⟩

private theorem Execution.fieldUnique (x : Execution c endian rs) (a : Address) (f : Field) :
    (fields c endian rs a f).Nodup := by
  have unique := x.distinct a (.payload f)
  have same : ((fields c endian rs a f).map (fun call => call.store f)) =
      (mods c rs a (.payload f)).map (Site.event c) := by
    rw [fields,List.map_map]
    apply List.map_congr_left
    intro s mem
    obtain ⟨_,_,_,loc⟩ := mem_mods.mp mem
    obtain ⟨k,pc⟩ := s
    cases fldPc loc
    rfl
  rw [← same] at unique
  exact List.Pairwise.of_map _ (fun _ _ (ne : _ ≠ _) eq => ne (congrArg _ eq)) unique

def Execution.toFields (x : Execution c endian rs) (a : Address) (init : Even (x.memory a)) (f : Field) :
    WriterTrace.FieldWrites (x.toSequence a init) f where
  history := fields c endian rs a f
  unique := x.fieldUnique a f
  source i r rf := by simpa [fields] using x.source a (.payload f) i r rf
  store := by
    intro i call atIndex
    obtain ⟨⟨k,pc⟩,hit,rfl⟩ := unmap atIndex
    obtain ⟨mem,addr,success,loc⟩ := mem_mods.mp (List.mem_of_getElem? hit)
    cases fldPc loc
    have toFence := x.earlier _ mem .fence (by cases f <;> simp [Pc.index])
    have fromCas := x.earlier _ toFence.1 .cas (by simp [Pc.index])
    have fence := x.ordering _ toFence.1
    simp [Site.event,Pc.index,Attempt.program,WriterMir.writeStable,mode,success,addr] at fence
    exact ⟨(x.writeAt hit).1,(x.writeAt hit).2,prevClaim fromCas.1 addr success,fence,
      addr ▸ fromCas.2,addr ▸ toFence.2⟩
  finish := by
    intro call member
    obtain ⟨⟨k,pc⟩,present,same⟩ := List.mem_map.mp member
    obtain ⟨mem,addr,success,loc⟩ := mem_mods.mp present
    rcases seqPc loc with rfl | rfl
    · contradiction
    · change WriterTrace.Action.publish _ = .publish _ at same
      cases same
      have previous := x.earlier _ mem (.field f) (by cases f <;> simp [Pc.index])
      exact ⟨List.mem_map.mpr ⟨⟨k,.field f⟩,mem_mods.mpr ⟨previous.1,addr,success,rfl⟩,rfl⟩,
        addr ▸ previous.2⟩

def Execution.slot (x : Execution c endian rs) (a : Address) (init : Even (x.memory a)) : WriterTrace.Slot (x.memory a) :=
  ⟨x.toSequence a init,x.toFields a init⟩

/-- Both histories consumed by the proof of the public score come from the same
    physical stream. Their agreement does not depend on packet acceptance. -/
def Execution.journal (x : Execution c endian rs) (log : FrameReturns.Log mc fc)
    (init : ∀ a, Even (x.memory a))
    (recorded : log.writes = NativeFrames.tableEvents (rs.filterMap (frameRecord c endian)))
    (agrees : ∀ a i j, (x.memory a).hb i j → log.before i j)
    (trans : ∀ {i j k}, log.before i j → log.before j k → log.before i k) :
    FrameTableReads.Journal log where
  trace := rs.filterMap (mark c endian)
  recorded := (marks_table c endian rs).trans recorded.symm
  memory := x.memory
  slot a := x.slot a (init a)
  journal := marks_sequence c endian rs
  agrees := agrees
  trans := trans

theorem Execution.published (x : Execution c endian rs) (k : Nat) (init : Even (x.memory (c k).address))
    (issued : Record.writer ⟨k,.finish⟩ ∈ rs) (success : (c k).success = true) :
    WriterTrace.Action.publish ((c k).call endian) ∈ (x.slot (c k).address init).history :=
  List.mem_map.mpr ⟨⟨k,.finish⟩,mem_mods.mpr ⟨issued,rfl,success,rfl⟩,rfl⟩

/-- An occupied snapshot uses either the allocation image or a concrete writer's
    last instruction. This refers to its invocation, not just its word value. -/
theorem Execution.origin (x : Execution c endian rs) (a : Address) (init : Even (x.memory a)) (ev source : Nat → Nat)
    (steps : ReaderMir.Realizes (x.memory a) (ReaderMir.run endian ((x.memory a).readValue ∘ ev)).actions ev source)
    (returned : (ReaderMir.run endian ((x.memory a).readValue ∘ ev)).returned = some result)
    (occupied : Snapshot.empty endian result.metadata = false) (noWrap : source 5 < source 0 + SeqLock.modulus) :
    let words : SeqLock.Payload := ⟨result.player,result.opponent,result.metadata⟩
    (source 0 = 0 ∧ words = (x.slot a init).payloadAt 0) ∨
      ∃ k, Record.writer ⟨k,.finish⟩ ∈ rs ∧ (c k).address = a ∧ (c k).success = true ∧
        (mods c rs a .sequence)[source 0 - 1]? = some ⟨k,.finish⟩ ∧
        (x.memory a).hb ((c k).event 5) (ev 0) ∧ words = ((c k).call endian).payload := by
  rcases ReaderMir.origin_of_occupied_return _ (x.slot a init) endian ev source result steps returned occupied noWrap with
    initial | ⟨call,atIndex,hb,words⟩
  · exact .inl initial
  · obtain ⟨⟨k,pc⟩,hit,same⟩ := unmap atIndex
    obtain ⟨issued,addr,success,loc⟩ := mem_mods.mp (List.mem_of_getElem? hit)
    rcases seqPc loc with rfl | rfl
    · contradiction
    · change WriterTrace.Action.publish _ = .publish _ at same
      cases same
      exact .inr ⟨k,issued,addr,success,hit,hb,words⟩

end NeuralReversi.PvsParallel.NativeTableWrites.Journal

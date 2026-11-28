module
public import NeuralReversi.PvsParallel.Interleaving.Nested
public import NeuralReversi.PvsParallel.Interleaving.SeedHistory

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.JournalResident

open TranspositionTable
open FrameTableReads (Address)
open NativeTableWrites.Journal (Calls Record Even Execution mods)
open SeqLock (Word Payload Field Location)

/-- All successful PVS releases and their score-producing histories for the
    preceding table lifetime or search prefix. -/
structure Session (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (endian : Bool) where
  calls : Calls
  records : List Record
  memory : NativeTableWrites.Journal.Execution calls endian records
  init : ∀ a, Even (memory.memory a)
  before : Nat → Nat → Prop
  frames : List CausalTable.Frame
  level : CausalTable.Frame → Selectivity
  levels : SeedHistory.Prefix mc fc memory init frames level before
  exactFrames : List CausalTable.Frame
  exactRun : CausalTable.Prefix memory init exactFrames before (Nested.Body mc fc)

def Valid (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (endian : Bool) (p : Payload) :=
  Table.Safe mc fc endian p ∧ FiniteTable.Words endian p ∧ SeedTable.Words endian p

def Session.initial (s : Session mc fc endian) (a : Address) := (s.memory.slot a (s.init a)).payloadAt 0

/-- Last modification of this location in the preceding prefix. Zero is the
    initial write and each subsequent position is a concrete journal store. -/
def Session.lastWord (s : Session mc fc endian) (a : Address) (loc : Location) :=
  (s.memory.memory a).writeValue loc (mods s.calls s.records a loc).length

def Session.finalImage (s : Session mc fc endian) (a : Address) : Payload :=
  ⟨s.lastWord a (.payload .player),s.lastWord a (.payload .opponent),s.lastWord a (.payload .metadata)⟩

def Session.released (s : Session mc fc endian) (a : Address) := s.lastWord a .sequence &&& 1 = 0

private theorem Session.release_valid (s : Session mc fc endian)
    (initial : ∀ a, Valid mc fc endian (s.initial a)) (a : Address) (n k : Nat)
    (finish : (mods s.calls s.records a .sequence)[n]? = some ⟨k,.finish⟩) :
    Valid mc fc endian ((s.calls k).call endian).payload := by
  let prior := Nested.toPrefix s.exactRun
  have frame := (CausalTable.certified s.memory s.init prior RootCausal.preserves (fun a => (initial a).1)).1
  have words := SeedHistory.stores s.levels (fun a => (initial a).2.1)
  apply Table.publication_invariant s.memory a (s.init a) (Valid mc fc endian) (initial a) _ n k finish
  intro k listed addr success previous
  subst a
  have scores : CausalTable.Correct mc fc (CausalTable.chosen s.before s.exactFrames ((s.calls k).event 5)) := by
    intro b pv alpha beta v win narrow chosen
    obtain ⟨f,mem,_,rfl,rfl,rfl,rfl,rfl⟩ := chosen
    exact frame f mem win narrow
  exact ⟨Table.Produced.safe ((prior.store k listed success).mapInputs fun _ _ source => source.safe scores)
      (fun p seen => (previous p seen).1),
    FiniteTable.written (s.levels.finiteStores.operands k listed success) (fun p seen => (previous p seen).2.1),
    FiniteTable.fieldWritten (words.operands k listed success) (fun _ _ range => range) (fun p seen => (previous p seen).2.2)⟩

private theorem field_length {mem : SeqLock.Memory Nat} (slot : WriterTrace.Slot mem) (field : Field)
    (even : slot.history.length = 2*n) : (slot.fields field).history.length = n := by
  have upper : (slot.fields field).history.length ≤ n := by
    apply Nat.le_of_not_gt
    intro beyond
    have inside : n < (slot.fields field).history.length := by omega
    have atWrite := List.getElem?_eq_getElem (l := (slot.fields field).history) inside
    have claim := (slot.fields field).aligned atWrite
    have bound := (List.getElem?_eq_some_iff.mp claim).1
    omega
  cases n with
  | zero => omega
  | succ k =>
    obtain ⟨c,done_,_⟩ := slot.publishedPayload (k+1) (by omega) (by omega)
    have last := (slot.fields field).atPublication (by omega : 0 < k+1) done_
    have bound := (List.getElem?_eq_some_iff.mp last).1
    omega

theorem Session.final_valid (s : Session mc fc endian) (initial : ∀ a, Valid mc fc endian (s.initial a))
    (a : Address) (released : s.released a) : Valid mc fc endian (s.finalImage a) := by
  let slot := s.memory.slot a (s.init a)
  let n := slot.history.length / 2
  obtain ⟨_,_,_,word⟩ := slot.order.ordered
  have closed : (s.memory.memory a).writeValue .sequence slot.history.length &&& 1 = 0 := by
    simpa [Session.released,Session.lastWord,slot,NativeTableWrites.Journal.Execution.slot,
      NativeTableWrites.Journal.Execution.toSequence,NativeTableWrites.Journal.sequence] using released
  rw [word] at closed
  have parity := (SeqLock.seqWord_even_iff _).mp closed
  have even : slot.history.length = 2*n := by dsimp only [n]; omega
  have fields (f : Field) : (mods s.calls s.records a (.payload f)).length = n := by
    simpa [slot,NativeTableWrites.Journal.Execution.slot,NativeTableWrites.Journal.Execution.toFields,
      NativeTableWrites.Journal.fields] using field_length slot f even
  have finalWords : s.finalImage a = slot.payloadAt n := by
    simp only [Session.finalImage,Session.lastWord,fields,WriterTrace.Slot.payloadAt]
  rw [finalWords]
  by_cases zero : n = 0
  · rw [zero]
    exact initial a
  · obtain ⟨call,published,source⟩ := slot.publishedPayload n (by omega) (by omega)
    rw [source]
    change ((mods s.calls s.records a .sequence).map (NativeTableWrites.Journal.Site.action s.calls endian))[2*n-1]? =
      some (.publish call) at published
    obtain ⟨⟨k,pc⟩,hit,act⟩ := Option.map_eq_some_iff.mp (List.getElem?_map .. ▸ published)
    have loc := (NativeTableWrites.Journal.mem_mods.mp (List.mem_of_getElem? hit)).2.2.2
    cases pc <;> simp [NativeTableWrites.Journal.Pc.location,NativeTableWrites.Journal.Site.action] at act loc
    cases act
    exact s.release_valid initial a _ _ hit

/-- A helper can stop without returning a search result. Handoff only needs the
    writers that actually acquired a slot to have performed their release. -/
def Session.drained (s : Session mc fc endian) := ∀ k, Record.writer ⟨k,.cas⟩ ∈ s.records →
  (s.calls k).success = true → Record.writer ⟨k,.finish⟩ ∈ s.records

def finished (k : Nat) : Record → Bool
  | .writer ⟨j,.finish⟩ => k == j
  | _ => false

def checkDrained (success : Nat → Bool) (rs : List Record) :=
  rs.all fun
    | .writer ⟨k,.cas⟩ => !success k || rs.any (finished k)
    | _ => true

private theorem finish_mem (rs : List Record) (k : Nat) :
    rs.any (finished k) = true ↔ Record.writer ⟨k,.finish⟩ ∈ rs := by
  constructor
  · intro found
    obtain ⟨rec,mem,eq⟩ := List.any_eq_true.mp found
    cases rec with
    | localCall => contradiction
    | writer site =>
      obtain ⟨j,pc⟩ := site
      cases pc <;> simp [finished] at eq
      exact eq ▸ mem
  · intro mem
    exact List.any_eq_true.mpr ⟨_,mem,by simp [finished]⟩

theorem Session.checked (s : Session mc fc endian) :
    checkDrained (fun k => (s.calls k).success) s.records = true ↔ s.drained := by
  constructor
  · intro complete k cas success
    have found := List.all_eq_true.mp complete _ cas
    apply (finish_mem _ _).mp
    simpa only [success,Bool.not_true,Bool.false_or] using found
  · intro complete
    apply List.all_eq_true.mpr
    intro rec mem
    cases rec with
    | localCall => rfl
    | writer site =>
      obtain ⟨k,pc⟩ := site
      cases pc <;> try rfl
      cases yes : (s.calls k).success with
      | false => simp [yes]
      | true => exact (show ((!true || s.records.any (finished k)) = true) from
          (finish_mem _ _).mpr (complete k mem yes)) |> (yes ▸ ·)

theorem Session.drained_released (s : Session mc fc endian) (drained : s.drained) (a : Address) : s.released a := by
  let slot := s.memory.slot a (s.init a)
  let len := slot.history.length
  have parity : len % 2 = 0 := by
    by_cases even : len % 2 = 0
    · exact even
    let epoch := len/2+1
    have last : 2*epoch-2+1 = len := by dsimp only [epoch]; omega
    have begun : 2*epoch-1 ≤ slot.history.length := by change _ ≤ len; omega
    obtain ⟨writer,claim,_⟩ := slot.order.callAt epoch (by dsimp only [epoch]; omega) begun
    change ((mods s.calls s.records a .sequence).map (NativeTableWrites.Journal.Site.action s.calls endian))[2*epoch-2]? =
      some (.claim writer) at claim
    obtain ⟨⟨k,pc⟩,record,action⟩ := Option.map_eq_some_iff.mp (List.getElem?_map .. ▸ claim)
    obtain ⟨issued,address,success,loc⟩ := NativeTableWrites.Journal.mem_mods.mp (List.mem_of_getElem? record)
    cases pc <;> simp [NativeTableWrites.Journal.Pc.location,NativeTableWrites.Journal.Site.action] at action loc
    cases action
    have closed := drained k issued success
    have finished : WriterTrace.Action.publish ((s.calls k).call endian) ∈ slot.history :=
      List.mem_map.mpr ⟨⟨k,.finish⟩,NativeTableWrites.Journal.mem_mods.mpr ⟨closed,address,success,rfl⟩,rfl⟩
    obtain ⟨j,atFinish⟩ := List.mem_iff_getElem?.mp finished
    have localOrder := (slot.toSequence.publish j _ atFinish).2.1
    have left := (slot.modified _ _ claim).1
    have right := (slot.modified _ _ atFinish).1
    have forward := (s.memory.memory a).write_write (loc := .sequence) (i := 2*epoch-2+1) (j := j+1)
      (by rw [left,right]; exact (s.memory.memory a).sb_hb localOrder)
    have size : j < len := (List.getElem?_eq_some_iff.mp atFinish).1
    omega
  obtain ⟨_,_,_,last⟩ := slot.order.ordered
  have even : (s.memory.memory a).writeValue .sequence slot.history.length &&& 1 = 0 := by
    rw [last]
    apply (SeqLock.seqWord_even_iff _).mpr
    change (2 * _ + len) % 2 = 0
    omega
  simpa [Session.released,Session.lastWord,slot,NativeTableWrites.Journal.Execution.slot,
    NativeTableWrites.Journal.Execution.toSequence,NativeTableWrites.Journal.sequence] using even

/-- Two consecutive sessions use the same allocation and slot indices. The next
    session starts at the final physical modification of each of its four words. -/
structure Handoff (s : Session mc fc endian) (next : NativeTableWrites.Journal.Execution c endian rs) : Prop where
  drained : s.drained
  image : ∀ a loc, (next.memory a).writeValue loc 0 = s.lastWord a loc

theorem Handoff.even {s : Session mc fc endian} {next : NativeTableWrites.Journal.Execution c endian rs}
    (link : Handoff s next) (a : Address) : Even (next.memory a) := by
  unfold Even
  rw [link.image]
  exact s.drained_released link.drained a

theorem Handoff.words {s : Session mc fc endian} {next : NativeTableWrites.Journal.Execution c endian rs}
    (link : Handoff s next) (init : ∀ a, Even (next.memory a)) (a : Address) :
    (next.slot a (init a)).payloadAt 0 = s.finalImage a := by
  simp [WriterTrace.Slot.payloadAt,Session.finalImage,link.image]

/-- A finite sequence of handoffs has its actual four-word image at each
    boundary, including the generation word even when the payload is unchanged. -/
inductive Chain (first : Session mc fc endian) : Session mc fc endian → Prop where
  | start : Chain first first
  | next (earlier : Chain first s) (joined : Handoff s nextSession.memory) : Chain first nextSession

example : checkDrained (fun _ => true) [.writer ⟨4,.cas⟩,.writer ⟨4,.field .player⟩] = false ∧
    checkDrained (fun _ => true) [.writer ⟨4,.cas⟩,.writer ⟨4,.finish⟩] = true ∧
    checkDrained (fun _ => false) [.writer ⟨4,.cas⟩] = true := by decide

end NeuralReversi.PvsParallel.Interleaving.JournalResident

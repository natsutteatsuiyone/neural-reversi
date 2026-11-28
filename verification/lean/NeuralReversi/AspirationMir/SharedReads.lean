module
public import NeuralReversi.AspirationMir.NativeExecution
public import NeuralReversi.PvsParallel.FrameTableReads.Schedule
public import NeuralReversi.PvsParallel.FrameTableReads.WriterWindow

@[expose] public section

namespace NeuralReversi.AspirationMir.NativeExecution

open PvsParallel
open TranspositionTable
open NativeTableWrites.Journal (Calls Record)

variable {pool : EndgameSearch.CacheHistory.Pool} {out : NativeFrameCache.Checked pool}

/-- A nonempty shared-table hit carries the exact argument of a completed
    seqlock_write on this trace, including the original producer of a copied slot. -/
def StoredBefore (calls : Calls) (records : List Record) (before : Nat → Nat → Prop) (consumer : Nat)
    (query : ProbeRetry.Key) (d : EntryData) : Prop :=
  ∃ k owner b, Record.writer ⟨k,.finish⟩ ∈ records ∧ (calls k).success = true ∧
    (calls k).origin = .store owner b d ∧ (b.val.player,b.val.opponent) = query ∧ before ((calls k).event 5) consumer

private theorem publisher (calls : Calls) (records : List Record)
    (captured : NativeFrameCache.decode pool mc fc inputs values
      (records.filterMap (NativeTableWrites.Journal.frameRecord calls endian)) = some out)
    (source : FrameTableReads.Published (trace out mc fc identify before seq) consumer endian query d) :
    StoredBefore calls records before consumer query d := by
  obtain ⟨occupied,e,member,call,wrote,prior,words⟩ := source
  have table := (NativeFrames.decode_keeps (NativeFrameCache.decode_physical captured)).1
  change e ∈ out.physical.table at member
  obtain ⟨k,finish,success,eq⟩ := NativeTableWrites.Journal.publishing_site calls endian records (table ▸ member)
  rw [eq,(calls k).published_call] at wrote
  cases Option.some.inj wrote
  obtain ⟨owner,b,origin,board⟩ := (calls k).stored_payload endian query d words occupied
  rw [eq] at prior
  exact ⟨k,owner,b,finish,success,origin,board,prior⟩

open FrameTableReads.Operations.Execution (Caller Turn dispatch journal selected budget Atomics NoWrap)

theorem shortSlots_of_journal_length (c : FrameTableReads.Operations.Execution.Call) (writers : Calls)
    (records : List Record) (small : records.length < SeqLock.modulus) :
    c.shortSlotHistories (fun a => (NativeTableWrites.Journal.mods writers records a .sequence).length) := by
  apply List.all_eq_true.mpr
  intro op _
  split
  · exact decide_eq_true (Nat.lt_of_le_of_lt (List.length_filterMap_le ..) small)
  · rfl

/-- Count sequence modifications within a slice of the shared writer journal.
    The bounds refer to actual write events at its ends, independently of which
    modifications either load in the reader observed. -/
theorem noWrap_of_journal_window (c : FrameTableReads.Operations.Execution.Call) (writers : Calls)
    (records : List Record) (start count : Nat) {t : FrameTableReads.Table log} (physical : Atomics t consumer c.records) :
    let preceding := fun a => (NativeTableWrites.Journal.mods writers (records.take start) a .sequence).length
    let amount := fun a => (NativeTableWrites.Journal.mods writers ((records.drop start).take count) a .sequence).length +
      (NativeTableWrites.Journal.mods writers ((records.drop start).drop count) a .sequence).findIdx NativeTableWrites.Journal.isClaim
    c.shortSlotHistories amount →
    (∀ r, FrameTableReads.Operations.Op.slot r (.load .sequence .relaxed) ∈ c.ops →
      FrameTableReads.Operations.Execution.Boundaries (t.slot r.address).toSequence consumer c.records
        (preceding r.address) (preceding r.address + amount r.address)) →
    NoWrap c.ops c.records := by
  intro preceding amount short edges
  apply physical.noWrap_of_window c.instructions
  intro r issued
  have limit : amount r.address < SeqLock.modulus := of_decide_eq_true (List.all_eq_true.mp short _ issued)
  exact ⟨_,_,edges r issued,by omega⟩

/-- Reader and writer instructions can alternate at every step. The schedule
    supplies the search journal and each completed reader's entire instruction
    stream. Its caller provides only per-atomic memory facts and the no-wrap bound. -/
theorem scheduled_store (readers : Caller → FrameTableReads.Operations.Execution.Call) (writers : Calls) (turns : List Turn)
    (f : FrameReturns.Frame) (opening : Nat) (t : FrameTableReads.Table (trace out mc fc identify before seq)) :
    let who : Caller := ⟨f.worker,opening⟩
    let stream := dispatch readers (fun _ => 0) turns
    NativeFrameCache.decode pool mc fc inputs values
      ((journal stream).filterMap (NativeTableWrites.Journal.frameRecord writers (readers who).endian)) = some out →
    (readers who).ops.length ≤ budget who turns →
    Atomics t (identify f.use) (selected who stream) →
    NoWrap (readers who).ops (selected who stream) →
    (readers who).returned = some d →
    StoredBefore writers (turns.filterMap (fun | .search r => some r | .reader _ => none))
      before (identify f.use) (readers who).method.query d := by
  intro who stream captured finished physical short hit
  dsimp only [stream] at captured physical short
  rw [FrameTableReads.Operations.Execution.dispatched_journal] at captured
  have reader := FrameTableReads.Operations.Execution.complete readers turns who finished
  rw [reader] at physical short
  exact publisher writers _ captured ((readers who).published (f:=f) physical short hit)

/-- The reader observes the table built by the very same interleaved writer
    instructions. Count only successful sequence modifications to slots this
    call validates. Unrelated slots and unsuccessful CAS attempts do not count. -/
theorem scheduled_store_of_slot_lengths
    (readers : Caller → FrameTableReads.Operations.Execution.Call) (writers : Calls) (turns : List Turn)
    (f : FrameReturns.Frame) (opening : Nat) :
    let who : Caller := ⟨f.worker,opening⟩
    let c := readers who
    let records := turns.filterMap (fun | .search r => some r | .reader _ => none)
    let stream := dispatch readers (fun _ => 0) turns
    ∀ (x : NativeTableWrites.Journal.Execution writers c.endian records)
      (captured : NativeFrameCache.decode pool mc fc inputs values
        (records.filterMap (NativeTableWrites.Journal.frameRecord writers c.endian)) = some out)
      (ready : FrameTableReads.Address → Nat) (starts : ∀ a, NativeTableWrites.Journal.Start x before ready a)
      (agrees : ∀ a i j, (x.memory a).hb i j → before i j)
      (trans : ∀ {i j k}, before i j → before j k → before i k),
    let t := tableOfWrites x captured identify before seq ready starts agrees trans
    c.shortSlotHistories (fun a => (NativeTableWrites.Journal.mods writers records a .sequence).length) →
    c.ops.length ≤ budget who turns → Atomics t (identify f.use) (selected who stream) →
    c.returned = some d → StoredBefore writers records before (identify f.use) c.method.query d := by
  intro who c records stream x captured ready starts agrees trans t lengths finished physical hit
  apply scheduled_store (inputs := inputs) (values := values) readers writers turns f opening t _ finished physical _ hit
  · rwa [FrameTableReads.Operations.Execution.dispatched_journal]
  · have completed := FrameTableReads.Operations.Execution.complete readers turns who finished
    change NoWrap c.ops (selected who stream)
    change selected who stream = c.records at completed
    rw [completed] at physical ⊢
    apply noWrap_of_journal_window c writers records 0 records.length physical
    · simpa [NativeTableWrites.Journal.mods] using lengths
    · intro r _
      simpa [t,tableOfWrites,NativeTableWrites.Journal.Execution.table,NativeTableWrites.Journal.Execution.journal,
        NativeTableWrites.Journal.Execution.slot,NativeTableWrites.Journal.mods] using x.readerBoundaries r.address (NativeTableWrites.Journal.Start.even starts _) c.records
        (identify f.use) [] records [] (by simp) (by simp [NativeTableWrites.Journal.mods])
        (by simp [NativeTableWrites.Journal.mods])

end NeuralReversi.AspirationMir.NativeExecution

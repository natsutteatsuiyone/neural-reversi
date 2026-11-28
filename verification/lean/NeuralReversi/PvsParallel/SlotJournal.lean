module
public import NeuralReversi.PvsParallel.TableReads

@[expose] public section

namespace NeuralReversi.PvsParallel.SlotJournal

open PvsBoardTree (Position)
open TranspositionTable
open Cutoffs (History SlotPublications RawSample ReadSample Lookup)
open ExactTableHistory (Admission Publications)
open WriterTrace (Call Slot)

def KnownPayload (history : History) (bigEndian : Bool) (payload : SeqLock.Payload) : Prop :=
  let entry := Packing.unpack bigEndian payload.metadata
  entry.bound ≠ 0 →
  ∃ board : Position, (board, entry) ∈ history ∧
    payload.player = board.val.player ∧ payload.opponent = board.val.opponent

theorem KnownPayload.cleared (history : History) (bigEndian : Bool) :
    KnownPayload history bigEndian ⟨0, 0, 0⟩ := by
  cases bigEndian <;> intro occupied <;> exact (occupied rfl).elim

theorem KnownPayload.fromSlot {memory : SeqLock.Memory Nat} {slot : Slot memory}
    (known : SlotPublications history slot bigEndian) (epoch : Nat)
    (published : 2 * epoch ≤ slot.history.length) :
    KnownPayload history bigEndian (slot.payloadAt epoch) :=
  known epoch published

private theorem KnownPayload.weaken (known : KnownPayload history bigEndian payload)
    (subset : ∀ pair ∈ history, pair ∈ extended) : KnownPayload extended bigEndian payload := by
  intro occupied
  obtain ⟨board, member, player, opponent⟩ := known occupied
  exact ⟨board, subset _ member, player, opponent⟩

def packed (bigEndian : Bool) (board : Position) (entry : EntryData) : SeqLock.Payload :=
  ⟨board.val.player, board.val.opponent, Packing.pack bigEndian entry⟩

inductive Journal (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (bigEndian : Bool) :
    List (Call Nat) → History → Prop
  | start : Publications mc fc history → Journal mc fc bigEndian [] history
  | clear (call : Call Nat) : Journal mc fc bigEndian calls history →
      call.payload = ⟨0, 0, 0⟩ → Journal mc fc bigEndian (calls ++ [call]) history
  | store (call : Call Nat) (board : Position) (entry : EntryData) :
      Journal mc fc bigEndian calls history → Admission mc fc history board entry →
      call.payload = packed bigEndian board entry →
      Journal mc fc bigEndian (calls ++ [call]) (history ++ [(board, entry)])

theorem Journal.publications (journal : Journal mc fc bigEndian calls history) :
    Publications mc fc history := by
  induction journal with
  | start valid => exact valid
  | clear _ _ _ prior => exact prior
  | store _ _ _ _ admitted _ prior => exact .append prior admitted

theorem Journal.payloads (journal : Journal mc fc bigEndian calls history) :
    ∀ call ∈ calls, KnownPayload history bigEndian call.payload := by
  induction journal with
  | start _ => simp
  | clear call _ zero ih =>
    intro write present
    rcases List.mem_append.mp present with old | new
    · exact ih write old
    · rw [List.mem_singleton.mp new, zero]
      exact KnownPayload.cleared _ _
  | store call board entry _ _ wrote ih =>
    intro write present
    rcases List.mem_append.mp present with old | new
    · exact (ih write old).weaken (fun _ mem => List.mem_append_left _ mem)
    · rw [List.mem_singleton.mp new, wrote]
      intro _
      exact ⟨board, by simp [packed, Packing.unpack_pack], rfl, rfl⟩

structure SlotCoverage {memory : SeqLock.Memory Nat} (slot : Slot memory) (calls : List (Call Nat))
    (history : History) (bigEndian : Bool) : Prop where
  origin : KnownPayload history bigEndian (slot.payloadAt 0)
  publication : ∀ epoch call, 0 < epoch → slot.history[2 * epoch - 1]? = some (.publish call) →
    call ∈ calls

def Contract : Prop :=
  ∀ mc fc bigEndian calls history (memory : SeqLock.Memory Nat) (slot : Slot memory),
    Journal mc fc bigEndian calls history → SlotCoverage slot calls history bigEndian →
    Publications mc fc history ∧ SlotPublications history slot bigEndian

theorem contract_proven : Contract := by
  intro mc fc endian calls history memory slot journal recorded
  refine ⟨journal.publications, ?_⟩
  intro epoch stable
  change KnownPayload history endian (slot.payloadAt epoch)
  by_cases initial : epoch = 0
  · exact initial ▸ recorded.origin
  · have positive := Nat.pos_of_ne_zero initial
    obtain ⟨call, index, words⟩ := slot.publishedPayload epoch positive stable
    rw [words]
    exact journal.payloads call (recorded.publication epoch call positive index)

def checkedSample (raw : RawSample) (journal : Journal mc fc raw.bigEndian calls history)
    (recorded : SlotCoverage raw.slot calls history raw.bigEndian) : ReadSample history :=
  ⟨raw, by
    have known := raw.slot.source _ _ raw.reader.first_rf
    have stable : 2 * (raw.firstIndex / 2) ≤ raw.slot.history.length := by omega
    exact (contract_proven mc fc raw.bigEndian calls history raw.memory raw.slot journal recorded).2
      (raw.firstIndex / 2) stable⟩

theorem copiedOrigin {sourceMemory destMemory : SeqLock.Memory Nat}
    {source : Slot sourceMemory} {dest : Slot destMemory}
    (known : SlotPublications history source bigEndian) (epoch : Nat)
    (published : 2 * epoch ≤ source.history.length)
    (copied : dest.payloadAt 0 = source.payloadAt epoch) :
    KnownPayload history bigEndian (dest.payloadAt 0) := by
  rw [copied]
  exact KnownPayload.fromSlot known epoch published

def LookupContract : Prop :=
  ∀ mc fc calls history (raw : RawSample) board entry,
    Journal mc fc raw.bigEndian calls history →
    SlotCoverage raw.slot calls history raw.bigEndian →
    LookupCut.readOccupied raw.bigEndian raw.first raw.last raw.payload
      board.val.player board.val.opponent = some entry →
    (board, entry) ∈ history ∧
      (Cutoffs.ExactEndgame entry → BoundSound entry (PvsBoardTree.minimax mc fc board))

theorem lookupContract_proven : LookupContract := by
  intro mc fc calls history raw board entry journal covered accepted
  have observed : TableReads.SlotLookup history board (some entry) :=
    .hit raw entry (checkedSample raw journal covered).published accepted
  obtain ⟨lookup⟩ := observed.published
  exact ⟨lookup.source, journal.publications.sound _ _ lookup.source⟩

def SnapshotContract : Prop :=
  ∀ mc fc calls history (raw : RawSample) snapshot,
    Journal mc fc raw.bigEndian calls history →
    SlotCoverage raw.slot calls history raw.bigEndian →
    Snapshot.read raw.bigEndian raw.first raw.last raw.payload = some snapshot →
    TableReads.SnapshotRead history
      (some ((snapshot.player, snapshot.opponent), Packing.unpack raw.bigEndian snapshot.metadata))

theorem snapshotContract_proven : SnapshotContract := by
  intro mc fc calls history raw snapshot journal covered accepted
  exact .sampled raw snapshot (checkedSample raw journal covered).published accepted

theorem penaltyAdmission (source : Lookup history board expected)
    (observed : Option SeqLock.Snapshot) (current : SeqLock.Word)
    (written : Penalty.tryPenalize bigEndian board.val.player board.val.opponent expected
      observed current = some result) : Admission mc fc history board result := by
  obtain ⟨positive, _, _, _, _, _, _, rfl⟩ :=
    (Penalty.writeContract_proven bigEndian board.val.player board.val.opponent expected observed current result).mp
      written
  exact .penalized board expected source.source positive

end NeuralReversi.PvsParallel.SlotJournal

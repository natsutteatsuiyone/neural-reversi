module
public import NeuralReversi.TranspositionTable.Packing

@[expose] public section

namespace NeuralReversi.TranspositionTable.Snapshot

open SeqLock

def empty (bigEndian : Bool) (raw : Word) : Bool :=
  (Packing.unpack bigEndian raw).bound == 0#8

def read (bigEndian : Bool) (first last : Word) (sample : Payload) : Option SeqLock.Snapshot :=
  snapshotDecision first last sample (empty bigEndian)

def ReadContract : Prop :=
  ∀ bigEndian first last sample result,
    read bigEndian first last sample = some result ↔
      first &&& 1 = 0 ∧ first = last ∧ result.metadata = sample.metadata ∧
      result.sequence = first ∧
      (if empty bigEndian sample.metadata then result.player = 0 ∧ result.opponent = 0
       else result.player = sample.player ∧ result.opponent = sample.opponent)

theorem readContract_proven : ReadContract := by
  intro bigEndian first last sample result
  rw [read, snapshot_success_iff]
  cases result
  cases hit : empty bigEndian sample.metadata <;>
    simp [snapshotValue, hit, and_comm, and_left_comm, and_assoc]

def OccupiedPublicationContract : Prop :=
  ∀ {Event : Type} {memory : Memory Event} {epoch finalIndex : Nat}
    (writers : Writers memory) (reader : Reader memory epoch finalIndex)
    {pe oe de : Nat} (pr : FieldRead reader .player pe) (op : FieldRead reader .opponent oe)
    (dr : FieldRead reader .metadata de) bigEndian result,
    empty bigEndian (memory.readValue dr.event) = false → finalIndex < 2 * epoch + modulus →
    read bigEndian (memory.readValue reader.first) (memory.readValue reader.last)
      ⟨memory.readValue pr.event, memory.readValue op.event, memory.readValue dr.event⟩ = some result →
    result = ⟨memory.writeValue (.payload .player) epoch, memory.writeValue (.payload .opponent) epoch,
      seqWord (2 * writers.baseEpoch + 2 * epoch), memory.writeValue (.payload .metadata) epoch⟩

theorem occupiedPublicationContract_proven : OccupiedPublicationContract := by
  intro Event memory epoch finalIndex writers reader pe oe de pr op dr endian result occupied noWrap hit
  exact occupied_snapshot_consistent writers reader pr op dr (empty endian) result occupied noWrap hit

def EmptyPublicationContract : Prop :=
  ∀ {Event : Type} {memory : Memory Event} {epoch finalIndex : Nat}
    (writers : Writers memory) (reader : Reader memory epoch finalIndex)
    {de : Nat} (dr : FieldRead reader .metadata de) unusedPlayer unusedOpponent bigEndian result,
    empty bigEndian (memory.readValue dr.event) = true → finalIndex < 2 * epoch + modulus →
    read bigEndian (memory.readValue reader.first) (memory.readValue reader.last)
      ⟨unusedPlayer, unusedOpponent, memory.readValue dr.event⟩ = some result →
    result = ⟨0, 0, seqWord (2 * writers.baseEpoch + 2 * epoch),
      memory.writeValue (.payload .metadata) epoch⟩

theorem emptyPublicationContract_proven : EmptyPublicationContract := by
  intro Event memory epoch finalIndex writers reader de dr unusedPlayer unusedOpponent endian result vacant noWrap hit
  exact empty_snapshot_consistent writers reader dr unusedPlayer unusedOpponent (empty endian) result vacant noWrap hit

end NeuralReversi.TranspositionTable.Snapshot

module
public import NeuralReversi.Board.Hash
public import NeuralReversi.PvsParallel.Cutoffs
public import NeuralReversi.TranspositionTable.ReaderMir.Cluster

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialTable

open TranspositionTable
open PvsBoardTree (Position)

structure Layout where
  clusters : BitVec 64
  positive : 0 < clusters.toNat
  bigEndian : Bool

abbrev Layout.Index (l : Layout) := Fin (l.clusters.toNat * Clear.clusterSize)

def Layout.index (l : Layout) (b : Position) (offset : Fin 2) : l.Index :=
  let key := Board.Hash.hash l.bigEndian b.val
  ⟨Clear.getClusterIdx key l.clusters + offset.val,
    Clear.probe_slot_in_bounds key l.clusters _ l.positive offset.isLt⟩

structure Image (l : Layout) where
  slots : Array Clear.SlotState
  size : slots.size = l.clusters.toNat * Clear.clusterSize
  generation : BitVec 8

def Image.get (im : Image l) (i : l.Index) : Clear.SlotState :=
  im.slots[i.val]'(im.size.symm ▸ i.isLt)

def payload (s : Clear.SlotState) : SeqLock.Payload := ⟨s.player,s.opponent,s.data⟩

def sample (endian : Bool) (s : Clear.SlotState) : Option (ProbeRetry.Key × EntryData) :=
  (Snapshot.read endian s.seq s.seq (payload s)).map fun sn =>
    ((sn.player,sn.opponent),Packing.unpack endian sn.metadata)

def Image.cell (im : Image l) (b : Position) (i : Fin 2) := im.get (l.index b i)

/-- A handled first slot suppresses the second load. Generation is loaded only on two occupied misses. -/
def Image.round (im : Image l) (b : Position) : ProbeRetry.RawAttempt :=
  let key := Cutoffs.key b
  let front := sample l.bigEndian (im.cell b 0)
  let first := ProbeRetry.classify key front
  let back := if ReaderMir.readsSecond first then sample l.bigEndian (im.cell b 1) else none
  let generation := if ReaderMir.readsGeneration first (ProbeRetry.classify key back) then im.generation else 0
  ⟨fun i => if i = 0 then front else back, generation⟩

def Image.attempts (im : Image l) (b : Position) : List ProbeRetry.RawAttempt :=
  let round := im.round b
  let observed := ProbeRetry.observe (Cutoffs.key b) round
  match (ReaderMir.probePass observed.generation observed.first observed.second).exit with
  | .done _ => [round]
  | .retry _ => List.replicate 3 round

def Image.probe (im : Image l) (b : Position) : ProbeRetry.Result :=
  ProbeRetry.probe ((im.attempts b).map (ProbeRetry.observe (Cutoffs.key b)))

def Image.slot (im : Image l) (b : Position) : l.Index :=
  l.index b (match im.probe b with | .hit i _ | .miss i => i)

def Image.lookupSlot (im : Image l) (b : Position) (i : Fin 2) : Option EntryData :=
  let s := im.cell b i
  LookupCut.readOccupied l.bigEndian s.seq s.seq (payload s) b.val.player b.val.opponent

def Image.lookup (im : Image l) (b : Position) : Option EntryData :=
  match im.lookupSlot b 0 with
  | some e => some e
  | none => im.lookupSlot b 1

def Resident (l : Layout) (h : Cutoffs.History) (s : Clear.SlotState) : Prop :=
  let e := Packing.unpack l.bigEndian s.data
  e.bound ≠ 0 → ∃ b : Position, (b,e) ∈ h ∧ s.player = b.val.player ∧ s.opponent = b.val.opponent

structure Image.Supported {l : Layout} (im : Image l) (h : Cutoffs.History) : Prop where
  even : ∀ i, (im.get i).seq &&& 1 = 0
  resident : ∀ i, Resident l h (im.get i)

end NeuralReversi.PvsParallel.SerialTable

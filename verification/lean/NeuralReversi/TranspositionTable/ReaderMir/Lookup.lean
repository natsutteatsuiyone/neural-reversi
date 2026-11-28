module
public import NeuralReversi.TranspositionTable.ReaderMir.Syntax
public import NeuralReversi.TranspositionTable.Clear
public import NeuralReversi.TranspositionTable.ProbeRetry

@[expose] public section

namespace NeuralReversi.TranspositionTable.ReaderMir

open SeqLock (Word)

def lookupRun (endian : Bool) (player opponent : Word) (w : Nat → Word) : List AtomicAction × Option EntryData :=
  if ((w 0) &&& 1) ≠ 0 then
    ⟨[.load .sequence .acquire], none⟩
  else
    if sameBoardBits (w 1) (w 2) player opponent then
      if (w 0) = (w 5) then
        ⟨[.load .sequence .acquire, .load (.payload .player) .relaxed, .load (.payload .opponent) .relaxed, .load (.payload .metadata) .relaxed, .fence .acquire, .load .sequence .relaxed], (if (Packing.unpack endian (w 3)).bound ≠ 0 then some (Packing.unpack endian (w 3)) else none)⟩
      else
        ⟨[.load .sequence .acquire, .load (.payload .player) .relaxed, .load (.payload .opponent) .relaxed, .load (.payload .metadata) .relaxed, .fence .acquire, .load .sequence .relaxed], none⟩
    else
      ⟨[.load .sequence .acquire, .load (.payload .player) .relaxed, .load (.payload .opponent) .relaxed], none⟩

def lookupCluster (clusters key player opponent : Word) (read : Nat → Word → Word → Option EntryData) : Option EntryData :=
  if let some hit := read (Clear.getClusterIdx key clusters) player opponent then
    some hit
  else
    read ((Clear.getClusterIdx key clusters) + 1) player opponent

def probeSlots (clusters key : Word) : Nat × Nat :=
  ((Clear.getClusterIdx key clusters), ((Clear.getClusterIdx key clusters) + 1))

def probeSecond (query : ProbeRetry.Key) : Option (ProbeRetry.Key × EntryData) → Bool
  | none => true
  | some (stored,data) => data.bound != 0 && !sameBoardBits stored.1 stored.2 query.1 query.2

def probeRetries : Nat := 3

end NeuralReversi.TranspositionTable.ReaderMir

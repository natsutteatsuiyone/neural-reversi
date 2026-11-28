module
public import NeuralReversi.TranspositionTable.ReaderMir.Syntax

@[expose] public section

namespace NeuralReversi.TranspositionTable.ReaderMir

open SeqLock (Word)

def run (bigEndian : Bool) (word : Nat → Word) : Outcome :=
  if (word 0 &&& 1) ≠ 0 then
    ⟨[.load .sequence .acquire], none⟩
  else
    if (Packing.unpack bigEndian (word 1)).bound = 0#8 then
      if word 0 ≠ word 3 then
        ⟨[.load .sequence .acquire, .load (.payload .metadata) .relaxed, .fence .acquire, .load .sequence .relaxed], none⟩
      else
        ⟨[.load .sequence .acquire, .load (.payload .metadata) .relaxed, .fence .acquire, .load .sequence .relaxed], some ⟨0, 0, word 0, word 1⟩⟩
    else
      if word 0 ≠ word 5 then
        ⟨[.load .sequence .acquire, .load (.payload .metadata) .relaxed, .load (.payload .player) .relaxed, .load (.payload .opponent) .relaxed, .fence .acquire, .load .sequence .relaxed], none⟩
      else
        ⟨[.load .sequence .acquire, .load (.payload .metadata) .relaxed, .load (.payload .player) .relaxed, .load (.payload .opponent) .relaxed, .fence .acquire, .load .sequence .relaxed], some ⟨word 2, word 3, word 0, word 1⟩⟩

end NeuralReversi.TranspositionTable.ReaderMir

module
public import NeuralReversi.TranspositionTable.WriterMir.Syntax

@[expose] public section

namespace NeuralReversi.TranspositionTable.WriterMir

open SeqLock (Word)

def writeStable (endian : Bool) (seq : { v : Word // v &&& 1 = 0 })
    (player opponent : Word) (entry : EntryData) (casSuccess : Bool) : List Atom × Bool :=
  if casSuccess then
    ⟨[.cas .sequence seq.val (seq.val ||| 1) .acqRel .relaxed, .fence .release, .store (.payload .player) player .relaxed, .store (.payload .opponent) opponent .relaxed, .store (.payload .metadata) (Packing.pack endian entry) .relaxed, .store .sequence (seq.val + 2) .release], true⟩
  else
    ⟨[.cas .sequence seq.val (seq.val ||| 1) .acqRel .relaxed], false⟩

end NeuralReversi.TranspositionTable.WriterMir

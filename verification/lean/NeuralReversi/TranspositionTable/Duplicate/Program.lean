module
public import NeuralReversi.TranspositionTable.Duplicate

@[expose] public section

namespace NeuralReversi.TranspositionTable.Duplicate

def wordIndex : SeqLock.Location → Fin 4
  | .payload .player => 0
  | .payload .opponent => 1
  | .payload .metadata => 2
  | .sequence => 3

def locationAt : Fin 4 → SeqLock.Location
  | 0 => .payload .player
  | 1 => .payload .opponent
  | 2 => .payload .metadata
  | 3 => .sequence

def zeroEntries (before : Bytes) (dest len : Nat) : Bytes :=
  fun addr => if dest ≤ addr ∧ addr < dest + len*32 then 0 else before addr

def copyEntries (before : Bytes) (source dest len : Nat) : Bytes :=
  copy (zeroEntries before dest len) source dest (len*32)

def exclusiveStores : List (SeqLock.Location × SeqLock.Word) :=
  [(.sequence,0), (.payload .player,0), (.payload .opponent,0), (.payload .metadata,0)]

def clearEntry (prior : SeqLock.Location → SeqLock.Word) : SeqLock.Location → SeqLock.Word :=
  exclusiveStores.foldl (fun words (loc,value) => fun l => if l = loc then value else words l) prior

end NeuralReversi.TranspositionTable.Duplicate

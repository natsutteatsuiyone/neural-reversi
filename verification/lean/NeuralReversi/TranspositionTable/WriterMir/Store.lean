import NeuralReversi.TranspositionTable.Packing

namespace NeuralReversi.TranspositionTable.WriterMir

def makeData (score : Score) (move : Option (Fin 64)) (bound : Bound) (depth : Depth)
    (selectivity : Selectivity) (generation : BitVec 8) (endgame pv : Bool) : EntryData :=
  if endgame then
    if pv then ⟨(score.raw.setWidth 16), (Packing.encodeBestMove move), (Packing.boundByte bound), (depth.setWidth 8), (BitVec.ofNat 8 selectivity.val), (generation &&& 127), (1 ||| 2)⟩
    else ⟨(score.raw.setWidth 16), (Packing.encodeBestMove move), (Packing.boundByte bound), (depth.setWidth 8), (BitVec.ofNat 8 selectivity.val), (generation &&& 127), (1 ||| 0)⟩
  else
    if pv then ⟨(score.raw.setWidth 16), (Packing.encodeBestMove move), (Packing.boundByte bound), (depth.setWidth 8), (BitVec.ofNat 8 selectivity.val), (generation &&& 127), (0 ||| 2)⟩
    else ⟨(score.raw.setWidth 16), (Packing.encodeBestMove move), (Packing.boundByte bound), (depth.setWidth 8), (BitVec.ofNat 8 selectivity.val), (generation &&& 127), (0 ||| 0)⟩

def storeInput (slot : σ) (board : β) (score : Score) (bound : Bound) (depth : Depth)
    (move : Option (Fin 64)) (selectivity : Selectivity) (generation : BitVec 8)
    (endgame pv : Bool) : Option (σ × β × EntryData) :=
  if score.raw.toInt.natAbs < 32767 then some (slot, board, (makeData score move bound depth selectivity (generation &&& 127) endgame pv))
  else none

end NeuralReversi.TranspositionTable.WriterMir

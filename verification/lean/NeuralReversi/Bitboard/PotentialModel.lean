module
public import NeuralReversi.Bitboard.MovesModel

@[expose] public section

namespace NeuralReversi.Bitboard.Moves

def verticalMask : Bits := 0x00ffffffffffff00
def diagonalMask : Bits := 0x007e7e7e7e7e7e00

def potentialCandidates (opponent : Bits) : Bits :=
  let hOpp := opponent &&& horizontalMask
  let vOpp := opponent &&& verticalMask
  let dOpp := opponent &&& diagonalMask
  (hOpp <<< (1 : Nat)) ||| (hOpp >>> (1 : Nat)) |||
  (vOpp <<< (8 : Nat)) ||| (vOpp >>> (8 : Nat)) |||
  (dOpp <<< (7 : Nat)) ||| (dOpp >>> (7 : Nat)) |||
  (dOpp <<< (9 : Nat)) ||| (dOpp >>> (9 : Nat))

def getMovesAndPotentialPortable (player opponent : Bits) : Bits × Bits :=
  let empty := ~~~(player ||| opponent)
  let potential := potentialCandidates opponent &&& empty
  (getMovesPortable player opponent, potential)

end NeuralReversi.Bitboard.Moves

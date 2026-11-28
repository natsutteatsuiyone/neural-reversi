module
public import NeuralReversi.Simd.ZmmModel
public import NeuralReversi.Bitboard.PotentialModel

@[expose] public section

namespace NeuralReversi.Bitboard.Moves
open NeuralReversi.Simd.X86

def getMovesAndPotentialAvx512 (player opponent : Bits) : Bits × Bits :=
  let sh := setEpi64 57 55 56 63 7 9 8 1
  let masks := setEpi64 diagonalMask diagonalMask verticalMask horizontalMask
    diagonalMask diagonalMask verticalMask horizontalMask
  let empty := ~~~(player ||| opponent)
  let pp := set1Epi64 player
  let oo := set1Epi64 opponent
  let maskedOo := and512 oo masks
  let potentialMm := rolv512 maskedOo sh
  let flips := and512 maskedOo (rolv512 pp sh)
  let flips := ternary512 flips maskedOo (rolv512 flips sh) 0xf8
  let pre := and512 maskedOo potentialMm
  let sh2 := add512 sh sh
  let flips := ternary512 flips pre (rolv512 flips sh2) 0xf8
  let flips := ternary512 flips pre (rolv512 flips sh2) 0xf8
  let movesMm := rolv512 flips sh
  let (moves,potential) := horizontalOr2Zmm movesMm potentialMm
  (moves &&& empty,potential &&& empty)


end NeuralReversi.Bitboard.Moves

module
public import NeuralReversi.Simd.TernaryFast
public import NeuralReversi.Bitboard.PotentialModel

@[expose] public section

namespace NeuralReversi.Bitboard.Moves
open NeuralReversi.Simd.X86

def getMovesAvx512 (player opponent : Bits) : Bits :=
  let sh := setEpi64x 7 9 8 1
  let masks := setEpi64x diagonalMask diagonalMask verticalMask horizontalMask
  let empty := ~~~(player ||| opponent)
  let pp := set1Epi64x player
  let oo := set1Epi64x opponent
  let maskedOo := and256 oo masks
  let fl := and256 maskedOo (sllvEpi64 pp sh)
  let fr := and256 maskedOo (srlvEpi64 pp sh)
  let fl := ternarylogicEpi64Fast fl maskedOo (sllvEpi64 fl sh) 0xf8
  let fr := ternarylogicEpi64Fast fr maskedOo (srlvEpi64 fr sh) 0xf8
  let preL := and256 maskedOo (sllvEpi64 maskedOo sh)
  let preR := srlvEpi64 preL sh
  let sh2 := addEpi64 sh sh
  let fl := ternarylogicEpi64Fast fl preL (sllvEpi64 fl sh2) 0xf8
  let fr := ternarylogicEpi64Fast fr preR (srlvEpi64 fr sh2) 0xf8
  let fl := ternarylogicEpi64Fast fl preL (sllvEpi64 fl sh2) 0xf8
  let fr := ternarylogicEpi64Fast fr preR (srlvEpi64 fr sh2) 0xf8
  let mm := or256 (sllvEpi64 fl sh) (srlvEpi64 fr sh)
  let moves := horizontalOrU64 mm
  moves &&& empty

end NeuralReversi.Bitboard.Moves

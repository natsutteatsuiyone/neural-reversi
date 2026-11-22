import NeuralReversi.Simd.PairModel
import NeuralReversi.Bitboard.PotentialModel

namespace NeuralReversi.Bitboard.Moves
open NeuralReversi.Simd.X86
def getMovesAndPotentialAvx2 (player opponent : Bits) : Bits × Bits :=
  let sh := setEpi64x 7 9 8 1
  let masks := setEpi64x diagonalMask diagonalMask verticalMask horizontalMask
  let empty := ~~~(player ||| opponent)
  let pp := set1Epi64x player
  let oo := set1Epi64x opponent
  let maskedOo := and256 oo masks
  let potentialMm := or256 (sllvEpi64 maskedOo sh) (srlvEpi64 maskedOo sh)
  let fl := and256 maskedOo (sllvEpi64 pp sh)
  let fr := and256 maskedOo (srlvEpi64 pp sh)
  let fl := or256 fl (and256 maskedOo (sllvEpi64 fl sh))
  let fr := or256 fr (and256 maskedOo (srlvEpi64 fr sh))
  let preL := and256 maskedOo (sllvEpi64 maskedOo sh)
  let preR := srlvEpi64 preL sh
  let shift2 := addEpi64 sh sh
  let fl := or256 fl (and256 preL (sllvEpi64 fl shift2))
  let fr := or256 fr (and256 preR (srlvEpi64 fr shift2))
  let fl := or256 fl (and256 preL (sllvEpi64 fl shift2))
  let fr := or256 fr (and256 preR (srlvEpi64 fr shift2))
  let mm := or256 (sllvEpi64 fl sh) (srlvEpi64 fr sh)
  let (moves,potential) := horizontalOr2U64 mm potentialMm
  (moves &&& empty,potential &&& empty)


end NeuralReversi.Bitboard.Moves

import NeuralReversi.Bitboard.Avx2Model
import NeuralReversi.Bitboard.Avx512Model
import NeuralReversi.Bitboard.Avx2PotentialModel
import NeuralReversi.Bitboard.Avx512PotentialModel

namespace NeuralReversi.Bitboard.Moves

structure X86Config where
  avx512vl : Bool
  avx2 : Bool

def getMovesX86 (config : X86Config) (player opponent : Bits) : Bits :=
  if config.avx512vl then getMovesAvx512 player opponent
  else if config.avx2 then getMovesAvx2 player opponent
  else getMovesPortable player opponent

def getMovesAndPotentialX86 (config : X86Config) (player opponent : Bits) : Bits × Bits :=
  if config.avx512vl then getMovesAndPotentialAvx512 player opponent
  else if config.avx2 then getMovesAndPotentialAvx2 player opponent
  else getMovesAndPotentialPortable player opponent

abbrev legalMovesX86 := getMovesX86

abbrev legalMovesAndPotentialX86 := getMovesAndPotentialX86

end NeuralReversi.Bitboard.Moves

namespace NeuralReversi.Bitboard

def getMoves (config : Moves.X86Config) (player opponent : Bits) : Bits :=
  Moves.getMovesX86 config player opponent

def getMovesAndPotential (config : Moves.X86Config) (player opponent : Bits) : Bits × Bits :=
  let (moves,potential) := Moves.getMovesAndPotentialX86 config player opponent
  (moves,potential)

abbrev legalMoves := getMoves

abbrev legalMovesAndPotential := getMovesAndPotential

end NeuralReversi.Bitboard

import NeuralReversi.Bitboard.SquareModel

namespace NeuralReversi.PvsParallel

def rootWipeoutExit (square : Bitboard.Square) : Prod (List (Prod Bitboard.Square (Prod Int Bool))) Int :=
  ([(square, 16384, true)], 16384)

end NeuralReversi.PvsParallel

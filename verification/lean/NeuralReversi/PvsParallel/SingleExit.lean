import NeuralReversi.Bitboard.SquareModel

namespace NeuralReversi.PvsParallel

def singleRootExit (square : Bitboard.Square) (child : Int) :
    Prod (List (Prod Bitboard.Square (Prod Int Bool))) Int :=
  if -32767 < -child then ([(square, (-child), true)], (-child))
  else ([(square, (-child), true)], -32767)

end NeuralReversi.PvsParallel

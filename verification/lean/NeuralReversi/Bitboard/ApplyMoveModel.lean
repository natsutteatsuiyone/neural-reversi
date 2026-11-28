module
public import NeuralReversi.Bitboard.SquareModel

@[expose] public section

namespace NeuralReversi.Bitboard

def applyMoveAny (debugAssertions overflowChecks : Bool) (bits flipped : Bits) (sq : AnySquare) :
    Except String Bits :=
  let toggled := bits ^^^ flipped
  (squareBitboard debugAssertions overflowChecks sq).map fun mask => toggled ^^^ mask

end NeuralReversi.Bitboard

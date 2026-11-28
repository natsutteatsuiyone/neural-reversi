module
public import NeuralReversi.Bitboard.Model
public import NeuralReversi.Square.Model

@[expose] public section

namespace NeuralReversi.Bitboard

abbrev AnySquare := NeuralReversi.Square.Square

def fromSquare (sq : AnySquare) : Bits :=
  (1 : Bits) <<< sq.val

def squareBitboard (_debugAssertions _overflowChecks : Bool) (sq : AnySquare) :
    Except String Bits :=
  .ok (fromSquare sq)

def setSquare (debugAssertions overflowChecks : Bool) (bits : Bits) (sq : AnySquare) :
    Except String Bits :=
  (squareBitboard debugAssertions overflowChecks sq).map fun mask => bits ||| mask

def containsSquare (debugAssertions overflowChecks : Bool) (bits : Bits) (sq : AnySquare) :
    Except String Bool :=
  (squareBitboard debugAssertions overflowChecks sq).map fun mask => bits &&& mask != 0

def isEmpty (bits : Bits) : Bool := bits == 0

end NeuralReversi.Bitboard

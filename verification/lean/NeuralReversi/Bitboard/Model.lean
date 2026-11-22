import Std

namespace NeuralReversi.Bitboard

abbrev Bits := BitVec 64
abbrev Square := Fin 64

def squareMask (sq : Square) : Bits := (1 : Bits) <<< sq.val

def applyFlip (bits flipped : Bits) : Bits := bits ^^^ flipped

def applyMove (bits flipped : Bits) (sq : Square) : Bits :=
  bits ^^^ flipped ^^^ squareMask sq

def clearLsb (bits : Bits) : Bits := bits &&& (bits - 1)

def hasSingleBitNonzero (bits : Bits) : Bool := (bits &&& (bits - 1)) == 0

def count (bits : Bits) : BitVec 32 := bits.cpop.setWidth 32

end NeuralReversi.Bitboard

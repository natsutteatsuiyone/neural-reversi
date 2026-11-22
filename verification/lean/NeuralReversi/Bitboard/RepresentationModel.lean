import NeuralReversi.Bitboard.SquareModel

namespace NeuralReversi.Bitboard

structure Value where
  storage : Bits
  deriving DecidableEq, Repr

def new (raw : Bits) : Value := ⟨raw⟩
def bits (value : Value) : Bits := value.storage
def fromU64 (raw : Bits) : Value := ⟨raw⟩
def intoU64 (value : Value) : Bits := value.storage

def valueContains (checked : Bool) (value : Value) (sq : AnySquare) : Except String Bool :=
  containsSquare checked checked value.storage sq

def valueCount (value : Value) : BitVec 32 := count value.storage

def fromSquareValue (checked : Bool) (sq : AnySquare) : Except String Value :=
  (squareBitboard checked checked sq).map Value.mk

def displayChar (bits : Bits) (rank file : Nat) : Char :=
  let sq := rank * 8 + file
  if ((bits >>> sq) &&& 1#64) != 0#64 then '1' else '.'

def displayRank (bits : Bits) (rank : Nat) : List Char :=
  (List.range 8).map (displayChar bits rank) ++ ['\n']

def displayChars (value : Value) : List Char :=
  ((List.range 8).map (displayRank value.storage)).flatten

def display (value : Value) : String :=
  String.ofList (displayChars value)

end NeuralReversi.Bitboard

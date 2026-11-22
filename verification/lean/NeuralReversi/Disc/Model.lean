import Std

namespace NeuralReversi.Disc

inductive Disc where
  | empty
  | black
  | white
  deriving DecidableEq, Repr

def toChar : Disc → Char
  | .empty => '-'
  | .black => 'X'
  | .white => 'O'

inductive PlayerColor where
  | black
  | white
  deriving DecidableEq, Repr

namespace PlayerColor

def opposite : PlayerColor → PlayerColor
  | .black => .white
  | .white => .black

def toDisc : PlayerColor → Disc
  | .black => .black
  | .white => .white

def toChar (c : PlayerColor) : Char :=
  Disc.toChar c.toDisc

end PlayerColor

end NeuralReversi.Disc

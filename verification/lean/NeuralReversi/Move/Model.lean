import NeuralReversi.Bitboard.Model
import NeuralReversi.Square.Model

namespace NeuralReversi.Move

structure Move where
  flipped : Bitboard.Bits
  value : BitVec 32
  sq : Square.Square
  deriving DecidableEq, Repr

inductive Failure where
  | emptyFlips
  deriving DecidableEq, Repr

def Failure.message : Failure → String
  | .emptyFlips => "Move must flip at least one disc"

def create (checked : Bool) (sq : Square.Square) (flipped : Bitboard.Bits) :
    Except Failure Move :=
  if checked && flipped == 0 then .error .emptyFlips
  else .ok ⟨flipped,0,sq⟩


end NeuralReversi.Move

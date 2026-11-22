import NeuralReversi.Move.Model

namespace NeuralReversi.Move

def ConstructorSpec (checked : Bool) (sq : Square.Square) (flipped : Bitboard.Bits) :
    Except Failure Move → Prop
  | .error .emptyFlips => checked = true ∧ flipped = 0
  | .ok value => (checked = true → flipped ≠ 0) ∧
      value.flipped = flipped ∧ value.value = 0 ∧ value.sq = sq

def FailureTextSpec (checked : Bool) (_sq : Square.Square) (flipped : Bitboard.Bits)
    (text : String) : Prop :=
  checked = true ∧ flipped = 0 ∧ text = "Move must flip at least one disc"


end NeuralReversi.Move

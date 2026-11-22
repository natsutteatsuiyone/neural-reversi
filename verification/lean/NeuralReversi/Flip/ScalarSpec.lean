import NeuralReversi.Flip.ScalarModel

namespace NeuralReversi.Flip.Scalar

def Least (x : Word) (j : Nat) : Prop :=
  x.getLsbD j = true ∧ ∀ k, k < j → x.getLsbD k = false

def Greatest (x : Word) (j : Nat) : Prop :=
  x.getLsbD j = true ∧ ∀ k, j < k → x.getLsbD k = false

def LeftBitSpec (mask player notOpponent : Word) (i : Nat) : Prop :=
  ∃ j, Least (notOpponent &&& mask) j ∧ player.getLsbD j = true ∧
    mask.getLsbD i = true ∧ i < j

def RightBitSpec (mask player notOpponent : Word) (i : Nat) : Prop :=
  ∃ j, Greatest (notOpponent &&& mask) j ∧ player.getLsbD j = true ∧
    mask.getLsbD i = true ∧ j < i

end NeuralReversi.Flip.Scalar

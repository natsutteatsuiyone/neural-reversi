module
public import NeuralReversi.Flip.LrmaskModel
public import NeuralReversi.Flip.ScalarModel

@[expose] public section

namespace NeuralReversi.Flip
open Scalar

def flip (checks : Bool) (sq : Square) (player opponent : Word) : Except String Word := do
  let n := ~~~opponent
  let left := flipLeft (mask sq 0) player n ||| flipLeft (mask sq 1) player n |||
    flipLeft (mask sq 2) player n ||| flipLeft (mask sq 3) player n
  let r4 ← flipRight checks (mask sq 4) player n
  let r5 ← flipRight checks (mask sq 5) player n
  let r6 ← flipRight checks (mask sq 6) player n
  let r7 ← flipRight checks (mask sq 7) player n
  pure (left ||| r4 ||| r5 ||| r6 ||| r7)

def value (sq : Square) (player opponent : Word) : Word :=
  let n := ~~~opponent
  flipLeft (mask sq 0) player n ||| flipLeft (mask sq 1) player n |||
    flipLeft (mask sq 2) player n ||| flipLeft (mask sq 3) player n |||
    flipRightValue (mask sq 4) player n ||| flipRightValue (mask sq 5) player n |||
    flipRightValue (mask sq 6) player n ||| flipRightValue (mask sq 7) player n

end NeuralReversi.Flip

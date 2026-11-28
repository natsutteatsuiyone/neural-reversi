module

@[expose] public section

namespace NeuralReversi.AspirationMir

inductive DriverOp where
  | prepare (next : Nat)
  | go (next : Nat)
  | query (next : Nat)
  | poll (next : Nat)
  | branchAbort (ifFalse ifTrue : Nat)
  | widen (next : Nat)
  | branchWiden (ifFalse ifTrue : Nat)
  | double (next : Nat)
  | saveScore (next : Nat)
  | returned
  | unwind
  deriving Repr, DecidableEq

inductive Expr where
  | score | alpha | beta | delta | inf
  | neg (x : Expr)
  | sub (x y : Expr)
  | add (x y : Expr)
  | min (x y : Expr)
  | max (x y : Expr)
  deriving Repr, DecidableEq

inductive Guard where
  | le (x y : Expr)
  | ge (x y : Expr)
  deriving Repr, DecidableEq

abbrev WidenPath := List (Guard × Bool) × Expr × Expr × Bool

end NeuralReversi.AspirationMir

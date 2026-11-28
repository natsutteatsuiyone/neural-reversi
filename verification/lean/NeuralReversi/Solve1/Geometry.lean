module

@[expose] public section

namespace NeuralReversi.Solve1.Geometry
abbrev Word := BitVec 64
def ray (p : Word) (acc : Nat) : List Nat → Nat
  | [] => 0
  | i :: rest => if p.getLsbD i then acc else ray p (acc + 1) rest
def raySquares (square : Nat) (dr dc : Int) : List Nat :=
  (List.range 7).filterMap fun k =>
    let r : Int := (square / 8 : Nat) + dr * (k + 1 : Nat)
    let c : Int := (square % 8 : Nat) + dc * (k + 1 : Nat)
    if 0 ≤ r ∧ r < 8 ∧ 0 ≤ c ∧ c < 8 then some (r.toNat * 8 + c.toNat) else none
def reference (p : Word) (square : Nat) : Nat :=
  ray p 0 (raySquares square 0 1) +
  ray p 0 (raySquares square 1 0) +
  ray p 0 (raySquares square 1 1) +
  ray p 0 (raySquares square 1 (-1)) +
  ray p 0 (raySquares square 0 (-1)) +
  ray p 0 (raySquares square (-1) 0) +
  ray p 0 (raySquares square (-1) (-1)) +
  ray p 0 (raySquares square (-1) 1)
end NeuralReversi.Solve1.Geometry

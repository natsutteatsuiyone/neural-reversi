module
public import NeuralReversi.Bitboard.LeastBit

@[expose] public section

namespace NeuralReversi.Bitboard

def iteratorNew (bits : Bits) : Bits := bits
def bitboardIter (bits : Bits) : Bits := iteratorNew bits
def intoIterator (bits : Bits) : Bits := iteratorNew bits

def iteratorNextDirect (state : Bits) : Option NeuralReversi.Square.Square × Bits :=
  match lsbSquare state with
  | none => (none, state)
  | some square => (some square, clearLsb state)

def iteratorNext (_checked : Bool) (state : Bits) :
    Except String (Option NeuralReversi.Square.Square × Bits) :=
  .ok (iteratorNextDirect state)

def iteratorTrace : Nat → Bool → Bits → Except String (List (Option Nat))
  | 0, _, _ => .ok []
  | fuel + 1, checked, state => do
    let (square, rest) ← iteratorNext checked state
    let tail ← iteratorTrace fuel checked rest
    pure (square.map Fin.val :: tail)

def setIndices (bits : Bits) : List Nat := (List.range 64).filter bits.getLsbD

def expectedIteratorTrace (fuel : Nat) (bits : Bits) : List (Option Nat) :=
  (setIndices bits).map some ++ List.replicate (fuel - (setIndices bits).length) none

end NeuralReversi.Bitboard

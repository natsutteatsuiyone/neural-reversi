module
public import NeuralReversi.Board.Model
public import NeuralReversi.Bitboard.SquareModel
public import NeuralReversi.Disc.Model
public import NeuralReversi.Square.Model

@[expose] public section

namespace NeuralReversi.Board
open Bitboard

def discAt (checked : Bool) (b : Board) (sq : AnySquare) (side : Disc.PlayerColor) :
    Except String Disc.Disc :=
  match containsSquare checked checked b.player sq with
  | .error reason => .error reason
  | .ok true => .ok side.toDisc
  | .ok false => match containsSquare checked checked b.opponent sq with
    | .error reason => .error reason
    | .ok true => .ok side.opposite.toDisc
    | .ok false => .ok .empty

def renderChars (checked : Bool) (b : Board) (side : Disc.PlayerColor) :
    Nat → List AnySquare → Except String (List Char)
  | _, [] => .ok []
  | i, sq :: rest => do
    let disc ← discAt checked b sq side
    let tail ← renderChars checked b side (i + 1) rest
    pure ((if 0 < i ∧ i % 8 = 0 then ['\n'] else []) ++ Disc.toChar disc :: tail)

def toStringAsBoard (checked : Bool) (b : Board) (side : Disc.PlayerColor) : Except String String :=
  (renderChars checked b side 0 Square.iter).map String.ofList

end NeuralReversi.Board

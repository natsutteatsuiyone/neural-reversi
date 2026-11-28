module
public import NeuralReversi.Board.TextModel

@[expose] public section

namespace NeuralReversi.Board
open Bitboard

abbrev Size := BitVec 64

inductive TextError where
  | tooShort (expected actual : Size)
  | tooLong (expected actual : Size)
  | invalidChar (char : Char) (position : Size)
  deriving DecidableEq, Repr

inductive TextFailure where
  | error (reason : TextError)
  | panic (message : String)
  deriving DecidableEq, Repr

def parseTokens (checked : Bool) (side : Disc.PlayerColor) (state : Board) :
    List (AnySquare × Char) → Except TextFailure Board
  | [] => .ok state
  | (sq, c) :: rest =>
    if c = side.toChar then
      match setSquare checked checked state.player sq with
      | .ok bits => parseTokens checked side { state with player := bits } rest
      | .error message => .error (.panic message)
    else if c = side.opposite.toChar then
      match setSquare checked checked state.opponent sq with
      | .ok bits => parseTokens checked side { state with opponent := bits } rest
      | .error message => .error (.panic message)
    else if c ≠ '-' then .error (.error (.invalidChar c (BitVec.ofNat 64 sq.val)))
    else parseTokens checked side state rest

def fromString (checked : Bool) (text : String) (side : Disc.PlayerColor) : Except TextFailure Board :=
  let count := BitVec.ofNat 64 text.toList.length
  if count.toNat < 64 then .error (.error (.tooShort 64 count))
  else if 64 < count.toNat then .error (.error (.tooLong 64 count))
  else parseTokens checked side ⟨0#64, 0#64⟩ (Square.iter.zip text.toList)

def RepresentableText (text : String) : Prop := text.toList.length < 2 ^ 64

end NeuralReversi.Board

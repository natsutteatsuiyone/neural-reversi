import NeuralReversi.Bitboard.Model
import NeuralReversi.Bitboard.SquareModel

namespace NeuralReversi.Board
open Bitboard

structure Board where
  player : Bits
  opponent : Bits
  deriving DecidableEq, Repr

def wellFormed (b : Board) : Prop := b.player &&& b.opponent = 0

def emptySquares (b : Board) : Bits := ~~~(b.player ||| b.opponent)

def playerCount (b : Board) : BitVec 32 := count b.player

def opponentCount (b : Board) : BitVec 32 := count b.opponent

def emptyCount (b : Board) : BitVec 32 := count (emptySquares b)

def switchPlayers (b : Board) : Board := ⟨b.opponent, b.player⟩

def place (b : Board) (flipped : Bits) (sq : Square) : Board :=
  let placed := squareMask sq &&& ~~~b.opponent
  ⟨b.opponent &&& ~~~flipped, (b.player ||| placed) ||| flipped⟩

def makeMoveWithFlipped (debugAssertions : Bool) (b : Board) (flipped : Bits)
    (sq : Square) : Except String Board :=
  if debugAssertions then
    if flipped = 0 then .error "flipped must not be empty"
    else if emptySquares b &&& squareMask sq = 0 then .error "sq must be an empty square"
    else if flipped &&& ~~~b.opponent ≠ 0 then
      .error "flipped must be a subset of opponent's discs"
    else .ok (place b flipped sq)
  else .ok (place b flipped sq)

def placeWithFlippedAny (debugAssertions overflowChecks : Bool) (b : Board)
    (flipped : Bits) (sq : AnySquare) : Except String Board :=
  match squareBitboard debugAssertions overflowChecks sq with
  | .error reason => .error reason
  | .ok mask =>
    let placed := mask &&& ~~~b.opponent
    .ok ⟨b.opponent &&& ~~~flipped, (b.player ||| placed) ||| flipped⟩

def makeMoveWithFlippedAny (debugAssertions overflowChecks : Bool) (b : Board)
    (flipped : Bits) (sq : AnySquare) : Except String Board :=
  if debugAssertions then
    if flipped = 0 then .error "flipped must not be empty"
    else match containsSquare debugAssertions overflowChecks (emptySquares b) sq with
      | .error reason => .error reason
      | .ok empty =>
        if !empty then .error "sq must be an empty square"
        else if flipped &&& ~~~b.opponent ≠ 0 then
          .error "flipped must be a subset of opponent's discs"
        else placeWithFlippedAny debugAssertions overflowChecks b flipped sq
  else placeWithFlippedAny debugAssertions overflowChecks b flipped sq

end NeuralReversi.Board

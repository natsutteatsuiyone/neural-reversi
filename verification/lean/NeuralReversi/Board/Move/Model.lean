import NeuralReversi.Board.Model
import NeuralReversi.Bitboard.ApplyMoveModel
import NeuralReversi.Flip.Dispatch.Model

namespace NeuralReversi.Board

def placeAny (debugAssertions overflowChecks : Bool) (b : Board) (flipped : Bitboard.Bits)
    (sq : Bitboard.AnySquare) : Except String Board :=
  placeWithFlippedAny debugAssertions overflowChecks b flipped sq

def makeMove (config : Flip.Dispatch.X86Config) (debugAssertions overflowChecks : Bool)
    (b : Board) (sq : Bitboard.AnySquare) : Except String Board := do
  if debugAssertions then
    let empty ← Bitboard.containsSquare debugAssertions overflowChecks (emptySquares b) sq
    if !empty then throw "sq must be an empty square"
  let flipped ← Flip.Dispatch.flip config overflowChecks sq b.player b.opponent
  if debugAssertions then
    if Bitboard.isEmpty flipped then throw "assertion failed: !flipped.is_empty()"
    else placeAny debugAssertions overflowChecks b flipped sq
  else placeAny debugAssertions overflowChecks b flipped sq

end NeuralReversi.Board

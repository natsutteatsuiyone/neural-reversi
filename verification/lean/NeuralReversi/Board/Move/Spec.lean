module
public import NeuralReversi.Board.Model
public import NeuralReversi.Square.Model
public import NeuralReversi.Flip.Dispatch.Spec

@[expose] public section

namespace NeuralReversi.Board.MoveSpec

def Placement (b : Board) (flipped : Bitboard.Bits) (index : Nat) (outcome : Except String Board) : Prop :=
  ∃ result, outcome = .ok result ∧ ∀ i,
    result.player.getLsbD i = (b.opponent.getLsbD i && !flipped.getLsbD i) ∧
    result.opponent.getLsbD i =
      ((b.player.getLsbD i || (decide (i = index) && !b.opponent.getLsbD i)) || flipped.getLsbD i)

def PlaceOutcome (_debugAssertions _overflowChecks : Bool) (b : Board) (flipped : Bitboard.Bits)
    (sq : NeuralReversi.Square.Square) (outcome : Except String Board) : Prop :=
  Placement b flipped sq.val outcome

def MakeOutcome (config : Flip.Dispatch.X86Config) (debugAssertions overflowChecks : Bool)
    (b : Board) (sq : NeuralReversi.Square.Square) (outcome : Except String Board) : Prop :=
  if debugAssertions && decide (b.player.getLsbD sq.val = true ∨ b.opponent.getLsbD sq.val = true) then
    outcome = .error "sq must be an empty square"
  else
    ∃ flipped, (∀ i, flipped.getLsbD i = true ↔ Flip.Dispatch.RawCoordinate config sq b.player b.opponent i) ∧
      if debugAssertions && decide (flipped = 0) then
        outcome = .error "assertion failed: !flipped.is_empty()"
      else PlaceOutcome debugAssertions overflowChecks b flipped sq outcome

end NeuralReversi.Board.MoveSpec

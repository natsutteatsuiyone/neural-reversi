import NeuralReversi.Game.Initialization.Properties
import NeuralReversi.Game.History.Fixture

namespace NeuralReversi.Game.Initialization.Fixture
open NeuralReversi.Game.History

def render (state : State) : String :=
  s!"{state.board.player.toNat},{state.board.opponent.toNat},{History.Fixture.sideCode state.side}/{state.history.val.lengthWord.toNat}/{state.history.val.allocation.address.toNat}"

def request (fields : List String) : Except String String := do
  match fields with
  | [_, "new"] => return render newGame
  | [_, "vec"] =>
    return s!"vec/{emptyVector.val.lengthWord.toNat},{emptyVector.val.allocation.capacityWord.toNat},{emptyVector.val.allocation.address.toNat}"
  | [_, "from", player, opponent, side] =>
    let value ← History.Fixture.parseState s!"{player},{opponent},{side}/-"
    return render (fromBoard value.board value.side)
  | _ => throw "invalid initialization request"

end NeuralReversi.Game.Initialization.Fixture

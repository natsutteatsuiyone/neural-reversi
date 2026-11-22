import NeuralReversi.Game.Queries.Properties
import NeuralReversi.Game.History.Fixture

namespace NeuralReversi.Game.Queries.Fixture
open NeuralReversi.Game.History

def actionRequest (fields : List String) : Except String String := do
  let [_,tag] := fields | throw "expected action tag"
  let value := actionSquare (← History.Fixture.parseAction tag)
  return value.map (fun square => toString square.val) |>.getD "-"

def valueRequest (fields : List String) : Except String String := do
  let [_,before] := fields | throw "expected query before-state"
  let state ← History.Fixture.parseState before
  let last := (valueLast state).map (fun square => toString square.val) |>.getD "-"
  let points := valueScore state
  return s!"{last}/{points.1.toNat},{points.2.toNat}"

end NeuralReversi.Game.Queries.Fixture

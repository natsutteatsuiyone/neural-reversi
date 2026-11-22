import NeuralReversi.Game.Terminal.Properties
import NeuralReversi.Game.History.Fixture

namespace NeuralReversi.Game.Terminal.Fixture
open NeuralReversi.Game.History

def request (fields : List String) : Except String String := do
  let [_, before] := fields | throw "expected terminal before-state"
  let state ← History.Fixture.parseState before
  return if valueProjection ⟨false, false⟩ state then "1" else "0"

end NeuralReversi.Game.Terminal.Fixture

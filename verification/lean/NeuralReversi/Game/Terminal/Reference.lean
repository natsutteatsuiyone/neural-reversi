import NeuralReversi.Game.Terminal.Model

namespace NeuralReversi.Game.Terminal
open NeuralReversi.Game.History

def valueProjection (config : Config) (state : ValueState) : Bool :=
  NeuralReversi.Board.isGameOver config state.board

theorem value_refinement (config : Config) (state : State) :
    isGameOver config state = valueProjection config (VecView.projectState state) := rfl

end NeuralReversi.Game.Terminal

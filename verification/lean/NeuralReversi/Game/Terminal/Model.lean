import NeuralReversi.Game.History.Model
import NeuralReversi.Board.MovesModel

namespace NeuralReversi.Game.Terminal
open NeuralReversi.Game.History

abbrev Config := NeuralReversi.Bitboard.Moves.X86Config

def isGameOver (config : Config) (state : State) : Bool :=
  NeuralReversi.Board.isGameOver config state.board

end NeuralReversi.Game.Terminal

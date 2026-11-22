import NeuralReversi.Game.History.Model
import NeuralReversi.Board.InitializationModel

namespace NeuralReversi.Game.Initialization
open NeuralReversi.Game.History

def emptyVector : VecView.CheckedVec entryBytes :=
  ⟨⟨[], 0, ⟨0, 8, 0⟩⟩, by
    change VecView.Valid 24 ⟨[], 0, ⟨0, 8, 0⟩⟩
    unfold VecView.Valid VecView.AllocationValid
    decide⟩

def fromBoard (board : Board) (side : Disc) : State := ⟨board, side, emptyVector⟩

def newGame : State := fromBoard NeuralReversi.Board.newBoard .black


end NeuralReversi.Game.Initialization

import NeuralReversi.Game.History.Model

namespace NeuralReversi.Game.Queries
open NeuralReversi.Game.History

def actionSquare : Action → Option Square
  | .play square => some square
  | .explicitPass | .automaticPass => none

def lastEntry (state : State) : Option Entry :=
  if empty : state.history.val.lengthWord = 0 then none
  else some (state.history.val.entries.getLast
    (VecView.nonempty_of_nonzero entryBytes state.history.val state.history.property empty))

def lastMove (state : State) : Option Square :=
  (lastEntry state).bind (fun entry => actionSquare entry.action)

def score (state : State) : BitVec 32 × BitVec 32 :=
  if state.side = .black then
    (NeuralReversi.Board.playerCount state.board,NeuralReversi.Board.opponentCount state.board)
  else
    (NeuralReversi.Board.opponentCount state.board,NeuralReversi.Board.playerCount state.board)

end NeuralReversi.Game.Queries

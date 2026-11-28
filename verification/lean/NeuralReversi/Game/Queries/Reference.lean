module
public import NeuralReversi.Game.Queries.Model

@[expose] public section

namespace NeuralReversi.Game.Queries
open NeuralReversi.Game.History

def valueLast (state : ValueState) : Option Square :=
  match state.history.reverse with
  | [] => none
  | entry :: _ => actionSquare entry.action

def valueScore (state : ValueState) : BitVec 32 × BitVec 32 :=
  if state.side = .black then
    (NeuralReversi.Board.playerCount state.board,NeuralReversi.Board.opponentCount state.board)
  else
    (NeuralReversi.Board.opponentCount state.board,NeuralReversi.Board.playerCount state.board)

end NeuralReversi.Game.Queries

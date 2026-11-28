module
public import NeuralReversi.Board.MovesModel
public import NeuralReversi.Board.InitializationModel
public import NeuralReversi.Flip.Dispatch.Model

@[expose] public section

namespace NeuralReversi.Perft

def countRaw (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config)
    (board : Board.Board) (depth : Nat) (canPass : Bool := true) : Nat :=
  match depth with
  | 0 => 1
  | nextDepth + 1 =>
    let moves := (List.finRange 64).filter fun sq => (Board.getMoves mc board).getLsbD sq.val
    match moves with
    | [] =>
      if canPass then
        let next := Board.switchPlayers board
        if Board.hasLegalMoves mc next then countRaw mc fc next (nextDepth + 1) false
        else 1
      else 1
    | first :: rest =>
      let step (sq : Bitboard.Square) :=
        let flipped := Flip.Dispatch.resultValue fc sq board.player board.opponent
        countRaw mc fc (Board.place board flipped sq) nextDepth true
      step first + (rest.map step).sum
termination_by (depth, if canPass then 1 else 0)

end NeuralReversi.Perft

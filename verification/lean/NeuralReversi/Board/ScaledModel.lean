module
public import NeuralReversi.Board.ScoreModel
public import NeuralReversi.ScaledScore.Model

@[expose] public section

namespace NeuralReversi.Board

def finalScoreScaled (checked : Bool) (board : Board) :
    Except String ScaledScore.ScaledScore :=
  match finalScore checked board with
  | .error reason => .error reason
  | .ok score => ScaledScore.fromDiscDiff checked score

def solveScaled (checked : Bool) (board : Board) (empties : BitVec 32) :
    Except String ScaledScore.ScaledScore :=
  match solve checked board empties with
  | .error reason => .error reason
  | .ok score => ScaledScore.fromDiscDiff checked score

end NeuralReversi.Board

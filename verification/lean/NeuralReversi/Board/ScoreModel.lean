import NeuralReversi.Board.Model

namespace NeuralReversi.Board

abbrev Score := BitVec 32

def discScore (b : Board) : Score := playerCount b * 2 - 64

def finalScore (checked : Bool) (b : Board) : Except String Score :=
  if checked && emptySquares b != 0 then
    .error "final_score requires no empty squares; use solve() for boards with empties"
  else .ok (discScore b)

def finishScore (score empties : Score) : Score :=
  let diff := score + empties
  if diff.toInt = 0 then diff
  else if diff.toInt > 0 then diff + empties
  else score

def solve (checked : Bool) (b : Board) (empties : BitVec 32) : Except String Score :=
  if checked && empties != emptyCount b then
    .error s!"n_empties ({empties.toNat}) does not match actual empty count ({(emptyCount b).toNat})"
  else .ok (finishScore (discScore b) empties)

end NeuralReversi.Board

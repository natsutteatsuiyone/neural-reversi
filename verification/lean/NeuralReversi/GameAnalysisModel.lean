module
public import NeuralReversi.Game.Transitions.Model
public import NeuralReversi.Board.ScoreProperties
public import NeuralReversi.MidgameScoreRange.Domain

@[expose] public section

namespace NeuralReversi.GameAnalysis

open Board (Board)
open Game.Transitions (MovesConfig FlipConfig GameStateError isLegalMove nextBoardAfterPlay)
open ScaledScore (ScaledScore)

inductive Action where
  | play (square : Square.Square)
  | pass

inductive Step where
  | play (before : Board)
  | pass

def replayStep (mc : MovesConfig) (fc : FlipConfig) (board : Board) :
    Action → Except GameStateError (Step × Board)
  | .play sq =>
    if isLegalMove mc board sq then .ok (.play board, nextBoardAfterPlay fc board sq)
    else .error (.illegalMove sq)
  | .pass =>
    if Board.isGameOver mc board then .error .passAfterGameOver
    else if Board.hasLegalMoves mc board then .error .passWithLegalMoves
    else .ok (.pass, Board.switchPlayers board)

def replay (mc : MovesConfig) (fc : FlipConfig) (board : Board) :
    List Action → Except GameStateError (List Step × Board)
  | [] => .ok ([], board)
  | action :: rest =>
    match replayStep mc fc board action with
    | .error err => .error err
    | .ok (step, next) =>
      (replay mc fc next rest).map fun (steps, final) => (step :: steps, final)

def checkedScore (score : ScaledScore) : Except Int ScaledScore :=
  if MidgameScoreRange.isFinite score.raw then .ok score else .error score.raw.toInt

inductive Seed where
  | searchable
  | forcedPass
  | terminal
  deriving DecidableEq

def classifySeed (mc : MovesConfig) (board : Board) : Seed :=
  if Board.hasLegalMoves mc board then .searchable
  else if Board.isGameOver mc board then .terminal
  else .forcedPass

inductive Outcome where
  | complete
  | cancelled
  | searchError (message : String)
  | invalidScore (raw : Int)
  deriving DecidableEq

structure Progress where
  index : Nat
  best : Int
  played : Int
  loss : Int
  bestMove : Square.Square
  pv : List Square.Square
  deriving DecidableEq

structure Report where
  progress : List Progress
  outcome : Outcome
  deriving DecidableEq

structure SearchResult where
  score : ScaledScore
  bestMove : Square.Square
  pv : List Square.Square

abbrev Search := Nat → Board → Except String SearchResult

def backward (search : Search) (cancelled : Nat → Bool) (poll : Nat) (score : Int) :
    List (Nat × Step) → Report
  | [] => ⟨[], .complete⟩
  | (i, step) :: rest =>
    if cancelled poll then ⟨[], .cancelled⟩
    else match step with
    | .pass => backward search cancelled (poll + 1) (-score) rest
    | .play before =>
      match search i before with
      | .error message => ⟨[], .searchError message⟩
      | .ok value =>
        if cancelled (poll + 1) then ⟨[], .cancelled⟩
        else match checkedScore value.score with
        | .error raw => ⟨[], .invalidScore raw⟩
        | .ok valid =>
          let best := valid.raw.toInt
          let tail := backward search cancelled (poll + 2) (max best (-score)) rest
          {tail with progress :=
            ⟨i, best, -score, max (best + score) 0, value.bestMove, value.pv⟩ :: tail.progress}

def searchSeed (search : Board → Except String SearchResult) (cancelled : Nat → Bool)
    (board : Board) : Except Outcome Int :=
  match search board with
  | .error message => .error (.searchError message)
  | .ok value =>
    if cancelled 1 then .error .cancelled
    else match checkedScore value.score with
    | .ok valid => .ok valid.raw.toInt
    | .error raw => .error (.invalidScore raw)

def reportGame (mc : MovesConfig) (fc : FlipConfig)
    (search : Option Nat → Board → Except String SearchResult) (cancelled : Nat → Bool)
    (board : Board) (actions : List Action) : Except GameStateError Report :=
  (replay mc fc board actions).map fun (steps, final) =>
    if cancelled 0 then ⟨[], .cancelled⟩
    else
      let reversed := steps.zipIdx.reverse.map fun (step, index) => (index, step)
      let analyzeSteps := backward (fun index => search (some index)) cancelled
      match classifySeed mc final with
      | .searchable =>
        match searchSeed (search none) cancelled final with
        | .ok score => analyzeSteps 2 score reversed
        | .error outcome => ⟨[], outcome⟩
      | .forcedPass =>
        match searchSeed (search none) cancelled (Board.switchPlayers final) with
        | .ok score => analyzeSteps 2 (-score) reversed
        | .error outcome => ⟨[], outcome⟩
      | .terminal => analyzeSteps 1 (Board.AwardedScore final * 256) reversed

end NeuralReversi.GameAnalysis

import NeuralReversi.Game.History.Types
import NeuralReversi.Board.InitializationModel
import NeuralReversi.Board.MovesModel
import NeuralReversi.Flip.Dispatch.Properties

namespace NeuralReversi.Game.Transitions
open NeuralReversi.Game.History

abbrev MovesConfig := NeuralReversi.Bitboard.Moves.X86Config
abbrev FlipConfig := NeuralReversi.Flip.Dispatch.X86Config

inductive GameStateError where
  | illegalMove (square : Square)
  | illegalMoveAt (position : Nat) (square : Square)
  | passAfterGameOver
  | passWithLegalMoves
  deriving DecidableEq, Repr

def newValueState : ValueState :=
  { board := NeuralReversi.Board.newBoard
    side := .black
    history := [] }

def isLegalMove (movesConfig : MovesConfig) (b : Board) (sq : Square) : Bool :=
  (NeuralReversi.Board.getMoves movesConfig b).getLsbD sq.val

def nextBoardAfterPlay (flipConfig : FlipConfig) (b : Board) (sq : Square) : Board :=
  let flipped := NeuralReversi.Flip.Dispatch.resultValue flipConfig sq b.player b.opponent
  NeuralReversi.Board.place b flipped sq

def applyAction (flipConfig : FlipConfig) (state : ValueState) (action : Action) : ValueState :=
  let nextBoard := match action with
    | .play sq => nextBoardAfterPlay flipConfig state.board sq
    | .explicitPass | .automaticPass => NeuralReversi.Board.switchPlayers state.board
  { board := nextBoard
    side := state.side.opposite
    history := state.history ++ [⟨action, state.board, state.side⟩] }

def applyPass (state : ValueState) (action : Action) : ValueState :=
  { board := NeuralReversi.Board.switchPlayers state.board
    side := state.side.opposite
    history := state.history ++ [⟨action, state.board, state.side⟩] }

def makeMove (movesConfig : MovesConfig) (flipConfig : FlipConfig)
    (state : ValueState) (sq : Square) : Except GameStateError ValueState :=
  if !isLegalMove movesConfig state.board sq then
    .error (.illegalMove sq)
  else
    let afterPlay := applyAction flipConfig state (.play sq)
    if !NeuralReversi.Board.hasLegalMoves movesConfig afterPlay.board &&
       NeuralReversi.Board.hasLegalMoves movesConfig
         (NeuralReversi.Board.switchPlayers afterPlay.board) then
      .ok (applyAction flipConfig afterPlay .automaticPass)
    else
      .ok afterPlay

def makePass (movesConfig : MovesConfig) (state : ValueState) :
    Except GameStateError ValueState :=
  if NeuralReversi.Board.isGameOver movesConfig state.board then
    .error .passAfterGameOver
  else if NeuralReversi.Board.hasLegalMoves movesConfig state.board then
    .error .passWithLegalMoves
  else
    .ok (applyPass state .explicitPass)

def replayMoves (movesConfig : MovesConfig) (flipConfig : FlipConfig)
    (startIndex : Nat) (state : ValueState) : List Square → Except GameStateError ValueState
  | [] => .ok state
  | sq :: rest =>
    match makeMove movesConfig flipConfig state sq with
    | .ok next => replayMoves movesConfig flipConfig (startIndex + 1) next rest
    | .error _ => .error (.illegalMoveAt (startIndex + 1) sq)

def fromMoves (movesConfig : MovesConfig) (flipConfig : FlipConfig)
    (moves : List Square) : Except GameStateError ValueState :=
  replayMoves movesConfig flipConfig 0 newValueState moves

end NeuralReversi.Game.Transitions

import NeuralReversi.Game.History.Model
import NeuralReversi.Board.TextModel

namespace NeuralReversi.Game.Initialization
open NeuralReversi.Game.History

def EmptyHistorySpec (observed : State) : Prop :=
  observed.history.val.entries = [] ∧ observed.history.val.lengthWord = 0 ∧
  observed.history.val.allocation.capacityWord = 0 ∧
  observed.history.val.allocation.address = 8

def FromBoardSpec (board : Board) (side : Disc) (observed : State) : Prop :=
  observed.board = board ∧ observed.side = side ∧ EmptyHistorySpec observed

def initialCell (row file : Nat) : NeuralReversi.Disc.Disc :=
  if (row = 3 ∧ file = 4) ∨ (row = 4 ∧ file = 3) then .black
  else if (row = 3 ∧ file = 3) ∨ (row = 4 ∧ file = 4) then .white
  else .empty

def NewGameSpec (observed : State) : Prop :=
  observed.side = .black ∧ EmptyHistorySpec observed ∧
  NeuralReversi.Board.wellFormed observed.board ∧
  (NeuralReversi.Board.playerCount observed.board).toNat = 2 ∧
  (NeuralReversi.Board.opponentCount observed.board).toNat = 2 ∧
  (NeuralReversi.Board.emptyCount observed.board).toNat = 60 ∧
  ∀ (square : NeuralReversi.Bitboard.Square) (checked : Bool),
    NeuralReversi.Board.discAt checked observed.board square .black =
      .ok (initialCell (square.val / 8) (square.val % 8))


end NeuralReversi.Game.Initialization

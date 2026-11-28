module
public import NeuralReversi.Game.Initialization.Model
public import NeuralReversi.Game.Initialization.Spec
public import NeuralReversi.Board.InitializationProperties

@[expose] public section

namespace NeuralReversi.Game.Initialization
open NeuralReversi.Game.History

def FromBoardContract : Prop := ∀ board side, FromBoardSpec board side (fromBoard board side)

def NewGameContract : Prop := NewGameSpec newGame

theorem fromBoard_contract : FromBoardContract := by
  intro board side
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem newGame_contract : NewGameContract := by
  have board := NeuralReversi.Board.initialBoardContract_proven
    NeuralReversi.Board.newBoard (by simp)
  exact ⟨rfl, ⟨rfl, rfl, rfl, rfl⟩, board.1, board.2.1,
    board.2.2.1, board.2.2.2.1, board.2.2.2.2⟩

theorem fromBoard_injective (first second : Board) (a b : Disc)
    (equal : fromBoard first a = fromBoard second b) : first = second ∧ a = b :=
  ⟨congrArg VecView.StoredState.board equal, congrArg VecView.StoredState.side equal⟩

theorem no_constant_fromBoard (constant : State) :
    ¬ ∀ board side, FromBoardSpec board side constant := by
  intro fulfills
  have black := (fulfills ⟨0, 0⟩ .black).2.1
  have white := (fulfills ⟨0, 0⟩ .white).2.1
  rw [black] at white
  cases white

theorem fromBoard_value (board : Board) (side : Disc) :
    VecView.projectState (fromBoard board side) = ⟨board, side, []⟩ := rfl

theorem no_fixed_board (constant : Board) :
    ¬ ∀ board side, FromBoardSpec board side (fromBoard constant side) := by
  intro fulfills
  have first := (fulfills ⟨0, 0⟩ .black).1
  have second := (fulfills ⟨1, 0⟩ .black).1
  have bad := congrArg NeuralReversi.Board.Board.player (first.symm.trans second)
  change (0 : BitVec 64) = 1 at bad
  have distinct : (0 : BitVec 64) ≠ 1 := by decide
  exact distinct bad

theorem no_fixed_side (constant : Disc) :
    ¬ ∀ board side, FromBoardSpec board side (fromBoard board constant) := by
  intro fulfills
  have black := (fulfills ⟨0, 0⟩ .black).2.1
  have white := (fulfills ⟨0, 0⟩ .white).2.1
  rw [black] at white
  cases white

theorem reject_empty_initial_board : ¬ NewGameSpec (fromBoard ⟨0, 0⟩ .black) := by
  intro fulfills
  have count := fulfills.2.2.2.1
  have emptyCount : (NeuralReversi.Board.playerCount
      (fromBoard ⟨0, 0⟩ .black).board).toNat = 0 := by decide
  rw [emptyCount] at count
  omega

theorem reject_swapped_initial_board : ¬ NewGameSpec
    (fromBoard ⟨NeuralReversi.Board.newBoard.opponent,
      NeuralReversi.Board.newBoard.player⟩ .black) := by
  intro fulfills
  have cell := fulfills.2.2.2.2.2.2 ⟨35, by decide⟩ false
  have rejected : NeuralReversi.Board.discAt false
      (fromBoard ⟨NeuralReversi.Board.newBoard.opponent,
        NeuralReversi.Board.newBoard.player⟩ .black).board ⟨35, by decide⟩ .black ≠
      .ok (initialCell (35 / 8) (35 % 8)) := by
    change (Except.ok (.white : NeuralReversi.Disc.Disc) : Except String NeuralReversi.Disc.Disc) ≠ .ok .black
    intro equal
    cases equal
  exact rejected cell


end NeuralReversi.Game.Initialization

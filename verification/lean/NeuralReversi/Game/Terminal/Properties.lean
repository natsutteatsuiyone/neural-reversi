module
public import NeuralReversi.Game.Terminal.Reference
public import NeuralReversi.Game.Terminal.Spec
public import NeuralReversi.Board.MovesProperties

@[expose] public section

namespace NeuralReversi.Game.Terminal
open NeuralReversi.Game.History

def FullContract : Prop := ∀ config (state : State),
  GameOverSpec (VecView.projectState state) (isGameOver config state)

theorem full_contract : FullContract := by
  intro config state
  exact NeuralReversi.Board.isGameOver_exact config state.board

theorem spec_unique (state : ValueState) (first second : Bool)
    (a : GameOverSpec state first) (b : GameOverSpec state second) : first = second := by
  cases first <;> cases second <;> simp_all [GameOverSpec]

private def emptyHistoryState (board : Board) : State :=
  ⟨board, .black, ⟨⟨[], 0, ⟨1, 8, 0⟩⟩, by
    change VecView.Valid 24 ⟨[], 0, ⟨1, 8, 0⟩⟩
    unfold VecView.Valid VecView.AllocationValid
    decide⟩⟩

theorem no_constant (constant : Bool) :
    ¬ ∀ state : State, GameOverSpec (VecView.projectState state) constant := by
  intro fulfills
  let terminal := emptyHistoryState ⟨0, 0⟩
  let playable := emptyHistoryState ⟨1, 2⟩
  have terminalValue : isGameOver ⟨false, false⟩ terminal = true := by decide
  have playableValue : isGameOver ⟨false, false⟩ playable = false := by decide
  have a := spec_unique _ _ _ (fulfills terminal) (full_contract ⟨false, false⟩ terminal)
  have b := spec_unique _ _ _ (fulfills playable) (full_contract ⟨false, false⟩ playable)
  rw [terminalValue] at a
  rw [playableValue] at b
  cases constant <;> simp_all

end NeuralReversi.Game.Terminal

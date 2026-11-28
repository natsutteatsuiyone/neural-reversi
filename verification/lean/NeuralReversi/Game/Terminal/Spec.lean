module
public import NeuralReversi.Game.History.Types
public import NeuralReversi.Bitboard.MovesSpec

@[expose] public section

namespace NeuralReversi.Game.Terminal
open NeuralReversi.Game.History

def GameOverSpec (state : ValueState) (observed : Bool) : Prop :=
  observed = true ↔
    (∀ square, ¬ NeuralReversi.Bitboard.Moves.LegalMove
      state.board.player state.board.opponent square) ∧
    (∀ square, ¬ NeuralReversi.Bitboard.Moves.LegalMove
      state.board.opponent state.board.player square)

end NeuralReversi.Game.Terminal

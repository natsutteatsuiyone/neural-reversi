module
public import NeuralReversi.Game.History.Types

@[expose] public section

namespace NeuralReversi.Game.Queries
open NeuralReversi.Game.History

def ActionSquareSpec (action : Action) (observed : Option Square) : Prop :=
  (∀ square, observed = some square ↔ action = .play square) ∧
  (observed = none ↔ action = .explicitPass ∨ action = .automaticPass)

def LastMoveSpec (state : ValueState) (observed : Option Square) : Prop :=
  (state.history = [] → observed = none) ∧
  (∀ earlier entry, state.history = earlier ++ [entry] →
    (∀ square, entry.action = .play square → observed = some square) ∧
    (entry.action = .explicitPass ∨ entry.action = .automaticPass → observed = none))

def ScoreSpec (state : ValueState) (observed : BitVec 32 × BitVec 32) : Prop :=
  let black := if state.side = .black then state.board.player else state.board.opponent
  let white := if state.side = .black then state.board.opponent else state.board.player
  observed.1.toNat = (List.range 64).countP black.getLsbD ∧
  observed.2.toNat = (List.range 64).countP white.getLsbD ∧
  observed.1.toNat ≤ 64 ∧ observed.2.toNat ≤ 64

end NeuralReversi.Game.Queries

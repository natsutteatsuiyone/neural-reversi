module
public import NeuralReversi.Game.History.Types

@[expose] public section

namespace NeuralReversi.Game.History

def restore (state : ValueState) (entry : Entry) (history : List Entry) : ValueState :=
  {state with board := entry.board, side := entry.side, history := history}

def UndoSpec (state : ValueState) (observed : Bool × ValueState) : Prop :=
  (state.history = [] → observed = (false,state)) ∧
  (∀ earlier entry, state.history = earlier ++ [entry] →
    (entry.action ≠ .automaticPass → observed = (true,restore state entry earlier)) ∧
    (entry.action = .automaticPass →
      (earlier = [] → observed = (true,restore state entry [])) ∧
      (∀ initial before, earlier = initial ++ [before] →
        observed = (true,restore state before initial))))

def ConfirmSpec (state : ValueState) (side : Disc) (observed : Bool × ValueState) : Prop :=
  (state.history = [] → observed = (false,state)) ∧
  (∀ earlier entry, state.history = earlier ++ [entry] →
    (entry.action = .automaticPass ∧ entry.side = side →
      observed = (true,{state with history := earlier ++ [{entry with action := .explicitPass}]})) ∧
    (¬ (entry.action = .automaticPass ∧ entry.side = side) → observed = (false,state)))

end NeuralReversi.Game.History

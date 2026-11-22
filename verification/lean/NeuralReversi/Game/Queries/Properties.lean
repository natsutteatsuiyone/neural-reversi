import NeuralReversi.Game.Queries.Spec
import NeuralReversi.Game.Queries.Reference
import NeuralReversi.Game.History.Properties
import NeuralReversi.Board.Counting

namespace NeuralReversi.Game.Queries
open NeuralReversi.Game.History

theorem actionSquare_full (action : Action) : ActionSquareSpec action (actionSquare action) := by
  cases action <;> simp [actionSquare,ActionSquareSpec]

theorem lastMove_full (state : State) :
    LastMoveSpec (VecView.projectState state) (lastMove state) := by
  constructor
  · intro empty
    have zero : state.history.val.lengthWord = 0 := by
      apply BitVec.eq_of_toNat_eq
      have length := state.history.property.1
      change state.history.val.entries = [] at empty
      simpa [empty] using length
    simp [lastMove,lastEntry,zero]
  · intro earlier entry history
    change state.history.val.entries = earlier ++ [entry] at history
    have nonzero : state.history.val.lengthWord ≠ 0 := by
      intro zero
      have length := state.history.property.1
      simp [zero,history] at length
    simp only [lastMove,lastEntry,dite_eq_right nonzero]
    simp only [history,List.getLast_append,List.getLast_singleton,Option.bind_some]
    constructor
    · intro square play
      simp [play,actionSquare]
    · intro pass
      rcases pass with explicit | automatic
      · simp [explicit,actionSquare]
      · simp [automatic,actionSquare]

theorem lastMoveSpec_unique (state : ValueState) (first second : Option Square)
    (a : LastMoveSpec state first) (b : LastMoveSpec state second) : first = second := by
  rcases Reference.suffix_cases state.history with empty | ⟨earlier,last,history⟩
  · exact (a.1 empty).trans (b.1 empty).symm
  · cases action : last.action with
    | play square =>
      exact ((a.2 earlier last history).1 square action).trans
        ((b.2 earlier last history).1 square action).symm
    | explicitPass =>
      exact ((a.2 earlier last history).2 (Or.inl action)).trans
        ((b.2 earlier last history).2 (Or.inl action)).symm
    | automaticPass =>
      exact ((a.2 earlier last history).2 (Or.inr action)).trans
        ((b.2 earlier last history).2 (Or.inr action)).symm

theorem score_full (state : State) : ScoreSpec (VecView.projectState state) (score state) := by
  have player := NeuralReversi.Bitboard.populationCountContract_proven state.board.player
  have opponent := NeuralReversi.Bitboard.populationCountContract_proven state.board.opponent
  by_cases black : state.side = .black
  · simpa [ScoreSpec,score,VecView.projectState,VecView.project,black,
      NeuralReversi.Board.playerCount,NeuralReversi.Board.opponentCount] using
      And.intro player.1 (And.intro opponent.1 (And.intro player.2.1 opponent.2.1))
  · simpa [ScoreSpec,score,VecView.projectState,VecView.project,black,
      NeuralReversi.Board.playerCount,NeuralReversi.Board.opponentCount] using
      And.intro opponent.1 (And.intro player.1 (And.intro opponent.2.1 player.2.1))

theorem valueLast_full (state : ValueState) : LastMoveSpec state (valueLast state) := by
  constructor
  · intro empty
    simp [valueLast,empty]
  · intro earlier entry history
    constructor
    · intro square play
      simp [valueLast,history,play,actionSquare]
    · intro pass
      rcases pass with explicit | automatic
      · simp [valueLast,history,explicit,actionSquare]
      · simp [valueLast,history,automatic,actionSquare]

theorem lastMove_value_refinement (state : State) :
    lastMove state = valueLast (VecView.projectState state) :=
  lastMoveSpec_unique _ _ _ (lastMove_full state) (valueLast_full _)

theorem score_value_refinement (state : State) :
    score state = valueScore (VecView.projectState state) := rfl

theorem valueLast_confirm_invariant (state : ValueState) (side : Disc) :
    valueLast (Reference.confirm state side).2 = valueLast state := by
  rcases Reference.suffix_cases state.history with empty | ⟨earlier,last,history⟩
  · simp [Reference.confirm,empty,Reference.popLast]
  · by_cases qualifies : last.action = .automaticPass ∧ last.side = side
    · simp [Reference.confirm,history,Reference.popLast_append,qualifies,valueLast,actionSquare]
    · simp [Reference.confirm,history,Reference.popLast_append,qualifies]

theorem valueScore_confirm_invariant (state : ValueState) (side : Disc) :
    valueScore (Reference.confirm state side).2 = valueScore state := by
  rcases Reference.suffix_cases state.history with empty | ⟨earlier,last,history⟩
  · simp [Reference.confirm,empty,Reference.popLast]
  · by_cases qualifies : last.action = .automaticPass ∧ last.side = side
    · simp [Reference.confirm,history,Reference.popLast_append,qualifies,valueScore]
    · simp [Reference.confirm,history,Reference.popLast_append,qualifies]

theorem lastMove_confirm_invariant (state : State) (side : Disc) :
    lastMove (confirmAutomaticPass state side).2 = lastMove state := by
  rw [lastMove_value_refinement,lastMove_value_refinement]
  have projection := congrArg Prod.snd
    (VecView.confirmState_value_refinement entryBytes state side)
  change VecView.projectState (confirmAutomaticPass state side).2 =
    (Reference.confirm (VecView.projectState state) side).2 at projection
  rw [projection]
  exact valueLast_confirm_invariant _ _

theorem score_confirm_invariant (state : State) (side : Disc) :
    score (confirmAutomaticPass state side).2 = score state := by
  rw [score_value_refinement,score_value_refinement]
  have projection := congrArg Prod.snd
    (VecView.confirmState_value_refinement entryBytes state side)
  change VecView.projectState (confirmAutomaticPass state side).2 =
    (Reference.confirm (VecView.projectState state) side).2 at projection
  rw [projection]
  exact valueScore_confirm_invariant _ _

def ActionSquareContract : Prop := ∀ action, ActionSquareSpec action (actionSquare action)

theorem actionSquare_contract : ActionSquareContract := actionSquare_full

def LastMoveContract : Prop := ∀ state : State,
  LastMoveSpec (VecView.projectState state) (lastMove state)

theorem lastMove_contract : LastMoveContract := lastMove_full

def ScoreContract : Prop := ∀ state : State,
  ScoreSpec (VecView.projectState state) (score state)

theorem score_contract : ScoreContract := score_full

end NeuralReversi.Game.Queries

module
public import NeuralReversi.Game.History.Model

@[expose] public section

namespace NeuralReversi.Game.History.VecView

open NeuralReversi.Game.History.Reference

theorem undo_allocation_frame (entryBytes : Nat) (state : StoredState entryBytes) :
    (undo entryBytes state).2.history.val.allocation = state.history.val.allocation := by
  have frame := typedPop_frame entryBytes state.history
  unfold undo
  generalize found : typedPop entryBytes state.history = first at frame ⊢
  rcases first with ⟨entry,rest⟩
  cases entry with
  | none => exact frame
  | some entry =>
    dsimp only
    cases entry.action with
    | play _ => exact frame
    | explicitPass => exact frame
    | automaticPass => exact (typedPop_frame entryBytes rest).trans frame

theorem undo_value_refinement (entryBytes : Nat) (state : StoredState entryBytes) :
    ((undo entryBytes state).1,projectState (undo entryBytes state).2) =
      NeuralReversi.Game.History.Reference.undo (projectState state) := by
  have firstProjection := typedPop_projection entryBytes state.history
  unfold undo
  generalize found : typedPop entryBytes state.history = first at firstProjection ⊢
  rcases first with ⟨entry,rest⟩
  cases firstValue : popLast state.history.val.entries with
  | none =>
    simp only [listPop,firstValue] at firstProjection
    have value := congrArg Prod.fst firstProjection
    have history := congrArg Prod.snd firstProjection
    dsimp only at value history
    subst entry
    simp [projectState,project,NeuralReversi.Game.History.Reference.undo,firstValue,history]
  | some pair =>
    rcases pair with ⟨last,earlier⟩
    simp only [listPop,firstValue] at firstProjection
    have value := congrArg Prod.fst firstProjection
    have history := congrArg Prod.snd firstProjection
    dsimp only at value history
    subst entry
    dsimp only
    cases action : last.action with
    | play _ =>
      simp [projectState,project,NeuralReversi.Game.History.Reference.undo,firstValue,action,history,finish,restore]
    | explicitPass =>
      simp [projectState,project,NeuralReversi.Game.History.Reference.undo,firstValue,action,history,finish,restore]
    | automaticPass =>
      have secondProjection := typedPop_projection entryBytes rest
      generalize secondFound : typedPop entryBytes rest = second at secondProjection ⊢
      rcases second with ⟨before,initial⟩
      cases secondValue : popLast earlier with
      | none =>
        simp only [history,listPop,secondValue] at secondProjection
        have value := congrArg Prod.fst secondProjection
        have tail := congrArg Prod.snd secondProjection
        dsimp only at value tail
        subst before
        simp [projectState,project,NeuralReversi.Game.History.Reference.undo,firstValue,action,secondValue,tail,finish,restore]
      | some pair =>
        rcases pair with ⟨previous,prefixEntries⟩
        simp only [history,listPop,secondValue] at secondProjection
        have value := congrArg Prod.fst secondProjection
        have tail := congrArg Prod.snd secondProjection
        dsimp only at value tail
        subst before
        simp [projectState,project,NeuralReversi.Game.History.Reference.undo,firstValue,action,secondValue,tail,finish,restore]

def UndoContract : Prop := ∀ entryBytes (state : StoredState entryBytes),
  UndoSpec (projectState state) ((undo entryBytes state).1,projectState (undo entryBytes state).2) ∧
  (undo entryBytes state).2.history.val.allocation = state.history.val.allocation

theorem undo_contract : UndoContract := by
  intro entryBytes state
  refine ⟨?_,undo_allocation_frame entryBytes state⟩
  rw [undo_value_refinement]
  exact NeuralReversi.Game.History.Reference.undo_full (projectState state)

theorem confirmState_value_refinement (entryBytes : Nat) (state : StoredState entryBytes) (side : Disc) :
    ((confirmState entryBytes state side).1,projectState (confirmState entryBytes state side).2) =
      NeuralReversi.Game.History.Reference.confirm (projectState state) side :=
  confirm_value_refinement entryBytes state.history.val state.history.property state.board state.side side

def ConfirmContract : Prop := ∀ entryBytes (state : StoredState entryBytes) side,
  ConfirmSpec (projectState state) side
    ((confirmState entryBytes state side).1,projectState (confirmState entryBytes state side).2) ∧
  (confirmState entryBytes state side).2.history.val.lengthWord = state.history.val.lengthWord ∧
  (confirmState entryBytes state side).2.history.val.allocation = state.history.val.allocation

theorem confirm_contract : ConfirmContract := by
  intro entryBytes state side
  refine ⟨?_,(confirm_valid_frame entryBytes state.history.val state.history.property side).2⟩
  rw [confirmState_value_refinement]
  exact NeuralReversi.Game.History.Reference.confirm_full (projectState state) side

end NeuralReversi.Game.History.VecView

namespace NeuralReversi.Game.History

def FullUndoContract : Prop := ∀ state : State,
  UndoSpec (VecView.projectState state)
    ((undo state).1,VecView.projectState (undo state).2) ∧
  (undo state).2.history.val.allocation = state.history.val.allocation

theorem fullUndoContract_proven : FullUndoContract := by
  intro state
  exact VecView.undo_contract entryBytes state

def FullConfirmContract : Prop := ∀ (state : State) side,
  ConfirmSpec (VecView.projectState state) side
    ((confirmAutomaticPass state side).1,VecView.projectState (confirmAutomaticPass state side).2) ∧
  (confirmAutomaticPass state side).2.history.val.lengthWord = state.history.val.lengthWord ∧
  (confirmAutomaticPass state side).2.history.val.allocation = state.history.val.allocation

theorem fullConfirmContract_proven : FullConfirmContract := by
  intro state side
  exact VecView.confirm_contract entryBytes state side
end NeuralReversi.Game.History

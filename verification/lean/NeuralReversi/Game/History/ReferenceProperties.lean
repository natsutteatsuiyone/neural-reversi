import NeuralReversi.Game.History.Reference

namespace NeuralReversi.Game.History.Reference

theorem popLast_append (earlier : List Entry) (last : Entry) :
    popLast (earlier ++ [last]) = some (last,earlier) := by
  induction earlier with
  | nil => rfl
  | cons head tail ih =>
    cases tail with
    | nil => rfl
    | cons next tail =>
      change Option.map _ (popLast ((next::tail) ++ [last])) = _
      rw [ih]
      rfl

theorem suffix_cases (history : List Entry) :
    history = [] ∨ ∃ earlier last, history = earlier ++ [last] := by
  have inverse := List.reverse_reverse history
  cases reversed : history.reverse with
  | nil => left; simpa [reversed] using inverse.symm
  | cons last rest =>
    right
    refine ⟨rest.reverse,last,?_⟩
    simpa [reversed] using inverse.symm

theorem undo_full (state : ValueState) : UndoSpec state (undo state) := by
  constructor
  · intro empty
    simp [undo,empty,popLast]
  · intro earlier entry history
    rw [undo,history,popLast_append]
    dsimp only
    constructor
    · intro notAuto
      cases action : entry.action with
      | play square => rfl
      | explicitPass => rfl
      | automaticPass => exact (notAuto action).elim
    · intro auto
      rw [auto]
      constructor
      · intro empty
        subst earlier
        rfl
      · intro initial before prefixEq
        rw [prefixEq,popLast_append]

theorem undoSpec_unique (state : ValueState) (first second : Bool × ValueState)
    (a : UndoSpec state first) (b : UndoSpec state second) : first = second := by
  rcases suffix_cases state.history with empty | ⟨earlier,last,history⟩
  · exact (a.1 empty).trans (b.1 empty).symm
  · by_cases auto : last.action = .automaticPass
    · rcases suffix_cases earlier with empty | ⟨initial,before,prefixEq⟩
      · exact ((a.2 earlier last history).2 auto).1 empty |>.trans
          (((b.2 earlier last history).2 auto).1 empty).symm
      · exact (((a.2 earlier last history).2 auto).2 initial before prefixEq).trans
          (((b.2 earlier last history).2 auto).2 initial before prefixEq).symm
    · exact ((a.2 earlier last history).1 auto).trans ((b.2 earlier last history).1 auto).symm

theorem confirm_full (state : ValueState) (side : Disc) : ConfirmSpec state side (confirm state side) := by
  constructor
  · intro empty
    simp [confirm,empty,popLast]
  · intro earlier entry history
    constructor <;> intro condition <;> simp [confirm,history,popLast_append,condition]

theorem confirm_twice (state : ValueState) (side : Disc) :
    confirm (confirm state side).2 side = (false,(confirm state side).2) := by
  rcases suffix_cases state.history with empty | ⟨earlier,last,history⟩
  · simp [confirm,empty,popLast]
  · by_cases qualifies : last.action = .automaticPass ∧ last.side = side
    · simp [confirm,history,popLast_append,qualifies]
    · simp [confirm,history,popLast_append,qualifies]

theorem confirmed_undo_single (state : ValueState) (earlier : List Entry) (entry : Entry)
    (history : state.history = earlier ++ [entry])
    (auto : entry.action = .automaticPass) :
    let after := (confirm state entry.side).2
    (confirm state entry.side).1 = true ∧ undo after = (true,restore state entry earlier) := by
  simp only [confirm,history,popLast_append,auto,true_and,↓reduceIte]
  simp [undo,popLast_append,restore]

end NeuralReversi.Game.History.Reference

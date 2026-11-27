import NeuralReversi.OuterMir.CompletionProgram

namespace NeuralReversi.OuterMir.Completion

open PublicationMir (Iteration SliceSort SearchResult capture)
open RootMoveStorage (Row)

theorem new_single : (new 1).marks = [none] ∧ (new 1).last = none := ⟨rfl, rfl⟩

theorem beginPhase_last (state : State) : (beginPhase state).last = state.last := rfl

theorem beginPhase_length (state : State) : (beginPhase state).marks.length = state.marks.length := by
  simp [beginPhase]

theorem beginPhase_single (state : State) (one : state.marks.length = 1) :
    beginPhase state = ⟨[none], state.last⟩ := by
  simp [beginPhase, one]

/-- The outer option models a normal return: an invalid index would panic in
    Rust. A valid, unfinished set of lines instead returns `some (_, none)`. -/
theorem lineCompleted_invalid (sort : SliceSort) (rows : List Row) (idx depth level : Nat)
    (endgame : Bool) (state : State) (outside : state.marks.length ≤ idx) :
    lineCompleted sort rows idx depth level endgame state = none := by
  simp [lineCompleted, Nat.not_lt_of_ge outside]

theorem lineCompleted_pending (sort : SliceSort) (rows : List Row) (idx depth level : Nat)
    (endgame : Bool) (state : State) (inside : idx < state.marks.length)
    (pending : (lineStamp state.marks idx depth level).2 = false) :
    lineCompleted sort rows idx depth level endgame state =
      some (⟨(lineStamp state.marks idx depth level).1, state.last⟩, none) := by
  simp [lineCompleted, inside, pending]

/-- A nonempty captured vector makes the native `best_move().expect(...)`
    return normally. The line stamps are untouched by this operation. -/
theorem iterationCompleted_returns (sort : SliceSort) (rows : List Row) (depth level : Nat)
    (endgame : Bool) (state : State) (nonempty : rows ≠ []) :
    ∃ best, iterationCompleted sort rows depth level endgame state =
      some (⟨state.marks, some (capture sort rows depth level endgame)⟩, best) := by
  have present : (sort.byReverseKey PublicationMir.key rows) ≠ [] := by
    intro empty
    have size := (sort.permutation PublicationMir.key rows).length_eq
    rw [empty] at size
    exact nonempty (List.length_eq_zero_iff.mp size.symm)
  obtain ⟨best,tail,sorted⟩ := List.exists_cons_of_ne_nil present
  exact ⟨best,by simp [iterationCompleted,PublicationMir.record,PublicationMir.first,
    PublicationMir.capture,PublicationMir.snapshot,sorted]⟩

/-- For the single native PV line, writing its current stamp makes every mark
    equal. Its completed iteration is captured immediately, for any previous
    mark or inherited snapshot. -/
theorem single_line (sort : SliceSort) (rows : List Row) (depth level : Nat)
    (endgame : Bool) (state : State) (one : state.marks.length = 1) (nonempty : rows ≠ []) :
    ∃ best, lineCompleted sort rows 0 depth level endgame state =
      some (⟨[some (depth, level)], some (capture sort rows depth level endgame)⟩, some best) := by
  obtain ⟨mark,marks⟩ := List.length_eq_one_iff.mp one
  obtain ⟨best,recorded⟩ := iterationCompleted_returns sort rows depth level endgame
    ⟨[some (depth,level)],state.last⟩ nonempty
  exact ⟨best,by simp [lineCompleted,marks,lineStamp,recorded]⟩

/-- This identity also covers an empty vector: both expressions have no normal
    result in that case. No property of the ordering of ties is needed. -/
theorem completed_eq_intoResult (sort : SliceSort) (rows : List Row) (depth level : Nat)
    (endgame : Bool) (nodes : Nat) (fallback : Unit → SearchResult) :
    PublicationMir.completed sort rows depth level endgame nodes fallback =
      PublicationMir.intoResult (some (capture sort rows depth level endgame)) nodes fallback := by
  rw [PublicationMir.completed_fromSnapshot, PublicationMir.intoResult_fromSnapshot]

theorem iterationCompleted_intoResult (sort : SliceSort) (rows : List Row) (depth level : Nat)
    (endgame : Bool) (state : State) (nodes : Nat) (fallback : Unit → SearchResult) :
    (iterationCompleted sort rows depth level endgame state).bind
      (fun result => intoResult result.1 nodes fallback) =
        PublicationMir.completed sort rows depth level endgame nodes fallback := by
  unfold iterationCompleted PublicationMir.completed
  cases PublicationMir.record sort rows depth level endgame <;> simp [intoResult]

/-- The normal final capture repeats the same live rows and metadata as the
    last single-PV line capture. Omitting that repeated capture preserves the
    returned result, including the selected move, PV, metadata and row order. -/
theorem recapture_same (sort : SliceSort) (rows : List Row) (depth level : Nat)
    (endgame : Bool) (state : State) (nodes : Nat) (fallback : Unit → SearchResult)
    (saved : state.last = some (capture sort rows depth level endgame)) :
    (iterationCompleted sort rows depth level endgame state).bind
      (fun result => intoResult result.1 nodes fallback) = intoResult state nodes fallback := by
  rw [iterationCompleted_intoResult, completed_eq_intoResult]
  simp only [intoResult, saved]

theorem normal_finish_same (minimum level : Nat) (sort : SliceSort) (rows : List Row)
    (depth nodes : Nat) (state : State) (fallback : Unit → SearchResult)
    (saved : state.last = some (capture sort rows depth level true)) :
    OuterMir.finish minimum level sort rows depth nodes state.last fallback =
      intoResult state nodes fallback := by
  by_cases reportable : minimum ≤ level
  · simp only [OuterMir.finish, reportable, ↓reduceIte, completed_eq_intoResult, intoResult, saved]
  · simp only [OuterMir.finish, reportable, ↓reduceIte, intoResult]

end NeuralReversi.OuterMir.Completion

import NeuralReversi.Solve34.Dispatch
import NeuralReversi.Solve34.Control

namespace NeuralReversi.Solve34.Dispatch
set_option linter.unusedSimpArgs false

theorem maxRun_score (oracle : Oracle) (alpha : Score) (first : Bool) (best : Score)
    (xs : List (Option Call)) (fallback : Outcome) :
    (maxRun oracle alpha first best xs fallback).score =
      sourceMaxWord alpha first best (xs.map (Option.map (fun c => -(oracle c).score))) fallback.score := by
  induction xs generalizing first best with
  | nil => simp only [maxRun, sourceMaxWord, List.map_nil]; split <;> rfl
  | cons x xs ih =>
    cases xs with
    | nil =>
      cases x with
      | none => simp only [maxRun, sourceMaxWord, List.map_cons, List.map_nil, Option.map]; split <;> rfl
      | some c => rfl
    | cons y ys =>
      cases x with
      | none => simpa only [maxRun, sourceMaxWord, List.map_cons, Option.map] using ih false best
      | some c =>
        simp only [maxRun, sourceMaxWord, List.map_cons, Option.map]
        split
        · rfl
        · exact ih false _

theorem minRun_score (oracle : Oracle) (alpha : Score) (first : Bool) (best : Score)
    (xs : List (Option Call)) (fallback : Outcome) :
    (minRun oracle alpha first best xs fallback).score =
      sourceMinWord alpha first best (xs.map (Option.map (fun c => (oracle c).score))) fallback.score := by
  induction xs generalizing first best with
  | nil => simp only [minRun, sourceMinWord, List.map_nil]; split <;> rfl
  | cons x xs ih =>
    cases xs with
    | nil =>
      cases x with
      | none => simp only [minRun, sourceMinWord, List.map_cons, List.map_nil, Option.map]; split <;> rfl
      | some c => rfl
    | cons y ys =>
      cases x with
      | none => simpa only [minRun, sourceMinWord, List.map_cons, Option.map] using ih false best
      | some c =>
        simp only [minRun, sourceMinWord, List.map_cons, Option.map]
        split
        · rfl
        · exact ih false _

theorem run_score (charge : Bool) (oracle : Oracle) (alpha : Score)
    (playerCalls opponentCalls : List (Option Call)) (terminal : Score) :
    (run charge oracle alpha playerCalls opponentCalls terminal).score =
      solveWord alpha (playerCalls.map (Option.map (fun c => (oracle c).score)))
        (opponentCalls.map (Option.map (fun c => (oracle c).score))) terminal := by
  cases charge <;>
    simp only [run, Bool.false_eq_true, ↓reduceIte, emit, maxRun_score, minRun_score,
      finish, solveWord, List.map_map, Option.map_map, Function.comp_def]

end NeuralReversi.Solve34.Dispatch

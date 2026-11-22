import NeuralReversi.Threading.SplitWindow

namespace NeuralReversi.PvsParallel.Prefix

open Aspiration (SoundResult Window)
open PvsMoveLoop (Bound ChildSearch bestValue moveScore scan scoreInf scoreMax)
open Threading.SplitWindow (State marksCutoff publish stateContract_proven)

def run (pv : Bool) (search : ChildSearch Move) (beta : Int) :
    List Move → Bool → State → Option State
  | [], _, state => some state
  | move :: rest, first, state =>
    let score := moveScore pv first search move state.alpha beta
    if marksCutoff pv beta state score then none
    else run pv search beta rest false (publish pv beta state score)

theorem scan_step (pv : Bool) (search : ChildSearch Move) (beta : Int)
    (move : Move) (rest : List Move) (first : Bool) (state : State) :
    let score := moveScore pv first search move state.alpha beta
    let next := publish pv beta state score
    scan pv search beta (move :: rest) first state.alpha state.best =
      if marksCutoff pv beta state score then score else scan pv search beta rest false next.alpha next.best := by
  let score := moveScore pv first search move state.alpha beta
  change scan pv search beta (move :: rest) first state.alpha state.best =
    if marksCutoff pv beta state score then score
    else scan pv search beta rest false (publish pv beta state score).alpha (publish pv beta state score).best
  rw [scan]
  by_cases improved : state.best < score
  · by_cases raises : state.alpha < score
    · cases pv
      · simp [score, marksCutoff, improved, raises]
      · by_cases below : score < beta
        · by_cases maximum : score < scoreMax
          · simp [score, marksCutoff, publish, improved, raises, below, maximum, Int.not_le.mpr below,
              Int.not_le.mpr maximum, Int.max_eq_right (Int.le_of_lt improved)]
          · simp [score, marksCutoff, improved, raises, maximum, Int.le_of_not_gt maximum]
        · simp [score, marksCutoff, improved, raises, below, Int.le_of_not_gt below]
    · simp [score, marksCutoff, publish, improved, raises, Int.max_eq_right (Int.le_of_lt improved)]
  · simp [score, marksCutoff, publish, improved, Int.max_eq_left (Int.le_of_not_gt improved)]

private theorem continued_scan (pv : Bool) (search : ChildSearch Move) (beta : Int)
    (moves : List Move) (first : Bool) (initial final : State)
    (continued : run pv search beta moves first initial = some final) :
    scan pv search beta moves first initial.alpha initial.best = final.best := by
  induction moves generalizing first initial with
  | nil =>
    have equal := Option.some.inj continued
    exact congrArg State.best equal
  | cons move rest ih =>
    let score := moveScore pv first search move initial.alpha beta
    cases cut : marksCutoff pv beta initial score
    · rw [scan_step]
      change marksCutoff pv beta initial (moveScore pv first search move initial.alpha beta) = false at cut
      simp only [cut, Bool.false_eq_true, ite_false]
      exact ih false _ (by simpa [run, score, cut] using continued)
    · simp [run, score, cut] at continued

private theorem continued_window (pv : Bool) (search : ChildSearch Move) (original beta : Int)
    (moves : List Move) (first : Bool) (initial final : State)
    (window : Window initial.alpha beta) (cap : initial.alpha ≤ max original initial.best)
    (continued : run pv search beta moves first initial = some final) :
    Window final.alpha beta ∧ final.alpha ≤ max original final.best ∧
      (pv = false → final.alpha = initial.alpha) := by
  induction moves generalizing first initial with
  | nil => cases Option.some.inj continued; exact ⟨window, cap, fun _ => rfl⟩
  | cons move rest ih =>
    let score := moveScore pv first search move initial.alpha beta
    cases cut : marksCutoff pv beta initial score
    · let next := publish pv beta initial score
      have state : Window next.alpha beta ∧ next.alpha ≤ max original next.best ∧
          (pv = false → next.alpha = initial.alpha) :=
        (stateContract_proven pv original beta initial [score] window cap).2
      obtain ⟨valid, capped, fixed⟩ := ih false next state.1 state.2.1
        (by simpa [run, score, cut, next] using continued)
      exact ⟨valid, capped, fun noPv => (fixed noPv).trans (state.2.2 noPv)⟩
    · simp [run, score, cut] at continued

def Contract : Prop :=
  ∀ (Move : Type) (pv : Bool) (search : ChildSearch Move) (value : Move → Int) (moves : List Move)
    (original beta : Int) (final : State),
    Window original beta → (pv = false → beta = original + 1) →
    run pv search beta moves true ⟨original, -scoreInf⟩ = some final →
    (∀ move ∈ moves, value move ≤ scoreMax) →
    (∀ move ∈ moves, ∀ childPv a b, Window a b → (childPv = false → b = a + 1) →
      SoundResult (-(value move)) a b (search move childPv a b)) →
    Window final.alpha beta ∧ final.alpha ≤ max original final.best ∧
      (pv = false → beta = final.alpha + 1) ∧
      Bound (bestValue value moves (-scoreInf)) original beta final.best

theorem contract_proven : Contract := by
  intro Move pv search value moves original beta final window null continued range children
  obtain ⟨valid, cap, fixed⟩ := continued_window pv search original beta moves true _
    final window (Int.le_max_left _ _) continued
  refine ⟨valid, cap, fun noPv => by rw [fixed noPv]; exact null noPv, ?_⟩
  have same := continued_scan pv search beta moves true _ final continued
  cases moves with
  | nil =>
    change -scoreInf = final.best at same
    rw [← same]
    exact ⟨fun _ => Int.le_refl _, fun _ => Int.le_refl _⟩
  | cons head tail =>
    rw [← same]
    exact Bound.ofSoundResult
      (PvsMoveLoop.contract_proven Move pv search value (head :: tail) original beta
        (by simp) window null range children)

example : (run true (fun (move : Int) _ _ _ => -move) 20 [4, 1] true
    ⟨0, -scoreInf⟩).map State.alpha = some 4 := rfl

example : run true (fun (move : Int) _ _ _ => -move) 6 [7] true ⟨0, -scoreInf⟩ = none := rfl

example : run false (fun (_ : Nat) _ _ _ => 0) 1 [] true ⟨0, -scoreInf⟩ =
    some ⟨0, -scoreInf⟩ := rfl

end NeuralReversi.PvsParallel.Prefix

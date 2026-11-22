import NeuralReversi.Aspiration

namespace NeuralReversi.MidgameAspiration

def FiniteScore (score : Int) : Prop := -32767 < score ∧ score < 32767

structure History where
  average : Int
  meanSquared : Int
  deriving DecidableEq, Repr

def History.empty : History := ⟨-32767, 0⟩

def squareLimit : Nat := 1073610756

def History.seeded (history : History) : Prop :=
  FiniteScore history.average ∧ history.meanSquared.natAbs ≤ squareLimit

def History.valid (history : History) : Prop := history = .empty ∨ history.seeded

theorem History.bounds {history : History} (valid : history.valid) :
    -32767 ≤ history.average ∧ history.average < 32767 ∧
      history.meanSquared.natAbs ≤ squareLimit := by
  rcases valid with rfl | ⟨⟨lo, hi⟩, mss⟩
  · decide
  · exact ⟨by omega, hi, mss⟩

def signedSquare (score : Int) : Int := score * (score.natAbs : Int)

def History.update (history : History) (score : Int) : History :=
  let squared := signedSquare score
  if history.average = -32767 then ⟨score, squared⟩
  else ⟨(history.average + score).tdiv 2, (history.meanSquared + squared).tdiv 2⟩

private theorem square_bound {score : Int} (finite : FiniteScore score) :
    (signedSquare score).natAbs ≤ squareLimit := by
  obtain ⟨lo, hi⟩ := finite
  have bound : score.natAbs ≤ 32766 := by omega
  simpa [signedSquare, Int.natAbs_mul, squareLimit] using Nat.mul_le_mul bound bound

private theorem half_abs {a b : Int} {limit : Nat}
    (left : a.natAbs ≤ limit) (right : b.natAbs ≤ limit) :
    ((a + b).tdiv 2).natAbs ≤ limit := by
  have sum := Int.natAbs_add_le a b
  rw [Int.natAbs_tdiv]
  change (a + b).natAbs / 2 ≤ limit
  omega

theorem history_updated {history : History} {score : Int}
    (valid : history.valid) (finite : FiniteScore score) : (history.update score).seeded := by
  have square := square_bound finite
  rcases valid with rfl | ⟨avg, mss⟩
  · exact ⟨finite, square⟩
  · obtain ⟨lo, hi⟩ := avg
    have range : history.average ≠ -32767 := by omega
    obtain ⟨slo, shi⟩ := finite
    have average := half_abs (limit := 32766) (a := history.average) (b := score)
      (by omega) (by omega)
    simp only [History.update, range, ite_false, History.seeded]
    exact ⟨⟨by omega, by omega⟩, half_abs mss square⟩

def FitsSigned (bits : Nat) (n : Int) : Prop :=
  -(2 ^ (bits - 1)) ≤ n ∧ n < 2 ^ (bits - 1)

theorem history_storage {history : History} {score : Int}
    (valid : history.valid) (finite : FiniteScore score) :
    FitsSigned 32 (score.natAbs : Int) ∧
      FitsSigned 64 (signedSquare score) ∧
      FitsSigned 32 (history.average + score) ∧
      FitsSigned 64 (history.meanSquared + signedSquare score) := by
  have s := square_bound finite
  obtain ⟨slo, shi⟩ := finite
  unfold squareLimit at s
  rcases valid with rfl | ⟨⟨lo, hi⟩, mss⟩
  · simp only [FitsSigned, History.empty]
    omega
  · unfold squareLimit at mss
    unfold FitsSigned
    omega

def HistoryContract : Prop :=
  ∀ (scores : List Int), (∀ score ∈ scores, FiniteScore score) →
    (scores.foldl History.update History.empty).valid ∧
    (scores ≠ [] → (scores.foldl History.update History.empty).seeded)

theorem historyContract_proven : HistoryContract := by
  have fold_seeded (history : History) (scores : List Int)
      (finite : ∀ score ∈ scores, FiniteScore score) (seeded : history.seeded) :
      (scores.foldl History.update history).seeded := by
    induction scores generalizing history with
    | nil => exact seeded
    | cons head tail ih =>
      exact ih _ (fun score member => finite score (List.mem_cons_of_mem _ member))
        (history_updated (Or.inr seeded) (finite head (by simp)))
  intro scores finite
  cases scores with
  | nil => exact ⟨Or.inl rfl, fun h => (h rfl).elim⟩
  | cons head tail =>
    have seeded := fold_seeded (History.empty.update head) tail
      (fun score member => finite score (List.mem_cons_of_mem _ member))
      (history_updated (Or.inl rfl) (finite head (by simp)))
    exact ⟨Or.inr seeded, fun _ => seeded⟩

def HistoryArithmeticContract : Prop :=
  ∀ (history : History) (score : Int), history.valid → FiniteScore score →
    FitsSigned 32 (score.natAbs : Int) ∧ FitsSigned 64 (signedSquare score) ∧
    FitsSigned 32 (history.average + score) ∧
    FitsSigned 64 (history.meanSquared + signedSquare score) ∧
    (history.update score).seeded

theorem historyArithmeticContract_proven : HistoryArithmeticContract := by
  intro history score valid finite
  obtain ⟨a, b, c, d⟩ := history_storage valid finite
  exact ⟨a, b, c, d, history_updated valid finite⟩

example : ([(-3 : Int), 2, -3].foldl History.update History.empty) = ⟨-1, -5⟩ := by decide

end NeuralReversi.MidgameAspiration

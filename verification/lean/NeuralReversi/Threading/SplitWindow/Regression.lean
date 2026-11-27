import NeuralReversi.Threading.SplitWindow

namespace NeuralReversi.Threading.SplitWindow.Regression

open PvsMoveLoop (Bound)

/-- A successful scout is only a lower bound. If a sibling raises alpha above
    that bound, comparing the scout with the new alpha can hide a better move.
    The retry decision must use the alpha with which the scout was searched;
    the new alpha can still be used as the lower end of the wide window.

    Scores here are in the parent's perspective. `selectReport` consumes the
    negated child returns, just like `search_split_point` in `search/pvs.rs`.
    This is an arbitrary-integer counterexample family, not a claim that a
    particular board or native execution realizes every such tuple. -/
def StaleAlphaRetryContract : Prop :=
  ∀ (alpha scout fresh value beta : Int),
    alpha < scout → scout ≤ fresh → fresh < value → value < beta →
    Bound value alpha (alpha + 1) scout ∧
    reSearch true alpha (-scout) = true ∧
    reSearch true fresh (-scout) = false ∧
    ¬ Bound value alpha beta (max fresh scout) ∧
    let report := selectReport true 0 alpha fresh (-scout) (-value)
    report.searchedAlpha = fresh ∧ report.score = value ∧
      Bound value alpha beta (max fresh report.score)

theorem staleAlphaRetryContract_proven : StaleAlphaRetryContract := by
  intro alpha scout fresh value beta improved hidden better below
  have retry : reSearch true alpha (-scout) = true := by
    simp [reSearch, improved]
  refine ⟨?_, retry, ?_, ?_, ?_⟩
  · constructor <;> omega
  · simp [reSearch]
    omega
  · intro bound
    have upper := bound.1 (by omega)
    omega
  · simp only [selectReport, retry, ite_true, Int.neg_neg]
    exact ⟨True.intro, True.intro, by constructor <;> omega⟩

/-- Concrete regression in Rust's scaled-score units: the scout proves at
    least two discs, a sibling has four discs, and this move really has six.
    Keeping the scout would publish four as an in-window exact result. -/
def ScaledWitness : Prop :=
  Bound 1536 0 1 512 ∧
    reSearch true 0 (-512) = true ∧
    reSearch true 1024 (-512) = false ∧
    ¬ Bound 1536 0 4096 (max 1024 512) ∧
    let report := selectReport true 0 0 1024 (-512) (-1536)
    report.searchedAlpha = 1024 ∧ report.score = 1536 ∧
      Bound 1536 0 4096 (max 1024 report.score)

theorem scaledWitness_proven : ScaledWitness :=
  staleAlphaRetryContract_proven 0 512 1024 1536 4096
    (by decide) (by decide) (by decide) (by decide)

end NeuralReversi.Threading.SplitWindow.Regression

import NeuralReversi.PvsParallel.SerialTable.Fallback
import NeuralReversi.PublicationMir.InterruptedExact

namespace NeuralReversi.PublicationMir.SerialFallback

open PvsParallel.SerialTable (Table)
open PvsBoardTree (Position minimax)
open InterruptedExact (SavedSafe)

/-- The no-snapshot branch reads the table left by the search. An exact, legal
    cache hit takes precedence over the ordinary one-ply fallback. -/
def fallback (t : Table l) (mc : Bitboard.Moves.X86Config) (fc : PvsParallel.Cutoffs.FlipConfig)
    (b : Position) (evaluate : Board.Board → Int) (nodes : Nat) : SearchResult :=
  match (t.image.lookup b).bind (ExactScore.cached mc b nodes) with
  | some report => .bestMove report
  | none =>
    match SearchFallback.quickMove mc fc b evaluate nodes with
    | some report => .bestMove report
    | none => .noLegalMove

/-- The executable fallback is exactly the no-snapshot public return branch;
    there is no caller-provided score or lookup result. -/
theorem fallback_returned (t : Table l) (mc fc) (b : Position) (evaluate : Board.Board → Int)
    (sort : SliceSort) (nodes : Nat) :
    ExactScore.returned mc fc b evaluate sort none (t.image.lookup b) nodes .search =
      some (fallback t mc fc b evaluate nodes) := by
  unfold ExactScore.returned intoResult fallback
  cases (t.image.lookup b).bind (ExactScore.cached mc b nodes) <;> rfl

/-- The physical table's existing invariant supplies the cache-history premise.
    A quick fallback never acquires the exact-endgame tag. -/
theorem fallback_ensures (t : Table l) (mc fc) (b : Position) (evaluate : Board.Board → Int)
    (sort : SliceSort) (nodes : Nat) (good : t.Good mc fc) :
    ExactScore.Ensures mc fc b (fallback t mc fc b evaluate nodes) := by
  exact PvsParallel.SerialTable.public_score _ t mc fc b evaluate sort none nodes .search _ good
    (fallback_returned t mc fc b evaluate sort nodes) (by simp)

/-- Return the already captured snapshot without sorting it again, or use the
    actual post-search table if no snapshot exists. Both sources satisfy the
    same public exact-score postcondition. -/
theorem intoResult_ensures (t : Table l) (mc fc) (b : Position) (evaluate : Board.Board → Int)
    (sort : SliceSort) (last : Option Iteration) (nodes : Nat) (result : SearchResult)
    (good : t.Good mc fc) (saved : SavedSafe (minimax mc fc b) (last.map asSnapshot))
    (run : intoResult last nodes (fun _ => fallback t mc fc b evaluate nodes) = some result) :
    ExactScore.Ensures mc fc b result := by
  cases found : last with
  | none =>
    simp only [found, intoResult] at run
    change some (fallback t mc fc b evaluate nodes) = some result at run
    rw [← Option.some.inj run]
    exact fallback_ensures t mc fc b evaluate sort nodes good
  | some iteration =>
    rw [found, intoResult_fromSnapshot] at run
    obtain ⟨report, built, rfl⟩ := Option.map_eq_some_iff.mp run
    have safe := saved (asSnapshot iteration) (by simp [found])
    intro value marked
    simp [ExactScore.scoreIfExact] at marked
    obtain ⟨⟨ending, level⟩, rfl⟩ := marked
    exact (InterruptedExact.fromSnapshot_exact safe built ending level).trans
      (EndgameTreeCache.scaled_discValue mc fc b).symm

end NeuralReversi.PublicationMir.SerialFallback

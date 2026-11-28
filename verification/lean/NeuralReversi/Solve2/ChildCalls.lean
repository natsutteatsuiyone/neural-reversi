module
public import NeuralReversi.Solve2.BoardBridge
public import NeuralReversi.Solve2.Control
public import NeuralReversi.Solve1.Bmi2Verified

@[expose] public section

namespace NeuralReversi.Solve2

def childOneEmpty (b : TwoEmpty) (flipped : Word) (subset : FlipSubset flipped b.opponent) :
    Solve1.Bmi2.Board where
  player := (childBoard b flipped).player
  square := b.second
  empty := child_remaining_empty b flipped subset

theorem childOneEmpty_exact_board (b : TwoEmpty) (flipped : Word)
    (subset : FlipSubset flipped b.opponent) :
    (childBoard b flipped).opponent =
      ~~~((childOneEmpty b flipped subset).player |||
        Bitboard.squareMask (childOneEmpty b flipped subset).square) :=
  child_opponent_reconstructed b flipped subset

def childSolve (b : TwoEmpty) (flipped : Word) (subset : FlipSubset flipped b.opponent)
    (alpha : Solve1.ScoreWord) : Solve1.ScoreWord :=
  Solve1.Bmi2.solve (childOneEmpty b flipped subset) alpha

def childValue (b : TwoEmpty) (flipped : Word) (subset : FlipSubset flipped b.opponent) : Int :=
  Solve1.Bmi2.exactScore (childOneEmpty b flipped subset)

theorem childSolve_oneEmpty (b : TwoEmpty) (flipped : Word)
    (subset : FlipSubset flipped b.opponent) (alpha : Solve1.ScoreWord) :
    OneEmpty alpha.toInt (childValue b flipped subset) (childSolve b flipped subset alpha).toInt := by
  have h := Solve1.Bmi2.solve_nws (childOneEmpty b flipped subset) alpha
  exact ⟨h.1, h.2.1⟩

theorem childSolve_result_bound (b : TwoEmpty) (flipped : Word)
    (subset : FlipSubset flipped b.opponent) (alpha : Solve1.ScoreWord) :
    -160 ≤ (childSolve b flipped subset alpha).toInt ∧
      (childSolve b flipped subset alpha).toInt ≤ 162 := by
  let cb := childOneEmpty b flipped subset
  have hp := Solve1.Bmi2.popCount_bound cb.player
  have hf := Solve1.Bmi2.reference_bound cb.player cb.square
  have ho := Solve1.Bmi2.reference_bound (~~~cb.player) cb.square
  have h := Solve1.word_result_bound (Solve1.Bmi2.popCount cb.player)
    (2 * Solve1.Bmi2.reference cb.player cb.square.val)
    (2 * Solve1.Bmi2.reference (~~~cb.player) cb.square.val) alpha hp (by omega) (by omega)
  rw [(Solve1.backend_word_equivalence _ _ _ _).1] at h
  change -160 ≤ (Solve1.Bmi2.solve cb alpha).toInt ∧
    (Solve1.Bmi2.solve cb alpha).toInt ≤ 162
  simpa only [Solve1.Bmi2.solve, Solve1.Bmi2.countWide_toNat,
    Solve1.Bmi2.countOppWide_toNat] using h

theorem generated_child_oneEmpty (b : TwoEmpty) (alpha : Solve1.ScoreWord) :
    let f := Flip.value ⟨b.first.val, by omega⟩ b.player b.opponent
    let h := scalar_flip_subset b.first b.player b.opponent
    OneEmpty alpha.toInt (childValue b f h) (childSolve b f h alpha).toInt :=
  childSolve_oneEmpty b _ (scalar_flip_subset b.first b.player b.opponent) alpha

end NeuralReversi.Solve2

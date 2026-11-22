import NeuralReversi.Solve1.NeonCounts
import NeuralReversi.Solve1.Bmi2Verified

namespace NeuralReversi.Solve1.Neon

abbrev Board := Bmi2.Board

def solve (b : Board) (alpha : ScoreWord) : ScoreWord :=
  neonWord (Bmi2.popCount b.player) (countLow b.square.val b.player).toNat
    (countHigh b.square.val b.player).toNat alpha

theorem solve_eq_bmi2 (b : Board) (alpha : ScoreWord) :
    solve b alpha = Bmi2.solve b alpha := by
  unfold solve Bmi2.solve
  rw [(counts_correct b.square b.player).1, (counts_correct b.square b.player).2,
    Bmi2.countWide_toNat, Bmi2.countOppWide_toNat,
    Bmi2.reference_eq_geometry, Bmi2.reference_eq_geometry]
  exact (backend_word_equivalence _ _ _ alpha).2

theorem solve_nws (b : Board) (alpha : ScoreWord) :
    NWSResult alpha.toInt (Bmi2.exactScore b) (solve b alpha).toInt := by
  rw [solve_eq_bmi2]
  exact Bmi2.solve_nws b alpha

end NeuralReversi.Solve1.Neon

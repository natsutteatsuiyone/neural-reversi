module
public import NeuralReversi.Solve2.Concrete

@[expose] public section

namespace NeuralReversi.Solve34
open Solve2
set_option linter.unusedSimpArgs false

theorem leaf_count_bounds (n nf no : Nat) (alpha : Int)
    (hn : n ≤ 64) (hf : nf ≤ 48) (ho : no ≤ 48) :
    -112 ≤ Solve1.solveCounts n nf no alpha ∧ Solve1.solveCounts n nf no alpha ≤ 114 := by
  by_cases hz : nf = 0 <;> by_cases hop : no > 0 <;>
    by_cases hb : Solve1.scoreBase n > 0 <;>
    by_cases ha : Solve1.scoreIfBothPass n > alpha <;>
    simp [Solve1.solveCounts, Solve1.scoreIfBothPass, Solve1.scoreBase, hz, hop] at * <;> omega

theorem child_bounds (b : TwoEmpty) (f : Word) (sub : FlipSubset f b.opponent) (alpha : Score) :
    -112 ≤ (childSolve b f sub alpha).toInt ∧ (childSolve b f sub alpha).toInt ≤ 114 := by
  let cb := childOneEmpty b f sub
  have hp := Solve1.Bmi2.popCount_bound cb.player
  have hf := Solve1.Bmi2.reference_bound cb.player cb.square
  have ho := Solve1.Bmi2.reference_bound (~~~cb.player) cb.square
  change -112 ≤ (Solve1.Bmi2.solve cb alpha).toInt ∧ (Solve1.Bmi2.solve cb alpha).toInt ≤ 114
  simp only [Solve1.Bmi2.solve, Solve1.Bmi2.countWide_toNat, Solve1.Bmi2.countOppWide_toNat]
  rw [Solve1.bmi2_bridge _ _ _ alpha hp (by omega) (by omega)]
  exact leaf_count_bounds _ _ _ _ hp (by omega) (by omega)

def BoundedOption (lo hi : Int) (o : Option Int) : Prop :=
  ∀ s, o = some s → lo ≤ s ∧ s ≤ hi

theorem candidate_bounds (b : TwoEmpty) (alpha : Score) :
    BoundedOption (-112) 114 ((candidateScore b alpha).map BitVec.toInt) := by
  intro s hs
  unfold candidateScore at hs
  split at hs
  · contradiction
  · cases hs
    exact child_bounds _ _ _ _

theorem option_neg_bounds {lo hi : Int} {o : Option Int} (h : BoundedOption lo hi o) :
    BoundedOption (-hi) (-lo) (o.map (- ·)) := by
  cases o with
  | none => intro s hs; contradiction
  | some v => intro s hs; cases hs; have hv := h v rfl; change -hi ≤ -v ∧ -v ≤ -lo; constructor <;> omega

theorem maxSearch_bounds (alpha lo hi : Int) (a b : Option Int) (fallback : Int)
    (ha : BoundedOption lo hi a) (hb : BoundedOption lo hi b)
    (hf : lo ≤ fallback ∧ fallback ≤ hi) :
    lo ≤ maxSearch alpha a b fallback ∧ maxSearch alpha a b fallback ≤ hi := by
  cases a <;> cases b <;> simp_all [BoundedOption, maxSearch, Int.max_def]
  all_goals split <;> simp_all <;> omega

theorem minSearch_bounds (alpha lo hi : Int) (a b : Option Int) (fallback : Int)
    (ha : BoundedOption lo hi a) (hb : BoundedOption lo hi b)
    (hf : lo ≤ fallback ∧ fallback ≤ hi) :
    lo ≤ minSearch alpha a b fallback ∧ minSearch alpha a b fallback ≤ hi := by
  cases a <;> cases b <;> simp_all [BoundedOption, minSearch, Int.min_def]
  all_goals split <;> simp_all <;> omega

theorem widen_option {lo hi lo' hi' : Int} {o : Option Int}
    (h : BoundedOption lo hi o) (l : lo' ≤ lo) (u : hi ≤ hi') : BoundedOption lo' hi' o := by
  intro s hs; have hh := h s hs; constructor <;> omega

theorem solve2_bounds (b : TwoEmpty) (alpha : Score) (safe : alpha.toInt < 2147483647) :
    -114 ≤ (solveConcrete b alpha).toInt ∧ (solveConcrete b alpha).toInt ≤ 114 := by
  unfold solveConcrete
  rw [solveWord_toInt _ _ _ _ _ _ safe (candidate_negSafe _ _) (candidate_negSafe _ _)]
  unfold solve
  apply maxSearch_bounds
  · exact widen_option (option_neg_bounds (candidate_bounds b _)) (by omega) (by omega)
  · exact widen_option (option_neg_bounds (candidate_bounds b.exchange _)) (by omega) (by omega)
  · apply minSearch_bounds
    · exact widen_option (candidate_bounds b.pass _) (by omega) (by omega)
    · exact widen_option (candidate_bounds b.pass.exchange _) (by omega) (by omega)
    · have h := terminal_bounds b; constructor <;> omega

theorem nws_allows_positive_sentinel : Solve2.NWS 127 0 127 := by
  constructor <;> intro h <;> omega

theorem nws_allows_negative_sentinel : Solve2.NWS (-128) 0 (-127) := by
  constructor <;> intro h <;> omega

end NeuralReversi.Solve34

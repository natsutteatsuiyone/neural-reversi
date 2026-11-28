module
public import NeuralReversi.Solve2.Concrete
public import NeuralReversi.Solve2.Guard

@[expose] public section

namespace NeuralReversi.Solve2

def verifiedOracle (c : Dispatch.Call) : Score :=
  if h : c.player.getLsbD c.square.val = false then
    Solve1.Bmi2.solve ⟨c.player,c.square,h⟩ c.alpha
  else 0

theorem candidate_map_verified (b : TwoEmpty) (alpha : Score) :
    (Dispatch.candidate b.opponent (generatedFlip b) alpha b.second).map verifiedOracle =
      candidateScore b alpha := by
  unfold Dispatch.candidate candidateScore
  split
  · rfl
  · have empty := child_remaining_empty b (generatedFlip b) (generatedFlip_subset b)
    change (b.opponent ^^^ generatedFlip b).getLsbD b.second.val = false at empty
    simp only [Option.map, verifiedOracle, empty, ↓reduceDIte]
    rfl

def sourceFallback (b : TwoEmpty) (alpha : Score) : Dispatch.Outcome :=
  Dispatch.fallback verifiedOracle Guard.scalarFlip Guard.adjacent
    b.player b.opponent alpha b.first b.second (terminalScore b)

theorem sourceFallback_score (b : TwoEmpty) (alpha : Score) :
    (sourceFallback b alpha).score = solveConcrete b alpha := by
  unfold sourceFallback
  rw [← Guard.scalar_eager_eq_fallback]
  unfold Dispatch.eager
  rw [Dispatch.run_score]
  change solveWord alpha
    ((Dispatch.candidate b.opponent (generatedFlip b) (childAlpha alpha) b.second).map verifiedOracle)
    ((Dispatch.candidate b.exchange.opponent (generatedFlip b.exchange) (childAlpha alpha) b.exchange.second).map verifiedOracle)
    ((Dispatch.candidate b.pass.opponent (generatedFlip b.pass) alpha b.pass.second).map verifiedOracle)
    ((Dispatch.candidate b.pass.exchange.opponent (generatedFlip b.pass.exchange) alpha b.pass.exchange.second).map verifiedOracle)
    (terminalScore b) = _
  simp only [candidate_map_verified, solveConcrete]

theorem sourceFallback_nws (b : TwoEmpty) (alpha : Score)
    (windowSafe : alpha.toInt < 2147483647) :
    NWS alpha.toInt (exactTwoValue b) (sourceFallback b alpha).score.toInt := by
  rw [sourceFallback_score]
  exact solveConcrete_nws b alpha windowSafe

end NeuralReversi.Solve2

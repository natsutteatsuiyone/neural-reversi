module
public import NeuralReversi.Solve2.ChildCalls
public import NeuralReversi.Solve2.Terminal

@[expose] public section

namespace NeuralReversi.Solve2

def generatedFlip (b : TwoEmpty) : Word :=
  Flip.value ⟨b.first.val, by omega⟩ b.player b.opponent

theorem generatedFlip_subset (b : TwoEmpty) : FlipSubset (generatedFlip b) b.opponent :=
  scalar_flip_subset b.first b.player b.opponent

def candidateScore (b : TwoEmpty) (alpha : Score) : Option Score :=
  if generatedFlip b = 0 then none
  else some (childSolve b (generatedFlip b) (generatedFlip_subset b) alpha)

def candidateValue (b : TwoEmpty) : Option Int :=
  if generatedFlip b = 0 then none
  else some (childValue b (generatedFlip b) (generatedFlip_subset b))

theorem candidate_contract (b : TwoEmpty) (alpha : Score) :
    OptionalRel (OneEmpty alpha.toInt) (candidateValue b)
      ((candidateScore b alpha).map BitVec.toInt) := by
  unfold candidateValue candidateScore
  split
  · trivial
  · exact childSolve_oneEmpty b _ _ alpha

theorem candidate_negSafe (b : TwoEmpty) (alpha : Score) : NegSafe (candidateScore b alpha) := by
  intro s hs
  unfold candidateScore at hs
  split at hs
  · contradiction
  · cases hs
    have bound := childSolve_result_bound b (generatedFlip b) (generatedFlip_subset b) alpha
    omega

def solveConcrete (b : TwoEmpty) (alpha : Score) : Score :=
  solveWord alpha
    (candidateScore b (childAlpha alpha))
    (candidateScore b.exchange (childAlpha alpha))
    (candidateScore b.pass alpha)
    (candidateScore b.pass.exchange alpha)
    (terminalScore b)

def exactTwoValue (b : TwoEmpty) : Int :=
  value (candidateValue b) (candidateValue b.exchange)
    (candidateValue b.pass) (candidateValue b.pass.exchange)
    (Board.AwardedScore b.asBoard)

theorem solveConcrete_nws (b : TwoEmpty) (alpha : Score)
    (windowSafe : alpha.toInt < 2147483647) :
    NWS alpha.toInt (exactTwoValue b) (solveConcrete b alpha).toInt := by
  have h := solveWord_nws windowSafe
    (candidate_negSafe b (childAlpha alpha))
    (candidate_negSafe b.exchange (childAlpha alpha))
    (candidate_contract b (childAlpha alpha))
    (candidate_contract b.exchange (childAlpha alpha))
    (candidate_contract b.pass alpha)
    (candidate_contract b.pass.exchange alpha)
    (terminal := terminalScore b)
  simpa only [terminal_awarded, exactTwoValue, solveConcrete] using h

theorem solveConcrete_classifies (b : TwoEmpty) (alpha : Score)
    (windowSafe : alpha.toInt < 2147483647) :
    ((solveConcrete b alpha).toInt ≤ alpha.toInt ↔ exactTwoValue b ≤ alpha.toInt) ∧
    (alpha.toInt < (solveConcrete b alpha).toInt ↔ alpha.toInt < exactTwoValue b) :=
  nws_classification (solveConcrete_nws b alpha windowSafe)

end NeuralReversi.Solve2

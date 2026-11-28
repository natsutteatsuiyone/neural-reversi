module
public import NeuralReversi.Binary64Arithmetic
public import NeuralReversi.EndgameProbCutModel

@[expose] public section

namespace NeuralReversi.MidgameProbCut

open EndgameProbCut (clamp)
open PvsMoveLoop (scoreInf)

def shallowDepth (depth : Nat) : Nat :=
  2 * (depth / 6)

def publicThreshold (threshold : Int) : Int :=
  clamp (-scoreInf) scoreInf threshold

def verificationBeta (threshold : Int) : Int :=
  max (-scoreInf + 1) (publicThreshold threshold)

structure Coefficients where
  mean : Int
  sigma : Int
  mean0 : Int
  sigma0 : Int

def thresholdsFromUnits (beta t : Int) (coefficients : Coefficients) (cutNode : Bool) :
    Option (Int × Int) := do
  let quantum : Int := EndgameTiming.quantum
  let raw := beta * quantum
  let half := quantum / 2
  let verificationDeviation ← Binary64Arithmetic.mul t coefficients.sigma
  let verificationOffset ← Binary64Arithmetic.add raw verificationDeviation
  let verification ← Binary64Arithmetic.sub verificationOffset coefficients.mean
  let halfMean0 ← Binary64Arithmetic.mul half coefficients.mean0
  let evaluationMean ← Binary64Arithmetic.add halfMean0 coefficients.mean
  let halfT ← Binary64Arithmetic.mul t half
  let evaluationDeviation0 ← Binary64Arithmetic.mul halfT coefficients.sigma0
  let evaluationDeviation ← Binary64Arithmetic.add evaluationDeviation0 coefficients.sigma
  let allMargin ← if cutNode then some 0 else Binary64Arithmetic.mul coefficients.sigma0 (3 * half)
  let preliminary ← Binary64Arithmetic.sub raw evaluationDeviation
  let corrected ← Binary64Arithmetic.sub preliminary evaluationMean
  let screen ← Binary64Arithmetic.add corrected allMargin
  return (EndgameProbCut.roundedUnits true verification, EndgameProbCut.roundedUnits false screen)

end NeuralReversi.MidgameProbCut

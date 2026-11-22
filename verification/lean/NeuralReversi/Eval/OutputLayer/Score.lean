import NeuralReversi.Eval.OutputLayer.Arithmetic
import NeuralReversi.MidgameScoreRange.Domain

namespace NeuralReversi.Eval.OutputLayer

def layerScore (raw : BitVec 32) : Int :=
  min (max (raw.sshiftRight 6).toInt (-32767)) 32767

def networkScore (raw : BitVec 32) : Int :=
  MidgameScoreRange.evalClamp (layerScore raw)

def ScaledScoreContract : Prop :=
  ∀ raw : BitVec 32,
    layerScore raw = min (max (raw.toInt / 64) (-32767)) 32767 ∧
    -32767 ≤ layerScore raw ∧ layerScore raw ≤ 32767

theorem scaledScoreContract_proven : ScaledScoreContract := by
  intro raw
  unfold layerScore
  rw [BitVec.toInt_sshiftRight, Int.shiftRight_eq_div_pow]
  change _ = _ ∧ -32767 ≤ min (max (raw.toInt / 64) (-32767)) 32767 ∧
    min (max (raw.toInt / 64) (-32767)) 32767 ≤ 32767
  exact ⟨rfl, by omega, by omega⟩

def NetworkScoreContract : Prop :=
  ∀ raw : BitVec 32,
    networkScore raw = MidgameScoreRange.evalClamp (raw.toInt / 64) ∧
    MidgameAspiration.FiniteScore (networkScore raw)

theorem networkScoreContract_proven : NetworkScoreContract := by
  intro raw
  unfold networkScore
  refine ⟨?_, MidgameScoreRange.eval_clamp_finite _⟩
  rw [(scaledScoreContract_proven raw).1]
  unfold MidgameScoreRange.evalClamp
  omega

end NeuralReversi.Eval.OutputLayer

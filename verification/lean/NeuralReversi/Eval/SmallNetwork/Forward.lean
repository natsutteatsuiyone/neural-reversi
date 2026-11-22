import NeuralReversi.Eval.SmallNetwork.Arithmetic
import NeuralReversi.Eval.InputLayer.Arithmetic
import NeuralReversi.MidgameScoreRange.Domain

namespace NeuralReversi.Eval.SmallNetwork

open LinearLayer (sumPrefix)

inductive Kernel where
  | scalar
  | avx2
  | avx512
  | neon

def hiddenWord (kernel : Kernel) (bias : Int) (selected : Nat → Int) : BitVec 16 :=
  let word := fun i => BitVec.ofInt 16 (selected i)
  match kernel with
  | .avx2 | .avx512 => InputLayer.addGroups (BitVec.ofInt 16 bias) word 8
  | .scalar | .neon => InputLayer.addRows (BitVec.ofInt 16 bias) word 32

def referenceHidden (bias : Int) (selected : Nat → Int) : Int :=
  (BitVec.ofInt 16 (InputLayer.sumRows bias selected 32)).toInt

def HiddenWordContract : Prop :=
  ∀ kernel bias selected, (hiddenWord kernel bias selected).toInt = referenceHidden bias selected

theorem hiddenWordContract_proven : HiddenWordContract := by
  intro kernel bias selected
  have add := InputLayer.wrappingRowsContract_proven bias selected 32
  have group := InputLayer.groupedAdditionContract_proven
    (BitVec.ofInt 16 bias) (fun i => BitVec.ofInt 16 (selected i)) 8
  have exactGroup := group.trans add
  cases kernel
  · exact congrArg BitVec.toInt add
  · exact congrArg BitVec.toInt exactGroup
  · exact congrArg BitVec.toInt exactGroup
  · exact congrArg BitVec.toInt add

def kernelActivation (kernel : Kernel) (n : Int) : Int :=
  match kernel with
  | .scalar => activation n
  | .avx2 | .avx512 => x86Activation n
  | .neon => neonActivation n

private theorem kernelActivation_eq (kernel : Kernel) (n : Int) :
    kernelActivation kernel n = activation n := by
  have same := activationContract_proven n
  cases kernel
  · rfl
  · exact same.1
  · exact same.1
  · exact same.2.1

def kernelDot (kernel : Kernel) (term : Nat → Int) : BitVec 32 :=
  match kernel with
  | .scalar => OutputLayer.wrapPrefix term 0 128
  | .avx2 => x86Dot false term
  | .avx512 => x86Dot true term
  | .neon => neonDot term

private theorem kernelDot_eq (kernel : Kernel) (term : Nat → Int) :
    kernelDot kernel term = OutputLayer.wrapPrefix term 0 128 := by
  cases kernel
  · rfl
  · exact (dotReductionContract_proven term false).1
  · exact (dotReductionContract_proven term true).1
  · exact (dotReductionContract_proven term false).2

def rawWord (kernel : Kernel) (hiddenBias : Nat → Int) (weight : Nat → Nat → Int)
    (selected : Nat → Nat) (outputWeight : Nat → Int) (outputBias : Int) : BitVec 32 :=
  let term := fun column =>
    let hidden := hiddenWord kernel (hiddenBias column) (fun i => weight (selected i) column)
    kernelActivation kernel hidden.toInt * outputWeight column
  kernelDot kernel term + BitVec.ofInt 32 outputBias

def referenceWord (hiddenBias : Nat → Int) (weight : Nat → Nat → Int)
    (selected : Nat → Nat) (outputWeight : Nat → Int) (outputBias : Int) : BitVec 32 :=
  BitVec.ofInt 32 (outputBias + sumPrefix (fun column =>
    activation (referenceHidden (hiddenBias column) (fun i => weight (selected i) column)) *
      outputWeight column) 128)

def ForwardContract : Prop :=
  ∀ kernel hiddenBias weight selected outputWeight outputBias,
    rawWord kernel hiddenBias weight selected outputWeight outputBias =
      referenceWord hiddenBias weight selected outputWeight outputBias

theorem forwardContract_proven : ForwardContract := by
  intro kernel hiddenBias weight selected outputWeight outputBias
  unfold rawWord referenceWord
  dsimp only
  rw [kernelDot_eq]
  have each (bias : Int) (selected : Nat → Int) := hiddenWordContract_proven kernel bias selected
  simp only [each, kernelActivation_eq]
  have wrapped := OutputLayer.wrappingDotContract_proven
    (fun column => activation (referenceHidden (hiddenBias column)
      (fun i => weight (selected i) column)) * outputWeight column) 0 128
  change OutputLayer.wrapPrefix _ 0 128 = _ at wrapped
  rw [wrapped, Int.zero_add, BitVec.add_comm, ← BitVec.ofInt_add]

def score (raw : BitVec 32) : Int :=
  MidgameScoreRange.evalClamp (min (max (raw.sshiftRight 8).toInt (-32767)) 32767)

def ScoreContract : Prop :=
  ∀ raw,
    score raw = MidgameScoreRange.evalClamp (raw.toInt / 256) ∧
    MidgameAspiration.FiniteScore (score raw)

theorem scoreContract_proven : ScoreContract := by
  intro raw
  unfold score
  refine ⟨?_, MidgameScoreRange.eval_clamp_finite _⟩
  rw [BitVec.toInt_sshiftRight, Int.shiftRight_eq_div_pow]
  change MidgameScoreRange.evalClamp (min (max (raw.toInt / 256) (-32767)) 32767) =
    MidgameScoreRange.evalClamp (raw.toInt / 256)
  unfold MidgameScoreRange.evalClamp
  omega

def BackendScoreContract : Prop :=
  ∀ kernel hiddenBias weight selected outputWeight outputBias,
    let actual := score (rawWord kernel hiddenBias weight selected outputWeight outputBias)
    actual = MidgameScoreRange.evalClamp
      ((referenceWord hiddenBias weight selected outputWeight outputBias).toInt / 256) ∧
    MidgameAspiration.FiniteScore actual

theorem backendScoreContract_proven : BackendScoreContract := by
  intro kernel hiddenBias weight selected outputWeight outputBias
  dsimp only
  rw [forwardContract_proven kernel hiddenBias weight selected outputWeight outputBias]
  exact scoreContract_proven _

end NeuralReversi.Eval.SmallNetwork

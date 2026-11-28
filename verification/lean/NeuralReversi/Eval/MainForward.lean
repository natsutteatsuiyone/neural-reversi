module
public import NeuralReversi.Eval.Activations
public import NeuralReversi.Eval.LinearLayer.Forward
public import NeuralReversi.Eval.MainScratch
public import NeuralReversi.Eval.OutputLayer.Score

@[expose] public section

namespace NeuralReversi.Eval.MainForward

open MainScratch

structure DenseWeights where
  bias : Nat → Int
  disk : Nat → BitVec 8

def dense (kernel : LinearLayer.DotKernel) (inputs chunks outputs row : Nat)
    (weights : DenseWeights) (input : Nat → Int) : Int32 :=
  Int32.ofBitVec (LinearLayer.simdForward kernel inputs chunks outputs chunks row
    (weights.bias row) input (fun i => (weights.disk i).toInt))

def rowMajor (inputs chunks row : Nat) (weights : DenseWeights) (input : Nat → Int) : Int32 :=
  Int32.ofBitVec (LinearLayer.scalarForward inputs chunks row (weights.bias row)
    input (fun i => (weights.disk i).toInt))

private theorem dense_eq_rowMajor (kernel inputs chunks outputs row weights input)
    (columnBounds : inputs ≤ 4 * chunks) (rowBounds : row < outputs)
    (inputBounds : ∀ i, i < 4 * chunks → 0 ≤ input i ∧ input i ≤ 255) :
    dense kernel inputs chunks outputs row weights input = rowMajor inputs chunks row weights input := by
  unfold dense rowMajor
  rw [LinearLayer.denseForwardContract_proven _ _ _ _ _ _ _ _ _ rowBounds
    (Nat.le_refl _) columnBounds inputBounds]
  intro i _
  have low := BitVec.le_toInt (weights.disk i)
  have high := BitVec.toInt_lt (x := weights.disk i)
  omega

structure StackWeights where
  l1 : DenseWeights
  l2 : DenseWeights
  outputBias : BitVec 32
  outputWeights : Nat → BitVec 32

def padded (input : Inputs) (i : Nat) : Int :=
  if h : i < 288 then (input ⟨i, h⟩).toNat else 0

private theorem paddedBounds (input : Inputs) (i : Nat) : 0 ≤ padded input i ∧ padded input i ≤ 255 := by
  unfold padded
  split
  · have h := (input ⟨i, ‹_›⟩).isLt
    omega
  · decide

def fusedSpec (hidden : Fin 16 → Int32) (i : Fin 32) : Int :=
  let x := (hidden ⟨i.val % 16, Nat.mod_lt _ (by decide)⟩).toInt
  if i.val < 16 then Activations.squareSpec x else Activations.reluSpec x

private theorem fusedSpecBounds (hidden : Fin 16 → Int32) (i : Fin 32) :
    0 ≤ fusedSpec hidden i ∧ fusedSpec hidden i ≤ 255 := by
  unfold fusedSpec
  split
  · unfold Activations.squareSpec
    change 0 ≤ min (max 0 _) 255 ∧ min (max 0 _) 255 ≤ 255
    omega
  · unfold Activations.reluSpec
    change 0 ≤ min (max 0 _) 255 ∧ min (max 0 _) 255 ≤ 255
    omega

def l2Inputs (activation : Fin 32 → Int) (i : Nat) : Int :=
  if h : i < 32 then activation ⟨i, h⟩ else 0

def l2Spec (weights : StackWeights) (input : Inputs) (i : Nat) : Int32 :=
  rowMajor 32 8 i weights.l2
    (l2Inputs (fusedSpec (fun j => rowMajor 257 72 j.val weights.l1 (padded input))))

def l2Dispatched (weights : StackWeights) (input : Inputs)
    (first second : LinearLayer.DotKernel) (activation : Activations.Backend) (i : Nat) : Int32 :=
  dense second 32 8 64 i weights.l2
    (l2Inputs (Activations.fusedOutput activation
      (fun j => dense first 257 72 16 j.val weights.l1 (padded input))))

def DenseStackContract : Prop :=
  ∀ weights input first second activation i, i < 64 →
    l2Dispatched weights input first second activation i = l2Spec weights input i

theorem denseStackContract_proven : DenseStackContract := by
  intro weights input first second activation i inBounds
  have l1 : (fun j : Fin 16 => dense first 257 72 16 j.val weights.l1 (padded input)) =
      fun j : Fin 16 => rowMajor 257 72 j.val weights.l1 (padded input) := by
    funext j
    exact dense_eq_rowMajor _ _ _ _ _ _ _ (by decide) j.isLt (fun k _ => paddedBounds input k)
  have activations (hidden : Fin 16 → Int32) :
      Activations.fusedOutput activation hidden = fusedSpec hidden := by
    funext j
    exact Activations.fusedContract_proven activation hidden j
  unfold l2Dispatched l2Spec
  rw [l1, activations]
  apply dense_eq_rowMajor _ _ _ _ _ _ _ (by decide) inBounds
  intro j hj
  have valid : j < 32 := hj
  simpa only [l2Inputs, dite_eq_left valid] using fusedSpecBounds
    (fun k => rowMajor 257 72 k.val weights.l1 (padded input)) ⟨j, valid⟩

def bypassDot (weights : StackWeights) (input : Inputs) : BitVec 32 :=
  ((List.range 256).map (fun i =>
    BitVec.ofInt 32 (padded input i) * weights.outputWeights (64 + i))).sum

def canonicalRaw (weights : StackWeights) (input : Inputs) : BitVec 32 :=
  weights.outputBias + ((List.range 64).map (fun i =>
    BitVec.ofInt 32 (Activations.scReluSpec (l2Spec weights input i).toInt) *
      weights.outputWeights i)).sum + bypassDot weights input

def dispatchedRaw (weights : StackWeights) (input : Inputs)
    (first second : LinearLayer.DotKernel) (activation : Activations.Backend) : BitVec 32 :=
  let hidden := l2Dispatched weights input first second activation
  weights.outputBias + ((List.range 64).map (fun i =>
    BitVec.ofInt 32 (Activations.scReluOutput activation 64 hidden i) *
      weights.outputWeights (Activations.scReluStoreOrder activation 64 i))).sum +
    bypassDot weights input

def RawStackContract : Prop :=
  ∀ weights input first second activation,
    dispatchedRaw weights input first second activation = canonicalRaw weights input

theorem rawStackContract_proven : RawStackContract := by
  intro weights input first second activation
  dsimp only [dispatchedRaw, canonicalRaw]
  rw [Activations.leadingDotProductContract_proven]
  congr 3
  apply List.map_congr_left
  intro i member
  rw [denseStackContract_proven _ _ _ _ _ _ (List.mem_range.mp member)]

def MainScoreContract : Prop :=
  ∀ weights first second activation history old base phase mobility,
    PaddingZero old →
    let raw := dispatchedRaw weights (prepare (prepareMany history old) base phase mobility)
      first second activation
    let initial := prepare (fun _ => 0) base phase mobility
    OutputLayer.networkScore raw = MidgameScoreRange.evalClamp ((canonicalRaw weights initial).toInt / 64) ∧
      MidgameAspiration.FiniteScore (OutputLayer.networkScore raw)

theorem mainScoreContract_proven : MainScoreContract := by
  intro weights first second activation history old base phase mobility zero
  dsimp
  rw [(reuseContract_proven history old zero).2 base phase mobility, rawStackContract_proven]
  exact OutputLayer.networkScoreContract_proven _

end NeuralReversi.Eval.MainForward

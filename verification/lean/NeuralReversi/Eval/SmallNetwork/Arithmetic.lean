import NeuralReversi.Eval.OutputLayer.Arithmetic

namespace NeuralReversi.Eval.SmallNetwork

open LinearLayer (sumPrefix)

def clip1023 (n : Int) : Int := min (max n 0) 1023

def activation (n : Int) : Int := clip1023 n * clip1023 n / 1024

def x86Activation (n : Int) : Int :=
  let clipped := clip1023 n
  ((64 * clipped) % 65536) * clipped / 65536

def neonActivation (n : Int) : Int :=
  let clipped := min n 1023
  let shifted := min (max (32 * clipped) 0) 65535
  let doubledHigh := 2 * (shifted * clipped) / 65536
  min (max doubledHigh (-32768)) 32767

private theorem clipRange (x : Int) : 0 ≤ clip1023 x ∧ clip1023 x ≤ 1023 := by
  unfold clip1023
  omega

theorem activationRange (n : Int) : 0 ≤ activation n ∧ activation n ≤ 1022 := by
  obtain ⟨lo, hi⟩ := clipRange n
  have nonneg := Int.mul_nonneg lo lo
  have first := Int.mul_le_mul_of_nonneg_left hi lo
  have second := Int.mul_le_mul_of_nonneg_right hi (by decide : (0 : Int) ≤ 1023)
  rw [Int.mul_comm _ 1023] at first
  unfold activation
  omega

def ActivationContract : Prop :=
  ∀ n : Int,
    x86Activation n = activation n ∧ neonActivation n = activation n ∧
    0 ≤ activation n ∧ activation n ≤ 1022 ∧
    64 * clip1023 n < 65536 ∧ 32 * clip1023 n ≤ 32767

theorem activationContract_proven : ActivationContract := by
  intro n
  have bounds := clipRange n
  have aRange := activationRange n
  refine ⟨?_, ?_, aRange.1, aRange.2, by omega, by omega⟩
  · unfold x86Activation activation
    have rem : (64 * clip1023 n) % 65536 = 64 * clip1023 n :=
      Int.emod_eq_of_lt (by omega) (by omega)
    dsimp only
    rw [rem, Int.mul_assoc]
    omega
  · have shift : min (max (32 * min n 1023) 0) 65535 = 32 * clip1023 n := by
      unfold clip1023
      omega
    have mul : clip1023 n * min n 1023 = clip1023 n * clip1023 n := by
      by_cases sign : n ≥ 0
      · rw [show min n 1023 = clip1023 n by unfold clip1023; omega]
      · have z : clip1023 n = 0 := by unfold clip1023; omega
        rw [z]
        simp
    have quot : 64 * (clip1023 n * clip1023 n) / 65536 = activation n := by
      unfold activation
      omega
    unfold neonActivation
    dsimp only
    rw [shift, Int.mul_assoc 32, ← Int.mul_assoc 2, show (2 : Int) * 32 = 64 by decide,
      mul, quot]
    omega

def ActivationMonotoneContract : Prop :=
  ∀ lo hi : Int, lo ≤ hi → activation lo ≤ activation hi

theorem activationMonotoneContract_proven : ActivationMonotoneContract := by
  intro lo hi order
  have h : clip1023 lo ≤ clip1023 hi := by unfold clip1023; omega
  have first := Int.mul_le_mul_of_nonneg_left h (clipRange lo).1
  have second := Int.mul_le_mul_of_nonneg_right h (clipRange hi).1
  unfold activation
  omega

private theorem dotProductBounds {x weight : Int}
    (xOk : 0 ≤ x ∧ x ≤ 1022) (wOk : -32768 ≤ weight ∧ weight ≤ 32767) :
    -33488896 ≤ x * weight ∧ x * weight ≤ 33487874 := by
  have lo := Int.mul_le_mul_of_nonneg_left wOk.1 xOk.1
  have hi := Int.mul_le_mul_of_nonneg_left wOk.2 xOk.1
  rw [Int.mul_comm _ (-32768)] at lo
  rw [Int.mul_comm _ 32767] at hi
  omega

def DotBoundsContract : Prop :=
  ∀ (hidden weights : Nat → Int) count,
    (∀ i, i < count → -32768 ≤ weights i ∧ weights i ≤ 32767) →
    let dot := sumPrefix (fun i => activation (hidden i) * weights i) count
    (-33488896 * (count : Int) ≤ dot ∧ dot ≤ 33487874 * (count : Int) ∧
    (count ≤ 64 → -2147483648 ≤ dot ∧ dot ≤ 2147483647))

theorem dotBoundsContract_proven : DotBoundsContract := by
  intro hidden weights count valid
  have sums (n : Nat) (h : n ≤ count) :
      -33488896 * (n : Int) ≤ sumPrefix (fun i => activation (hidden i) * weights i) n ∧
      sumPrefix (fun i => activation (hidden i) * weights i) n ≤ 33487874 * (n : Int) := by
    induction n with
    | zero => simp [sumPrefix]
    | succ n ih =>
      have p := dotProductBounds (activationRange (hidden n)) (valid n (by omega))
      have prefixBounds := ih (by omega)
      rw [sumPrefix]
      exact ⟨by omega, by omega⟩
  have h := sums count (by omega)
  exact ⟨h.1, h.2, fun _ => ⟨by omega, by omega⟩⟩

private theorem sumPrefix_split (f : Nat → Int) (first second : Nat) :
    sumPrefix f (first + second) = sumPrefix f first + sumPrefix (fun i => f (first + i)) second := by
  induction second with
  | zero => simp [sumPrefix]
  | succ n ih =>
    rw [Nat.add_succ, sumPrefix, sumPrefix, ih, Int.add_assoc]

def x86Dot (wide : Bool) (term : Nat → Int) : BitVec 32 :=
  if wide then BitVec.ofInt 32 (OutputLayer.laneSum term 16 2 4)
  else BitVec.ofInt 32 (OutputLayer.laneSum term 8 2 8)

def neonDot (term : Nat → Int) : BitVec 32 :=
  BitVec.ofInt 32 (OutputLayer.laneSum term 4 1 16) +
    BitVec.ofInt 32 (OutputLayer.laneSum (fun i => term (64 + i)) 4 1 16)

def DotReductionContract : Prop :=
  ∀ term wide,
    x86Dot wide term = OutputLayer.wrapPrefix term 0 128 ∧
    neonDot term = OutputLayer.wrapPrefix term 0 128

theorem dotReductionContract_proven : DotReductionContract := by
  intro term wide
  have scalarEq := OutputLayer.wrappingDotContract_proven term 0 128
  change OutputLayer.wrapPrefix term 0 128 = _ at scalarEq
  rw [scalarEq, Int.zero_add]
  refine ⟨?_, ?_⟩
  · cases wide
    · exact congrArg (BitVec.ofInt 32) (OutputLayer.laneReductionContract_proven term 8 2 8)
    · exact congrArg (BitVec.ofInt 32) (OutputLayer.laneReductionContract_proven term 16 2 4)
  · unfold neonDot
    rw [OutputLayer.laneReductionContract_proven term 4 1 16,
      OutputLayer.laneReductionContract_proven (fun i => term (64 + i)) 4 1 16,
      ← BitVec.ofInt_add]
    exact congrArg (BitVec.ofInt 32) (sumPrefix_split term 64 64).symm

end NeuralReversi.Eval.SmallNetwork

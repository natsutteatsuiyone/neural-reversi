module
public import NeuralReversi.Eval.LinearLayer.Forward
public import NeuralReversi.Eval.OutputLayer.Layout

@[expose] public section

namespace NeuralReversi.Eval.OutputLayer

open LinearLayer (sumPrefix)

def lowByte (w : Int) : Int := w % 256

def highByte (w : Int) : Int := (w / 256) % 256

def signedByte (b : Int) : Int := if b < 128 then b else b - 256

def WeightBytesContract : Prop :=
  ∀ w : Int, -32768 ≤ w → w ≤ 32767 →
    (0 ≤ lowByte w ∧ lowByte w ≤ 255) ∧
    (-128 ≤ signedByte (highByte w) ∧ signedByte (highByte w) ≤ 127) ∧
    lowByte w + 256 * signedByte (highByte w) = w

theorem weightBytesContract_proven : WeightBytesContract := by
  intro w lo hi
  unfold lowByte highByte signedByte
  split <;> omega

private theorem productBounds (a w : Int) (ha : 0 ≤ a ∧ a ≤ 255)
    (hw : -32768 ≤ w ∧ w ≤ 32767) :
    -8355840 ≤ a * w ∧ a * w ≤ 8355585 := by
  have p := Int.mul_le_mul_of_nonneg_left hw.1 ha.1
  have q := Int.mul_le_mul_of_nonneg_left hw.2 ha.1
  omega

def WidePairContract : Prop :=
  ∀ a b w x : Int,
    (0 ≤ a ∧ a ≤ 255) → (0 ≤ b ∧ b ≤ 255) →
    (-32768 ≤ w ∧ w ≤ 32767) → (-32768 ≤ x ∧ x ≤ 32767) →
    -16711680 ≤ a * w + b * x ∧ a * w + b * x ≤ 16711170 ∧
      (BitVec.ofInt 32 (a * w + b * x)).toInt = a * w + b * x

theorem widePairContract_proven : WidePairContract := by
  intro a b w x ha hb hw hx
  have p := productBounds a w ha hw
  have q := productBounds b x hb hx
  refine ⟨by omega, by omega, ?_⟩
  apply BitVec.toInt_ofInt_eq_self (by decide)
  all_goals simp <;> omega

private theorem sumPrefix_add (f g : Nat → Int) (n : Nat) :
    sumPrefix (fun i => f i + g i) n = sumPrefix f n + sumPrefix g n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [sumPrefix, sumPrefix, sumPrefix, ih]
    omega

private theorem sumPrefix_scale (f : Nat → Int) (k : Int) (n : Nat) :
    sumPrefix (fun i => k * f i) n = k * sumPrefix f n := by
  induction n with
  | zero => simp [sumPrefix]
  | succ n ih => rw [sumPrefix, sumPrefix, ih, Int.mul_add]

private theorem sumPrefix_congr (f g : Nat → Int) (n : Nat)
    (h : ∀ i, i < n → f i = g i) : sumPrefix f n = sumPrefix g n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [sumPrefix, sumPrefix, h n (by omega), ih (fun i hi => h i (by omega))]

def I8mmRecombineContract : Prop :=
  ∀ (input weight : Nat → Int) n,
    (∀ i, i < n → -32768 ≤ weight i ∧ weight i ≤ 32767) →
    BitVec.ofInt 32 (sumPrefix (fun i => input i * lowByte (weight i)) n) +
      (BitVec.ofInt 32 (sumPrefix (fun i => input i * signedByte (highByte (weight i))) n) <<< 8) =
      BitVec.ofInt 32 (sumPrefix (fun i => input i * weight i) n)

theorem i8mmRecombineContract_proven : I8mmRecombineContract := by
  intro input weight n valid
  let lo (i : Nat) := input i * lowByte (weight i)
  let hi (i : Nat) := input i * signedByte (highByte (weight i))
  change BitVec.ofInt 32 (sumPrefix lo n) + (BitVec.ofInt 32 (sumPrefix hi n) <<< 8) = _
  rw [BitVec.shiftLeft_eq_mul_twoPow]
  change BitVec.ofInt 32 (sumPrefix lo n) +
    BitVec.ofInt 32 (sumPrefix hi n) * BitVec.ofInt 32 256 = _
  rw [BitVec.mul_comm, ← BitVec.ofInt_mul, ← BitVec.ofInt_add, ← sumPrefix_scale, ← sumPrefix_add]
  congr 1
  apply sumPrefix_congr
  intro i ih
  obtain ⟨lower, upper⟩ := valid i ih
  have code := (weightBytesContract_proven (weight i) lower upper).2.2
  change input i * lowByte (weight i) + 256 * (input i * signedByte (highByte (weight i))) = _
  rw [Int.mul_left_comm 256, ← Int.mul_add, code]

def wrapPrefix (term : Nat → Int) (bias : BitVec 32) : Nat → BitVec 32
  | 0 => bias
  | n + 1 => wrapPrefix term bias n + BitVec.ofInt 32 (term n)

def WrappingDotContract : Prop :=
  ∀ (term : Nat → Int) bias n,
    wrapPrefix term (BitVec.ofInt 32 bias) n = BitVec.ofInt 32 (bias + sumPrefix term n)

theorem wrappingDotContract_proven : WrappingDotContract := by
  intro term bias n
  induction n with
  | zero => simp [wrapPrefix, sumPrefix]
  | succ n ih => rw [wrapPrefix, ih, ← BitVec.ofInt_add, sumPrefix, Int.add_assoc]

private theorem sumPrefix_split (f : Nat → Int) (offset count : Nat) :
    sumPrefix f (offset + count) = sumPrefix f offset + sumPrefix (fun i => f (offset + i)) count := by
  induction count with
  | zero => simp [sumPrefix]
  | succ n ih =>
    rw [Nat.add_succ, sumPrefix, sumPrefix, ih]
    omega

private theorem sumPrefix_groups (f : Nat → Int) (width count : Nat) :
    sumPrefix (fun chunk => sumPrefix (fun lane => f (chunk * width + lane)) width) count =
      sumPrefix f (count * width) := by
  induction count with
  | zero => simp [sumPrefix]
  | succ n ih => rw [sumPrefix, ih, Nat.succ_mul, sumPrefix_split]

private theorem sumPrefix_zero (n : Nat) : sumPrefix (fun _ => 0) n = 0 := by
  induction n with
  | zero => rfl
  | succ n ih => rw [sumPrefix, ih]; rfl

private theorem sumPrefix_transpose (f : Nat → Nat → Int) (rows cols : Nat) :
    sumPrefix (fun j => sumPrefix (fun i => f i j) rows) cols =
      sumPrefix (fun i => sumPrefix (f i) cols) rows := by
  induction rows with
  | zero => exact sumPrefix_zero cols
  | succ r ih =>
    have e : (fun j => sumPrefix (fun i => f i j) (r + 1)) =
        fun j => sumPrefix (fun i => f i j) r + f r j := rfl
    rw [e, sumPrefix_add, ih, sumPrefix]

def laneSum (term : Nat → Int) (lanes group chunks : Nat) : Int :=
  sumPrefix (fun lane => sumPrefix (fun chunk =>
    sumPrefix (fun j => term (chunk * (lanes * group) + (lane * group + j))) group) chunks) lanes

def LaneReductionContract : Prop :=
  ∀ term lanes group chunks,
    laneSum term lanes group chunks = sumPrefix term (chunks * (lanes * group))

theorem laneReductionContract_proven : LaneReductionContract := by
  intro term lanes group chunks
  unfold laneSum
  rw [sumPrefix_transpose]
  calc
    _ = sumPrefix (fun i => sumPrefix (fun j => term (i * (lanes * group) + j)) (lanes * group))
          chunks := by
      apply sumPrefix_congr
      intro i _
      exact sumPrefix_groups (fun j => term (i * (lanes * group) + j)) group lanes
    _ = _ := sumPrefix_groups term (lanes * group) chunks

def segmentTerm (first second weights : Nat → Int) (len₀ i : Nat) : Int :=
  (if i < len₀ then first i else second (i - len₀)) * weights i

def SegmentedForwardContract : Prop :=
  ∀ first second weights len₀ len₁ bias lanes group chunks₀ chunks₁,
    len₀ = chunks₀ * (lanes * group) → len₁ = chunks₁ * (lanes * group) →
    BitVec.ofInt 32 (bias +
      laneSum (fun i => first i * weights i) lanes group chunks₀ +
      laneSum (fun i => second i * weights (len₀ + i)) lanes group chunks₁) =
      wrapPrefix (segmentTerm first second weights len₀) (BitVec.ofInt 32 bias) (len₀ + len₁)

theorem segmentedForwardContract_proven : SegmentedForwardContract := by
  intro first second weights len₀ len₁ bias lanes group chunks₀ chunks₁ fits₀ fits₁
  rw [laneReductionContract_proven, laneReductionContract_proven, ← fits₀, ← fits₁,
    wrappingDotContract_proven, sumPrefix_split]
  congr 1
  have firstSum : sumPrefix (segmentTerm first second weights len₀) len₀ =
      sumPrefix (fun i => first i * weights i) len₀ := by
    apply sumPrefix_congr
    intro i hi
    simp [segmentTerm, hi]
  have secondSum : sumPrefix (fun i => segmentTerm first second weights len₀ (len₀ + i)) len₁ =
      sumPrefix (fun i => second i * weights (len₀ + i)) len₁ := by
    apply sumPrefix_congr
    intro i _
    have h : ¬ len₀ + i < len₀ := by omega
    simp [segmentTerm, h]
  rw [firstSum, secondSum, Int.add_assoc]

end NeuralReversi.Eval.OutputLayer

import Lean

namespace NeuralReversi.Eval.InputLayer

def patternSquares (i : Nat) : Nat :=
  if i < 20 then 8 else if i < 28 then 9 else 7

def patternSize (i : Nat) : Nat := 3 ^ patternSquares i

def patternStart (i : Nat) : Nat :=
  min i 20 * 6561 + min (i - 20) 8 * 19683 + (i - 28) * 2187

def totalRows : Nat := 297432

def checkedRow (i value : Nat) : Option Nat :=
  if i < 32 ∧ value < patternSize i then some (patternStart i + value) else none

def PatternOffsetsContract : Prop :=
  patternStart 0 = 0 ∧ patternStart 32 = totalRows ∧
  (∀ i, i < 32 → patternStart (i + 1) = patternStart i + patternSize i) ∧
  (∀ i j, i ≤ j → patternStart i ≤ patternStart j)

theorem patternOffsetsContract_proven : PatternOffsetsContract := by
  refine ⟨rfl, rfl, ?_, ?_⟩
  · intro i _
    unfold patternSize patternSquares patternStart
    split <;> (try split) <;> omega
  · intro i j h
    unfold patternStart
    omega

def CheckedRowsContract : Prop :=
  (∀ i value, i ≥ 32 ∨ value ≥ patternSize i → checkedRow i value = none) ∧
  (∀ i value, i < 32 → value < patternSize i →
    checkedRow i value = some (patternStart i + value) ∧
    patternStart i ≤ patternStart i + value ∧ patternStart i + value < totalRows) ∧
  (∀ i j value other, i < j → j < 32 → value < patternSize i →
    patternStart i + value < patternStart j + other)

theorem checkedRowsContract_proven : CheckedRowsContract := by
  obtain ⟨_, total, step, mono⟩ := patternOffsetsContract_proven
  refine ⟨?_, ?_, ?_⟩
  · intro i value bad
    unfold checkedRow
    exact ite_eq_right (by omega)
  · intro i value hi hv
    have next := step i hi
    have limit := mono (i + 1) 32 (by omega)
    exact ⟨ite_eq_left ⟨hi, hv⟩, by omega, by omega⟩
  · intro i j value other order jOk valueOk
    have next := step i (by omega)
    have bound := mono (i + 1) j (by omega)
    omega

def validatePrefix (feature : Nat → Nat) : Nat → Bool
  | 0 => true
  | n + 1 => validatePrefix feature n && decide (feature n < patternSize n)

def ValidateFeatureContract : Prop :=
  ∀ feature,
    (validatePrefix feature 32 = true ↔ ∀ i, i < 32 → feature i < patternSize i) ∧
    (validatePrefix feature 32 = true →
      ∀ i, i < 32 → patternStart i + feature i < totalRows)

theorem validateFeatureContract_proven : ValidateFeatureContract := by
  intro feature
  have valid (n : Nat) : validatePrefix feature n = true ↔ ∀ i, i < n → feature i < patternSize i := by
    induction n with
    | zero => simp [validatePrefix]
    | succ n ih =>
      simp only [validatePrefix, Bool.and_eq_true, decide_eq_true_eq, ih]
      constructor
      · rintro ⟨earlier, last⟩ i hi
        by_cases h : i = n
        · simpa [h] using last
        · exact earlier i (by omega)
      · intro all
        exact ⟨fun i hi => all i (by omega), all n (by omega)⟩
  refine ⟨valid 32, fun ok i hi => ?_⟩
  exact (checkedRowsContract_proven.2.1 i (feature i) hi ((valid 32).mp ok i hi)).2.2

def ternaryValue : List Nat → Nat
  | [] => 0
  | d :: rest => d + 3 * ternaryValue rest

private theorem ternaryBound (digits : List Nat) (valid : ∀ d ∈ digits, d < 3) :
    ternaryValue digits < 3 ^ digits.length := by
  induction digits with
  | nil => decide
  | cons d rest ih =>
    have hd := valid d (by simp)
    have tail := ih (fun x hx => valid x (by simp [hx]))
    simp only [ternaryValue, List.length_cons, Nat.pow_succ]
    rw [Nat.mul_comm]
    omega

def FeatureValueContract : Prop :=
  ∀ i digits, i < 32 → digits.length = patternSquares i →
    (∀ d ∈ digits, d < 3) →
    ternaryValue digits < patternSize i ∧ ternaryValue digits < 65536

theorem featureValueContract_proven : FeatureValueContract := by
  intro i digits _ len valid
  have bound := ternaryBound digits valid
  rw [len] at bound
  refine ⟨bound, ?_⟩
  unfold patternSquares at bound
  split at bound <;> (try split at bound) <;> omega

def RowLoadsContract : Prop :=
  ∀ (rows row regs reg width : Nat), row < rows → reg < regs → 0 < width →
    let words := regs * width
    let start := width * (row * regs + reg)
    start = row * words + width * reg ∧
    start + width ≤ rows * words ∧ (2 * width) ∣ 2 * start

theorem rowLoadsContract_proven : RowLoadsContract := by
  intro rows row regs reg width rowOk regOk _
  have next : row * regs + reg + 1 ≤ (row + 1) * regs := by
    rw [Nat.add_mul, Nat.one_mul]
    omega
  have within := Nat.le_trans next (Nat.mul_le_mul_right regs (show row + 1 ≤ rows by omega))
  have offset : width * (row * regs + reg) = row * (regs * width) + width * reg := by
    rw [Nat.mul_add, Nat.mul_comm width (row * regs), Nat.mul_assoc]
  refine ⟨offset, ?_, ⟨row * regs + reg, by rw [Nat.mul_assoc]⟩⟩
  change width * (row * regs + reg) + width ≤ rows * (regs * width)
  rw [← Nat.mul_succ, ← Nat.mul_assoc, Nat.mul_comm (rows * regs)]
  exact Nat.mul_le_mul_left width within

def OutputStoresContract : Prop :=
  ∀ (capacity width count block : Nat),
    0 < width → width * count = 128 → 128 ≤ capacity → block < count →
    width * block + width ≤ 128 ∧ width * block + width ≤ capacity

theorem outputStoresContract_proven : OutputStoresContract := by
  intro capacity width count block _ exactWidth fits h
  have stop := Nat.mul_le_mul_left width (show block + 1 ≤ count by omega)
  rw [Nat.mul_succ, exactWidth] at stop
  exact ⟨stop, Nat.le_trans stop fits⟩

def StreamSizeContract : Prop :=
  (2 * 256 * (totalRows + 1) = 152285696) ∧
  (2 * 128 * (totalRows + 1) = 76142848) ∧
  (∀ width, width = 128 ∨ width = 256 → 2 * width * (totalRows + 1) < 2147483648)

theorem streamSizeContract_proven : StreamSizeContract := by
  exact ⟨rfl, rfl, fun _ h => by rcases h with (rfl | rfl) <;> decide⟩

def PhaseBucketContract : Prop :=
  ∀ ply : Nat, (ply / 10 < 6 ↔ ply < 60) ∧
    (∀ bucket, ply / 10 = bucket ↔ 10 * bucket ≤ ply ∧ ply < 10 * (bucket + 1))

theorem phaseBucketContract_proven : PhaseBucketContract := by
  intro ply
  exact ⟨by omega, fun _ => by omega⟩

end NeuralReversi.Eval.InputLayer

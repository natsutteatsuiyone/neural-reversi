import NeuralReversi.Eval.InputLayer.Layout

namespace NeuralReversi.Eval.SmallNetwork

def releaseOffset (modulus ply : Nat) : Nat :=
  if ply ≥ 30 then ply - 30 else modulus - (30 - ply)

def RouteContract : Prop :=
  ∀ modulus ply, 60 ≤ modulus → ply < modulus →
    (releaseOffset modulus ply / 10 < 3 ↔ 30 ≤ ply ∧ ply < 60) ∧
    (releaseOffset modulus ply / 10 < 3 →
      releaseOffset modulus ply = ply - 30 ∧ releaseOffset modulus ply < 30 ∧
      ∀ bank, releaseOffset modulus ply / 10 = bank ↔
        30 + 10 * bank ≤ ply ∧ ply < 30 + 10 * (bank + 1))

theorem routeContract_proven : RouteContract := by
  intro modulus ply large below
  unfold releaseOffset
  split
  · refine ⟨by omega, fun bankOk => ⟨rfl, by omega, fun _ => by omega⟩⟩
  · exact ⟨by omega, by omega⟩

def inputBytes : Nat := 2 * 128 * (InputLayer.totalRows + 1)
def outputBytes : Nat := 4 + 2 * 128
def modelBytes : Nat := 3 * inputBytes + 30 * outputBytes

inductive DecodedLength where
  | short
  | exact
  | trailing

def classifyLength (bytes : Nat) : DecodedLength :=
  if bytes < modelBytes then .short else if bytes = modelBytes then .exact else .trailing

def StreamContract : Prop :=
  modelBytes = 228436344 ∧
  (∀ bank, bank < 3 → bank * inputBytes + inputBytes ≤ 3 * inputBytes) ∧
  (∀ head, head < 30 →
    3 * inputBytes + head * outputBytes + outputBytes ≤ modelBytes) ∧
  (∀ bytes, (classifyLength bytes = .short ↔ bytes < modelBytes) ∧
    (classifyLength bytes = .exact ↔ bytes = modelBytes) ∧
    (classifyLength bytes = .trailing ↔ modelBytes < bytes))

theorem streamContract_proven : StreamContract := by
  refine ⟨rfl, ?_, ?_, ?_⟩
  · intro bank h
    change bank * 76142848 + 76142848 ≤ 228428544
    omega
  · intro head h
    change 228428544 + head * 260 + 260 ≤ 228436344
    omega
  · intro bytes
    rcases Nat.lt_trichotomy bytes modelBytes with short | eq | long
    · simp [classifyLength, short, Nat.ne_of_lt short, Nat.not_lt_of_ge (Nat.le_of_lt short)]
    · subst bytes
      simp [classifyLength]
    · simp [classifyLength, Nat.not_lt_of_ge (Nat.le_of_lt long), (Nat.ne_of_lt long).symm, long]

def NeonLoadsContract : Prop :=
  ∀ feature,
    InputLayer.validatePrefix feature 32 = true →
    ∀ i, i < 32 → ∀ half reg, half < 2 → reg < 8 →
      let row := InputLayer.patternStart i + feature i
      let col := 64 * half + 8 * reg
      col + 8 ≤ 128 ∧
      128 * row + col + 8 ≤ 128 * InputLayer.totalRows ∧
      16 ∣ 2 * (128 * row + col)

theorem neonLoadsContract_proven : NeonLoadsContract := by
  intro feature valid i iOk half reg halfOk regOk
  have rowOk := (InputLayer.validateFeatureContract_proven feature).2 valid i iOk
  refine ⟨by omega, by omega, ?_⟩
  refine ⟨16 * (InputLayer.patternStart i + feature i) + 8 * half + reg, ?_⟩
  omega

end NeuralReversi.Eval.SmallNetwork

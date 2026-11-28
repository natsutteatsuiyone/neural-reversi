module

@[expose] public section

namespace NeuralReversi.Eval.LinearLayer

def low7 (a : Int) : Int := a % 128
def highBit (a : Int) : Int := a / 128 * 128
def signedHighBit (a : Int) : Int :=
  if 128 ≤ highBit a then highBit a - 256 else highBit a

def byteDot (a b : Nat → Int) : Int :=
  a 0 * b 0 + a 1 * b 1 + a 2 * b 2 + a 3 * b 3

def satPair (a₀ a₁ b₀ b₁ : Int) : Int :=
  min (max (-32768) (a₀ * b₀ + a₁ * b₁)) 32767

def x86EmulatedDot (a b : Nat → Int) : Int :=
  (satPair (low7 (a 0)) (low7 (a 1)) (b 0) (b 1) +
    satPair (low7 (a 2)) (low7 (a 3)) (b 2) (b 3)) +
  (satPair (highBit (a 0)) (highBit (a 1)) (b 0) (b 1) +
    satPair (highBit (a 2)) (highBit (a 3)) (b 2) (b 3))

def sdotEmulatedDot (a b : Nat → Int) : Int :=
  byteDot (fun lane => low7 (a lane)) b - byteDot (fun lane => signedHighBit (a lane)) b

def widenedDot (a b : Nat → Int) : Int :=
  (a 0 * b 0 + a 1 * b 1) + (a 2 * b 2 + a 3 * b 3)

private theorem splitByte {a : Int} (h : 0 ≤ a ∧ a ≤ 255) :
    0 ≤ low7 a ∧ low7 a ≤ 127 ∧ 0 ≤ highBit a ∧ highBit a ≤ 128 ∧
    low7 a + highBit a = a ∧ low7 a - signedHighBit a = a := by
  unfold low7 signedHighBit highBit
  split <;> omega

private theorem productBound {a b : Int}
    (aLo : 0 ≤ a) (bLo : -128 ≤ b) (bHi : b ≤ 127) :
    (-128) * a ≤ a * b ∧ a * b ≤ 127 * a := by
  simpa [Int.mul_comm a] using
    And.intro (Int.mul_le_mul_of_nonneg_left bLo aLo) (Int.mul_le_mul_of_nonneg_left bHi aLo)

private theorem exactPair {a₀ a₁ b₀ b₁ : Int}
    (a₀ok : 0 ≤ a₀ ∧ a₀ ≤ 128) (a₁ok : 0 ≤ a₁ ∧ a₁ ≤ 128)
    (b₀ok : -128 ≤ b₀ ∧ b₀ ≤ 127) (b₁ok : -128 ≤ b₁ ∧ b₁ ≤ 127) :
    satPair a₀ a₁ b₀ b₁ = a₀ * b₀ + a₁ * b₁ := by
  have p₀ := productBound a₀ok.1 b₀ok.1 b₀ok.2
  have p₁ := productBound a₁ok.1 b₁ok.1 b₁ok.2
  unfold satPair
  omega

private theorem x86EmulatedExact (a b : Nat → Int)
    (ha : ∀ lane, lane < 4 → 0 ≤ a lane ∧ a lane ≤ 255)
    (hb : ∀ lane, lane < 4 → -128 ≤ b lane ∧ b lane ≤ 127) :
    x86EmulatedDot a b = byteDot a b := by
  have low (lane) (h : lane < 4) : 0 ≤ low7 (a lane) ∧ low7 (a lane) ≤ 128 := by
    have s := splitByte (ha lane h)
    exact ⟨s.1, by omega⟩
  have high (lane) (h : lane < 4) : 0 ≤ highBit (a lane) ∧ highBit (a lane) ≤ 128 := by
    have s := splitByte (ha lane h)
    exact ⟨s.2.2.1, s.2.2.2.1⟩
  have product (lane) (h : lane < 4) :
      low7 (a lane) * b lane + highBit (a lane) * b lane = a lane * b lane := by
    rw [← Int.add_mul, (splitByte (ha lane h)).2.2.2.2.1]
  unfold x86EmulatedDot byteDot
  rw [exactPair (low 0 (by decide)) (low 1 (by decide)) (hb 0 (by decide)) (hb 1 (by decide)),
    exactPair (low 2 (by decide)) (low 3 (by decide)) (hb 2 (by decide)) (hb 3 (by decide)),
    exactPair (high 0 (by decide)) (high 1 (by decide)) (hb 0 (by decide)) (hb 1 (by decide)),
    exactPair (high 2 (by decide)) (high 3 (by decide)) (hb 2 (by decide)) (hb 3 (by decide))]
  have p₀ := product 0 (by decide)
  have p₁ := product 1 (by decide)
  have p₂ := product 2 (by decide)
  have p₃ := product 3 (by decide)
  omega

private theorem sdotEmulatedExact (a b : Nat → Int)
    (ha : ∀ lane, lane < 4 → 0 ≤ a lane ∧ a lane ≤ 255) :
    sdotEmulatedDot a b = byteDot a b := by
  have product (lane) (h : lane < 4) :
      low7 (a lane) * b lane - signedHighBit (a lane) * b lane = a lane * b lane := by
    rw [← Int.sub_mul, (splitByte (ha lane h)).2.2.2.2.2]
  have p₀ := product 0 (by decide)
  have p₁ := product 1 (by decide)
  have p₂ := product 2 (by decide)
  have p₃ := product 3 (by decide)
  unfold sdotEmulatedDot byteDot
  dsimp
  omega

def MixedDotContract : Prop :=
  ∀ (a b : Nat → Int),
    (∀ lane, lane < 4 → 0 ≤ a lane ∧ a lane ≤ 255) →
    (∀ lane, lane < 4 → -128 ≤ b lane ∧ b lane ≤ 127) →
    x86EmulatedDot a b = byteDot a b ∧ sdotEmulatedDot a b = byteDot a b ∧
    widenedDot a b = byteDot a b ∧
    (∀ lane, lane < 4 → -32640 ≤ a lane * b lane ∧ a lane * b lane ≤ 32385) ∧
    -130560 ≤ byteDot a b ∧ byteDot a b ≤ 129540

theorem mixedDotContract_proven : MixedDotContract := by
  intro a b ha hb
  have bounds (lane) (h : lane < 4) : -32640 ≤ a lane * b lane ∧ a lane * b lane ≤ 32385 := by
    have p := productBound (ha lane h).1 (hb lane h).1 (hb lane h).2
    have hi := (ha lane h).2
    omega
  refine ⟨x86EmulatedExact a b ha hb, sdotEmulatedExact a b ha, ?_, bounds, ?_⟩
  · unfold widenedDot byteDot
    omega
  · have p₀ := bounds 0 (by decide)
    have p₁ := bounds 1 (by decide)
    have p₂ := bounds 2 (by decide)
    have p₃ := bounds 3 (by decide)
    unfold byteDot
    omega

end NeuralReversi.Eval.LinearLayer

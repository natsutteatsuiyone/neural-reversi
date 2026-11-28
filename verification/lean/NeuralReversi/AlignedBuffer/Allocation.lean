module

@[expose] public section

namespace NeuralReversi.AlignedBuffer

inductive LayoutResult where
  | multiplyOverflow
  | noBytes
  | invalidLayout
  | allocated (bytes : Nat)
  deriving DecidableEq, Repr

def cacheLine : Nat := 64

def roundToLine (bytes : Nat) : Nat := cacheLine * ((bytes + cacheLine - 1) / cacheLine)

def checkedBytes (pointerBits bytes : Nat) : LayoutResult :=
  if 2 ^ pointerBits ≤ bytes then .multiplyOverflow
  else if bytes = 0 then .noBytes
  else if bytes ≤ 2 ^ (pointerBits - 1) - cacheLine then .allocated bytes
  else .invalidLayout

def checkedLayout (pointerBits count itemSize : Nat) : LayoutResult :=
  checkedBytes pointerBits (count * itemSize)

theorem layout_limit (bits bytes : Nat) (width : bits = 32 ∨ bits = 64) :
    bytes ≤ 2 ^ (bits - 1) - cacheLine ↔ roundToLine bytes < 2 ^ (bits - 1) := by
  rcases width with rfl | rfl <;> unfold roundToLine cacheLine <;> omega

def LayoutContract : Prop :=
  ∀ bits count itemSize : Nat, (bits = 32 ∨ bits = 64) →
    let bytes := count * itemSize
    (checkedLayout bits count itemSize = .multiplyOverflow ↔ 2 ^ bits ≤ bytes) ∧
    (checkedLayout bits count itemSize = .noBytes ↔ count = 0 ∨ itemSize = 0) ∧
    (checkedLayout bits count itemSize = .invalidLayout ↔
      bytes < 2 ^ bits ∧ 2 ^ (bits - 1) ≤ roundToLine bytes) ∧
    (∀ value, checkedLayout bits count itemSize = .allocated value ↔
      0 < bytes ∧ bytes < 2 ^ bits ∧ roundToLine bytes < 2 ^ (bits - 1) ∧ value = bytes)

theorem layoutContract_proven : LayoutContract := by
  intro bits count itemSize width
  simp only [checkedLayout]
  rw [← Nat.mul_eq_zero]
  generalize count * itemSize = bytes
  have limit := layout_limit bits bytes width
  by_cases overflow : 2 ^ bits ≤ bytes
  · have nonzero : bytes ≠ 0 := by have := Nat.two_pow_pos bits; omega
    have large : ¬ bytes < 2 ^ bits := by omega
    simp [checkedBytes, overflow, nonzero, large]
  · have fits : bytes < 2 ^ bits := by omega
    by_cases zero : bytes = 0
    · subst bytes
      have signed : 0 < 2 ^ (bits - 1) := Nat.two_pow_pos _
      simp [checkedBytes, overflow, roundToLine, cacheLine, signed, fits]
    · have positive : 0 < bytes := Nat.pos_of_ne_zero zero
      by_cases localLayout : bytes ≤ 2 ^ (bits - 1) - cacheLine
      · have rounded := limit.mp localLayout
        have signed : ¬ 2 ^ (bits - 1) ≤ roundToLine bytes := by omega
        simp [checkedBytes, overflow, zero, localLayout, fits, positive, rounded, signed, eq_comm]
      · have rounded : ¬ roundToLine bytes < 2 ^ (bits - 1) := mt limit.mpr localLayout
        have signed : 2 ^ (bits - 1) ≤ roundToLine bytes := by omega
        simp [checkedBytes, overflow, zero, localLayout, fits, rounded, signed]

def SlotContract : Prop :=
  ∀ (base count itemSize align first second : Nat), cacheLine ∣ base → align ∣ cacheLine →
    align ∣ itemSize → first < count → second < count →
    align ∣ (base + first * itemSize) ∧
    first * itemSize + itemSize ≤ count * itemSize ∧
    (first < second → first * itemSize + itemSize ≤ second * itemSize) ∧
    (0 < itemSize → first * itemSize < first * itemSize + itemSize)

theorem slotContract_proven : SlotContract := by
  intro base count itemSize align first second line element stride valid _
  have fit (n : Nat) (before : first < n) := Nat.mul_le_mul_right itemSize (Nat.succ_le_of_lt before)
  refine ⟨Nat.dvd_add (Nat.dvd_trans element line) (Nat.dvd_mul_left_of_dvd stride first), ?_, ?_, by omega⟩
  · simpa [Nat.succ_mul] using fit count valid
  · intro order
    simpa [Nat.succ_mul] using fit second order

def AlignmentContract : Prop :=
  ∀ exponent : Nat, 2 ^ exponent ≤ cacheLine → 2 ^ exponent ∣ cacheLine

theorem alignmentContract_proven : AlignmentContract := by
  intro exponent permitted
  change 2 ^ exponent ∣ 2 ^ 6
  apply Nat.pow_dvd_pow
  by_cases bounded : exponent ≤ 6
  · exact bounded
  · have tooWide := Nat.pow_lt_pow_right (by decide : 1 < 2) (by omega : 6 < exponent)
    unfold cacheLine at permitted
    omega

def NoBytesContract : Prop :=
  ∀ bits count itemSize align : Nat, (bits = 32 ∨ bits = 64) → align ∣ cacheLine →
    checkedLayout bits count itemSize = .noBytes →
    align ∣ cacheLine ∧ (∀ i, i < count → i * itemSize = 0)

theorem noBytesContract_proven : NoBytesContract := by
  intro bits count itemSize align width aligned noBytes
  refine ⟨aligned, fun i valid => ?_⟩
  rcases ((layoutContract_proven bits count itemSize width).2.1.mp noBytes) with zero | zero
  · omega
  · simp [zero]

def pageRounded (bytes page : Nat) : Nat := page * ((bytes + page - 1) / page)

def LargePageContract : Prop :=
  ∀ (pointerBits count itemSize page base bytes : Nat),
    (pointerBits = 32 ∨ pointerBits = 64) →
    checkedLayout pointerBits count itemSize = .allocated bytes →
    0 < page → page ≤ bytes → cacheLine ∣ page → page ∣ base →
    bytes ≤ pageRounded bytes page ∧ pageRounded bytes page < 2 ^ pointerBits ∧
      page ∣ pageRounded bytes page ∧ cacheLine ∣ base

theorem largePageContract_proven : LargePageContract := by
  intro pointerBits count itemSize page base bytes width allocated nonzero eligible large aligned
  have same := ((layoutContract_proven pointerBits count itemSize width).2.2.2 _).mp allocated
  have limit := (layout_limit pointerBits (count * itemSize) width).mpr same.2.2.1
  rw [← same.2.2.2] at limit
  have rem := Nat.mod_lt (bytes + page - 1) nonzero
  have exactSize := Nat.mod_add_div (bytes + page - 1) page
  refine ⟨by unfold pageRounded; omega, ?_, ?_, Nat.dvd_trans large aligned⟩
  · unfold pageRounded
    rcases width with rfl | rfl <;> unfold cacheLine at limit <;> omega
  · exact ⟨_, rfl⟩

example : checkedLayout 32 1 2147483584 = .allocated 2147483584 := by decide
example : checkedLayout 32 1 2147483585 = .invalidLayout := by decide
example : checkedLayout 64 18446744073709551615 2 = .multiplyOverflow := by decide
example : checkedLayout 64 18446744073709551615 0 = .noBytes := by decide

end NeuralReversi.AlignedBuffer

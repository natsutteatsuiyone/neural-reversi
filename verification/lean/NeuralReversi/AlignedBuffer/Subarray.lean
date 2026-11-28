module
public import NeuralReversi.AlignedBuffer.Allocation

@[expose] public section

namespace NeuralReversi.AlignedBuffer

def admittedSubarray (capacity offset length : Nat) : Prop :=
  cacheLine ∣ offset ∧ offset + length ≤ capacity ∧ roundToLine length = length

theorem noPadding (length : Nat) : roundToLine length = length ↔ cacheLine ∣ length := by
  constructor
  · intro exactSize
    rw [← exactSize]
    exact ⟨_, rfl⟩
  · rintro ⟨lines, rfl⟩
    have whole := Nat.mul_add_mod_self_left (cacheLine - 1) cacheLine lines
    have rem : (cacheLine * lines + cacheLine - 1) % cacheLine = cacheLine - 1 := by
      unfold cacheLine at *
      omega
    have := Nat.mod_add_div (cacheLine * lines + cacheLine - 1) cacheLine
    unfold roundToLine cacheLine at *
    omega

def SubarrayContract : Prop :=
  ∀ bits base capacity offset length : Nat,
    cacheLine ∣ base → capacity < 2 ^ bits → admittedSubarray capacity offset length →
    cacheLine ∣ base + offset ∧ offset + length < 2 ^ bits ∧ cacheLine ∣ length ∧
    offset ≤ capacity ∧
    (∀ byte, byte < roundToLine length → offset + byte < capacity)

theorem subarrayContract_proven : SubarrayContract := by
  intro bits base capacity offset length aligned addressable ⟨start, fits, exactSize⟩
  refine ⟨Nat.dvd_add aligned start, Nat.lt_of_le_of_lt fits addressable,
    (noPadding length).mp exactSize, by omega, ?_⟩
  intro byte valid
  rw [exactSize] at valid
  omega

def subarrayRead (storage : Nat → UInt8) (offset byte : Nat) : UInt8 :=
  storage (offset + byte)

def subarrayWrite (storage replacement : Nat → UInt8) (offset length byte : Nat) : UInt8 :=
  if offset ≤ byte ∧ byte < offset + length then replacement (byte - offset) else storage byte

def SubarrayWriteContract : Prop :=
  ∀ capacity offset length storage replacement, admittedSubarray capacity offset length →
    (∀ byte, byte < length →
      subarrayRead (subarrayWrite storage replacement offset length) offset byte = replacement byte) ∧
    (∀ byte, byte < offset ∨ offset + length ≤ byte →
      subarrayWrite storage replacement offset length byte = storage byte)

theorem subarrayWriteContract_proven : SubarrayWriteContract := by
  intro capacity offset length storage replacement _
  constructor
  · intro byte inside
    have hit : offset ≤ offset + byte ∧ offset + byte < offset + length := by omega
    simp [subarrayRead, subarrayWrite, hit]
  · intro byte outside
    have miss : ¬(offset ≤ byte ∧ byte < offset + length) := by omega
    simp [subarrayWrite, miss]

end NeuralReversi.AlignedBuffer

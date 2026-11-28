module
public import NeuralReversi.Square.UncheckedModel

@[expose] public section

namespace NeuralReversi.Square

inductive UncheckedObservation where
  | value (index : Nat)
  | panic (input : Nat)
  | invalidDiscriminant (byte : Nat)
  deriving DecidableEq, Repr

def observeUnchecked : UncheckedResult → UncheckedObservation
  | .value square => .value square.val
  | .panic input => .panic input
  | .invalidDiscriminant byte => .invalidDiscriminant byte.toNat

def uncheckedSpec (checked : Bool) (input : Nat) : UncheckedObservation :=
  if checked && decide (64 ≤ input) then .panic input
  else if input % 256 < 64 then .value (input % 256)
  else .invalidDiscriminant (input % 256)

theorem u8_unchecked_spec (checked : Bool) (value : Byte) :
    observeUnchecked (fromU8Unchecked checked value) = uncheckedSpec checked value.toNat := by
  have bound := value.isLt
  simp only [fromU8Unchecked, uncheckedSpec, Nat.mod_eq_of_lt bound]
  split
  · rfl
  · by_cases small : value.toNat < 64 <;> simp [transmuteByte, observeUnchecked, small]

theorem lowByte_spec (width : Nat) (checked : Bool) (value : BitVec width)
    (small : value.toNat < 64) :
    observeUnchecked (fromU8Unchecked checked (value.setWidth 8)) = .value value.toNat := by
  rw [u8_unchecked_spec]
  have bound : value.toNat < 256 := by omega
  simp [uncheckedSpec, BitVec.toNat_setWidth, Nat.mod_eq_of_lt bound, show ¬ 64 ≤ value.toNat by omega,
    small]

theorem wider_unchecked_spec (width : Nat) (checked : Bool) (value : BitVec width) :
    observeUnchecked (if checked && decide (64 ≤ value.toNat) then
      .panic value.toNat else fromU8Unchecked checked (value.setWidth 8)) =
        uncheckedSpec checked value.toNat := by
  cases checked
  · simp [uncheckedSpec, u8_unchecked_spec, BitVec.toNat_setWidth]
  · by_cases large : 64 ≤ value.toNat
    · simp [large, uncheckedSpec, observeUnchecked]
    · have small : value.toNat < 64 := by omega
      simp [large, lowByte_spec width true value small, uncheckedSpec,
        Nat.mod_eq_of_lt (show value.toNat < 256 by omega), small]

def UncheckedRuntimeContract : Prop :=
  (∀ (checked : Bool) (value : Byte),
    observeUnchecked (fromU8Unchecked checked value) = uncheckedSpec checked value.toNat) ∧
  (∀ (checked : Bool) (value : BitVec 32),
    observeUnchecked (fromU32Unchecked checked value) = uncheckedSpec checked value.toNat) ∧
  (∀ (checked : Bool) (value : BitVec 64),
    observeUnchecked (fromUsizeUnchecked64 checked value) = uncheckedSpec checked value.toNat)

theorem uncheckedRuntimeContract_proven : UncheckedRuntimeContract := by
  exact ⟨u8_unchecked_spec, wider_unchecked_spec 32, wider_unchecked_spec 64⟩

def UncheckedSafetyDomainContract : Prop :=
  ∀ (checked : Bool) (square : Square),
    fromU8Unchecked checked (BitVec.ofNat 8 square.val) = .value square ∧
    fromU32Unchecked checked (BitVec.ofNat 32 square.val) = .value square ∧
    fromUsizeUnchecked64 checked (BitVec.ofNat 64 square.val) = .value square

theorem uncheckedSafetyDomainContract_proven : UncheckedSafetyDomainContract := by
  intro checked square
  have small : square.val < 64 := square.isLt
  have b8 : square.val < 2^8 := by omega
  have b32 : square.val < 2^32 := by omega
  have b64 : square.val < 2^64 := by omega
  simp [fromU8Unchecked, fromU32Unchecked, fromUsizeUnchecked64, transmuteByte,
    BitVec.toNat_ofNat, Nat.mod_eq_of_lt b8,
    Nat.mod_eq_of_lt b32, Nat.mod_eq_of_lt b64, show ¬ 64 ≤ square.val by omega, small]

end NeuralReversi.Square

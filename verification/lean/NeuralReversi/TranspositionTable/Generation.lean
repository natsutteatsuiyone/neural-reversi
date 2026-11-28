module
public import NeuralReversi.TranspositionTable.Packing

@[expose] public section

namespace NeuralReversi.TranspositionTable.Generation

def current (raw : BitVec 8) : BitVec 8 := raw &&& 127#8

def increment (raw : BitVec 8) : BitVec 8 := current (raw + 1#8)

def advance : Nat → BitVec 8 → BitVec 8
  | 0, raw => current raw
  | n + 1, raw => advance n (increment raw)

theorem current_exact (raw : BitVec 8) : (current raw).toNat = raw.toNat % 128 := by
  rw [current, BitVec.toNat_and]
  change raw.toNat &&& (2^7 - 1) = raw.toNat % 128
  exact Nat.and_two_pow_sub_one_eq_mod _ _

theorem increment_exact (raw : BitVec 8) : (increment raw).toNat = (raw.toNat + 1) % 128 := by
  rw [increment, current_exact, BitVec.toNat_add]
  change (raw.toNat + 1) % 256 % 128 = (raw.toNat + 1) % 128
  omega

theorem advance_exact (n : Nat) (raw : BitVec 8) : (advance n raw).toNat = (raw.toNat + n) % 128 := by
  induction n generalizing raw with
  | zero => simpa [advance] using current_exact raw
  | succ n ih => rw [advance, ih, increment_exact]; omega

def RingContract : Prop :=
  (∀ raw, (current raw).toNat = raw.toNat % 128) ∧
  (∀ raw, (increment raw).toNat = ((current raw).toNat + 1) % 128) ∧
  (∀ n raw, (advance n raw).toNat = (raw.toNat + n) % 128) ∧
  (∀ n, (advance n 0).toNat = n % 128) ∧
  (∀ score move bound depth sel raw endgame pv,
    (Packing.newData score move bound depth sel (current raw) endgame pv).generation = current raw)

theorem ringContract_proven : RingContract := by
  refine ⟨current_exact, ?_, advance_exact, fun n => by simpa using advance_exact n 0, ?_⟩
  · intro raw
    rw [increment_exact, current_exact]
    omega
  · intro score move bound depth sel raw endgame pv
    change current raw &&& 127#8 = current raw
    simp [current, BitVec.and_assoc]

end NeuralReversi.TranspositionTable.Generation

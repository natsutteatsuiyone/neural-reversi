import NeuralReversi.Solve1.LineTable

namespace NeuralReversi.Solve1.LineSafety
open LineTable

def above (pos : Fin 8) (line : Byte) : Usize := line.setWidth 64 >>> (pos.val + 1)
def below (pos : Fin 8) (line : Byte) : Usize :=
  line.setWidth 64 &&& (((1 : Usize) <<< pos.val) - 1)
def logIndex (pos : Fin 8) (line : Byte) : Usize := 63 - (below pos line).clz

def upperCount (pos : Fin 8) (line : Byte) : Nat :=
  if above pos line = 0 then 0 else 2 * (above pos line).ctz.toNat

def lowerCount (pos : Fin 8) (line : Byte) : Nat :=
  if below pos line = 0 then 0 else 2 * (pos.val - 1 - (logIndex pos line).toNat)

def Safe (pos : Fin 8) (line : Byte) : Prop :=
  (above pos line ≠ 0 →
    (above pos line).ctz.toNat ≤ 6 ∧
    ((above pos line).ctz.setWidth 8).toNat = (above pos line).ctz.toNat) ∧
  (below pos line ≠ 0 →
    1 ≤ pos.val ∧ (below pos line).clz.toNat ≤ 63 ∧
    (logIndex pos line).toNat < pos.val ∧
    BitVec.usubOverflow (BitVec.ofNat 64 pos.val) 1 = false ∧
    BitVec.usubOverflow 63 (below pos line).clz = false ∧
    BitVec.usubOverflow (BitVec.ofNat 64 pos.val - 1) (logIndex pos line) = false ∧
    ((BitVec.ofNat 64 pos.val - 1 - logIndex pos line).setWidth 8).toNat =
      pos.val - 1 - (logIndex pos line).toNat) ∧
  upperCount pos line + lowerCount pos line ≤ 12 ∧
  (countFlip pos line).toNat = upperCount pos line + lowerCount pos line

set_option maxRecDepth 100000
set_option maxHeartbeats 0

private theorem safe_0_0 : ∀ n : Fin 16, Safe ⟨0, by decide⟩ (BitVec.ofNat 8 (16 * 0 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_0_1 : ∀ n : Fin 16, Safe ⟨0, by decide⟩ (BitVec.ofNat 8 (16 * 1 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_0_2 : ∀ n : Fin 16, Safe ⟨0, by decide⟩ (BitVec.ofNat 8 (16 * 2 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_0_3 : ∀ n : Fin 16, Safe ⟨0, by decide⟩ (BitVec.ofNat 8 (16 * 3 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_0_4 : ∀ n : Fin 16, Safe ⟨0, by decide⟩ (BitVec.ofNat 8 (16 * 4 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_0_5 : ∀ n : Fin 16, Safe ⟨0, by decide⟩ (BitVec.ofNat 8 (16 * 5 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_0_6 : ∀ n : Fin 16, Safe ⟨0, by decide⟩ (BitVec.ofNat 8 (16 * 6 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_0_7 : ∀ n : Fin 16, Safe ⟨0, by decide⟩ (BitVec.ofNat 8 (16 * 7 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_0_8 : ∀ n : Fin 16, Safe ⟨0, by decide⟩ (BitVec.ofNat 8 (16 * 8 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_0_9 : ∀ n : Fin 16, Safe ⟨0, by decide⟩ (BitVec.ofNat 8 (16 * 9 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_0_10 : ∀ n : Fin 16, Safe ⟨0, by decide⟩ (BitVec.ofNat 8 (16 * 10 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_0_11 : ∀ n : Fin 16, Safe ⟨0, by decide⟩ (BitVec.ofNat 8 (16 * 11 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_0_12 : ∀ n : Fin 16, Safe ⟨0, by decide⟩ (BitVec.ofNat 8 (16 * 12 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_0_13 : ∀ n : Fin 16, Safe ⟨0, by decide⟩ (BitVec.ofNat 8 (16 * 13 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_0_14 : ∀ n : Fin 16, Safe ⟨0, by decide⟩ (BitVec.ofNat 8 (16 * 14 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_0_15 : ∀ n : Fin 16, Safe ⟨0, by decide⟩ (BitVec.ofNat 8 (16 * 15 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_0 (line : Byte) : Safe ⟨0, by decide⟩ line := by
  have bound := line.isLt
  generalize hq : line.toNat / 16 = q
  have small : q < 16 := by omega
  match q with
  | 0 =>
    have eq : BitVec.ofNat 8 (16 * 0 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_0_0 ⟨line.toNat % 16, by omega⟩
  | 1 =>
    have eq : BitVec.ofNat 8 (16 * 1 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_0_1 ⟨line.toNat % 16, by omega⟩
  | 2 =>
    have eq : BitVec.ofNat 8 (16 * 2 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_0_2 ⟨line.toNat % 16, by omega⟩
  | 3 =>
    have eq : BitVec.ofNat 8 (16 * 3 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_0_3 ⟨line.toNat % 16, by omega⟩
  | 4 =>
    have eq : BitVec.ofNat 8 (16 * 4 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_0_4 ⟨line.toNat % 16, by omega⟩
  | 5 =>
    have eq : BitVec.ofNat 8 (16 * 5 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_0_5 ⟨line.toNat % 16, by omega⟩
  | 6 =>
    have eq : BitVec.ofNat 8 (16 * 6 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_0_6 ⟨line.toNat % 16, by omega⟩
  | 7 =>
    have eq : BitVec.ofNat 8 (16 * 7 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_0_7 ⟨line.toNat % 16, by omega⟩
  | 8 =>
    have eq : BitVec.ofNat 8 (16 * 8 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_0_8 ⟨line.toNat % 16, by omega⟩
  | 9 =>
    have eq : BitVec.ofNat 8 (16 * 9 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_0_9 ⟨line.toNat % 16, by omega⟩
  | 10 =>
    have eq : BitVec.ofNat 8 (16 * 10 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_0_10 ⟨line.toNat % 16, by omega⟩
  | 11 =>
    have eq : BitVec.ofNat 8 (16 * 11 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_0_11 ⟨line.toNat % 16, by omega⟩
  | 12 =>
    have eq : BitVec.ofNat 8 (16 * 12 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_0_12 ⟨line.toNat % 16, by omega⟩
  | 13 =>
    have eq : BitVec.ofNat 8 (16 * 13 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_0_13 ⟨line.toNat % 16, by omega⟩
  | 14 =>
    have eq : BitVec.ofNat 8 (16 * 14 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_0_14 ⟨line.toNat % 16, by omega⟩
  | 15 =>
    have eq : BitVec.ofNat 8 (16 * 15 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_0_15 ⟨line.toNat % 16, by omega⟩
  | q + 16 => omega

private theorem safe_1_0 : ∀ n : Fin 16, Safe ⟨1, by decide⟩ (BitVec.ofNat 8 (16 * 0 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_1_1 : ∀ n : Fin 16, Safe ⟨1, by decide⟩ (BitVec.ofNat 8 (16 * 1 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_1_2 : ∀ n : Fin 16, Safe ⟨1, by decide⟩ (BitVec.ofNat 8 (16 * 2 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_1_3 : ∀ n : Fin 16, Safe ⟨1, by decide⟩ (BitVec.ofNat 8 (16 * 3 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_1_4 : ∀ n : Fin 16, Safe ⟨1, by decide⟩ (BitVec.ofNat 8 (16 * 4 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_1_5 : ∀ n : Fin 16, Safe ⟨1, by decide⟩ (BitVec.ofNat 8 (16 * 5 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_1_6 : ∀ n : Fin 16, Safe ⟨1, by decide⟩ (BitVec.ofNat 8 (16 * 6 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_1_7 : ∀ n : Fin 16, Safe ⟨1, by decide⟩ (BitVec.ofNat 8 (16 * 7 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_1_8 : ∀ n : Fin 16, Safe ⟨1, by decide⟩ (BitVec.ofNat 8 (16 * 8 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_1_9 : ∀ n : Fin 16, Safe ⟨1, by decide⟩ (BitVec.ofNat 8 (16 * 9 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_1_10 : ∀ n : Fin 16, Safe ⟨1, by decide⟩ (BitVec.ofNat 8 (16 * 10 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_1_11 : ∀ n : Fin 16, Safe ⟨1, by decide⟩ (BitVec.ofNat 8 (16 * 11 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_1_12 : ∀ n : Fin 16, Safe ⟨1, by decide⟩ (BitVec.ofNat 8 (16 * 12 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_1_13 : ∀ n : Fin 16, Safe ⟨1, by decide⟩ (BitVec.ofNat 8 (16 * 13 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_1_14 : ∀ n : Fin 16, Safe ⟨1, by decide⟩ (BitVec.ofNat 8 (16 * 14 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_1_15 : ∀ n : Fin 16, Safe ⟨1, by decide⟩ (BitVec.ofNat 8 (16 * 15 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_1 (line : Byte) : Safe ⟨1, by decide⟩ line := by
  have bound := line.isLt
  generalize hq : line.toNat / 16 = q
  have small : q < 16 := by omega
  match q with
  | 0 =>
    have eq : BitVec.ofNat 8 (16 * 0 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_1_0 ⟨line.toNat % 16, by omega⟩
  | 1 =>
    have eq : BitVec.ofNat 8 (16 * 1 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_1_1 ⟨line.toNat % 16, by omega⟩
  | 2 =>
    have eq : BitVec.ofNat 8 (16 * 2 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_1_2 ⟨line.toNat % 16, by omega⟩
  | 3 =>
    have eq : BitVec.ofNat 8 (16 * 3 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_1_3 ⟨line.toNat % 16, by omega⟩
  | 4 =>
    have eq : BitVec.ofNat 8 (16 * 4 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_1_4 ⟨line.toNat % 16, by omega⟩
  | 5 =>
    have eq : BitVec.ofNat 8 (16 * 5 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_1_5 ⟨line.toNat % 16, by omega⟩
  | 6 =>
    have eq : BitVec.ofNat 8 (16 * 6 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_1_6 ⟨line.toNat % 16, by omega⟩
  | 7 =>
    have eq : BitVec.ofNat 8 (16 * 7 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_1_7 ⟨line.toNat % 16, by omega⟩
  | 8 =>
    have eq : BitVec.ofNat 8 (16 * 8 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_1_8 ⟨line.toNat % 16, by omega⟩
  | 9 =>
    have eq : BitVec.ofNat 8 (16 * 9 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_1_9 ⟨line.toNat % 16, by omega⟩
  | 10 =>
    have eq : BitVec.ofNat 8 (16 * 10 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_1_10 ⟨line.toNat % 16, by omega⟩
  | 11 =>
    have eq : BitVec.ofNat 8 (16 * 11 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_1_11 ⟨line.toNat % 16, by omega⟩
  | 12 =>
    have eq : BitVec.ofNat 8 (16 * 12 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_1_12 ⟨line.toNat % 16, by omega⟩
  | 13 =>
    have eq : BitVec.ofNat 8 (16 * 13 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_1_13 ⟨line.toNat % 16, by omega⟩
  | 14 =>
    have eq : BitVec.ofNat 8 (16 * 14 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_1_14 ⟨line.toNat % 16, by omega⟩
  | 15 =>
    have eq : BitVec.ofNat 8 (16 * 15 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_1_15 ⟨line.toNat % 16, by omega⟩
  | q + 16 => omega

private theorem safe_2_0 : ∀ n : Fin 16, Safe ⟨2, by decide⟩ (BitVec.ofNat 8 (16 * 0 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_2_1 : ∀ n : Fin 16, Safe ⟨2, by decide⟩ (BitVec.ofNat 8 (16 * 1 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_2_2 : ∀ n : Fin 16, Safe ⟨2, by decide⟩ (BitVec.ofNat 8 (16 * 2 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_2_3 : ∀ n : Fin 16, Safe ⟨2, by decide⟩ (BitVec.ofNat 8 (16 * 3 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_2_4 : ∀ n : Fin 16, Safe ⟨2, by decide⟩ (BitVec.ofNat 8 (16 * 4 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_2_5 : ∀ n : Fin 16, Safe ⟨2, by decide⟩ (BitVec.ofNat 8 (16 * 5 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_2_6 : ∀ n : Fin 16, Safe ⟨2, by decide⟩ (BitVec.ofNat 8 (16 * 6 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_2_7 : ∀ n : Fin 16, Safe ⟨2, by decide⟩ (BitVec.ofNat 8 (16 * 7 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_2_8 : ∀ n : Fin 16, Safe ⟨2, by decide⟩ (BitVec.ofNat 8 (16 * 8 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_2_9 : ∀ n : Fin 16, Safe ⟨2, by decide⟩ (BitVec.ofNat 8 (16 * 9 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_2_10 : ∀ n : Fin 16, Safe ⟨2, by decide⟩ (BitVec.ofNat 8 (16 * 10 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_2_11 : ∀ n : Fin 16, Safe ⟨2, by decide⟩ (BitVec.ofNat 8 (16 * 11 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_2_12 : ∀ n : Fin 16, Safe ⟨2, by decide⟩ (BitVec.ofNat 8 (16 * 12 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_2_13 : ∀ n : Fin 16, Safe ⟨2, by decide⟩ (BitVec.ofNat 8 (16 * 13 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_2_14 : ∀ n : Fin 16, Safe ⟨2, by decide⟩ (BitVec.ofNat 8 (16 * 14 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_2_15 : ∀ n : Fin 16, Safe ⟨2, by decide⟩ (BitVec.ofNat 8 (16 * 15 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_2 (line : Byte) : Safe ⟨2, by decide⟩ line := by
  have bound := line.isLt
  generalize hq : line.toNat / 16 = q
  have small : q < 16 := by omega
  match q with
  | 0 =>
    have eq : BitVec.ofNat 8 (16 * 0 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_2_0 ⟨line.toNat % 16, by omega⟩
  | 1 =>
    have eq : BitVec.ofNat 8 (16 * 1 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_2_1 ⟨line.toNat % 16, by omega⟩
  | 2 =>
    have eq : BitVec.ofNat 8 (16 * 2 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_2_2 ⟨line.toNat % 16, by omega⟩
  | 3 =>
    have eq : BitVec.ofNat 8 (16 * 3 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_2_3 ⟨line.toNat % 16, by omega⟩
  | 4 =>
    have eq : BitVec.ofNat 8 (16 * 4 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_2_4 ⟨line.toNat % 16, by omega⟩
  | 5 =>
    have eq : BitVec.ofNat 8 (16 * 5 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_2_5 ⟨line.toNat % 16, by omega⟩
  | 6 =>
    have eq : BitVec.ofNat 8 (16 * 6 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_2_6 ⟨line.toNat % 16, by omega⟩
  | 7 =>
    have eq : BitVec.ofNat 8 (16 * 7 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_2_7 ⟨line.toNat % 16, by omega⟩
  | 8 =>
    have eq : BitVec.ofNat 8 (16 * 8 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_2_8 ⟨line.toNat % 16, by omega⟩
  | 9 =>
    have eq : BitVec.ofNat 8 (16 * 9 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_2_9 ⟨line.toNat % 16, by omega⟩
  | 10 =>
    have eq : BitVec.ofNat 8 (16 * 10 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_2_10 ⟨line.toNat % 16, by omega⟩
  | 11 =>
    have eq : BitVec.ofNat 8 (16 * 11 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_2_11 ⟨line.toNat % 16, by omega⟩
  | 12 =>
    have eq : BitVec.ofNat 8 (16 * 12 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_2_12 ⟨line.toNat % 16, by omega⟩
  | 13 =>
    have eq : BitVec.ofNat 8 (16 * 13 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_2_13 ⟨line.toNat % 16, by omega⟩
  | 14 =>
    have eq : BitVec.ofNat 8 (16 * 14 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_2_14 ⟨line.toNat % 16, by omega⟩
  | 15 =>
    have eq : BitVec.ofNat 8 (16 * 15 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_2_15 ⟨line.toNat % 16, by omega⟩
  | q + 16 => omega

private theorem safe_3_0 : ∀ n : Fin 16, Safe ⟨3, by decide⟩ (BitVec.ofNat 8 (16 * 0 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_3_1 : ∀ n : Fin 16, Safe ⟨3, by decide⟩ (BitVec.ofNat 8 (16 * 1 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_3_2 : ∀ n : Fin 16, Safe ⟨3, by decide⟩ (BitVec.ofNat 8 (16 * 2 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_3_3 : ∀ n : Fin 16, Safe ⟨3, by decide⟩ (BitVec.ofNat 8 (16 * 3 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_3_4 : ∀ n : Fin 16, Safe ⟨3, by decide⟩ (BitVec.ofNat 8 (16 * 4 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_3_5 : ∀ n : Fin 16, Safe ⟨3, by decide⟩ (BitVec.ofNat 8 (16 * 5 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_3_6 : ∀ n : Fin 16, Safe ⟨3, by decide⟩ (BitVec.ofNat 8 (16 * 6 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_3_7 : ∀ n : Fin 16, Safe ⟨3, by decide⟩ (BitVec.ofNat 8 (16 * 7 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_3_8 : ∀ n : Fin 16, Safe ⟨3, by decide⟩ (BitVec.ofNat 8 (16 * 8 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_3_9 : ∀ n : Fin 16, Safe ⟨3, by decide⟩ (BitVec.ofNat 8 (16 * 9 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_3_10 : ∀ n : Fin 16, Safe ⟨3, by decide⟩ (BitVec.ofNat 8 (16 * 10 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_3_11 : ∀ n : Fin 16, Safe ⟨3, by decide⟩ (BitVec.ofNat 8 (16 * 11 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_3_12 : ∀ n : Fin 16, Safe ⟨3, by decide⟩ (BitVec.ofNat 8 (16 * 12 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_3_13 : ∀ n : Fin 16, Safe ⟨3, by decide⟩ (BitVec.ofNat 8 (16 * 13 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_3_14 : ∀ n : Fin 16, Safe ⟨3, by decide⟩ (BitVec.ofNat 8 (16 * 14 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_3_15 : ∀ n : Fin 16, Safe ⟨3, by decide⟩ (BitVec.ofNat 8 (16 * 15 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_3 (line : Byte) : Safe ⟨3, by decide⟩ line := by
  have bound := line.isLt
  generalize hq : line.toNat / 16 = q
  have small : q < 16 := by omega
  match q with
  | 0 =>
    have eq : BitVec.ofNat 8 (16 * 0 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_3_0 ⟨line.toNat % 16, by omega⟩
  | 1 =>
    have eq : BitVec.ofNat 8 (16 * 1 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_3_1 ⟨line.toNat % 16, by omega⟩
  | 2 =>
    have eq : BitVec.ofNat 8 (16 * 2 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_3_2 ⟨line.toNat % 16, by omega⟩
  | 3 =>
    have eq : BitVec.ofNat 8 (16 * 3 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_3_3 ⟨line.toNat % 16, by omega⟩
  | 4 =>
    have eq : BitVec.ofNat 8 (16 * 4 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_3_4 ⟨line.toNat % 16, by omega⟩
  | 5 =>
    have eq : BitVec.ofNat 8 (16 * 5 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_3_5 ⟨line.toNat % 16, by omega⟩
  | 6 =>
    have eq : BitVec.ofNat 8 (16 * 6 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_3_6 ⟨line.toNat % 16, by omega⟩
  | 7 =>
    have eq : BitVec.ofNat 8 (16 * 7 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_3_7 ⟨line.toNat % 16, by omega⟩
  | 8 =>
    have eq : BitVec.ofNat 8 (16 * 8 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_3_8 ⟨line.toNat % 16, by omega⟩
  | 9 =>
    have eq : BitVec.ofNat 8 (16 * 9 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_3_9 ⟨line.toNat % 16, by omega⟩
  | 10 =>
    have eq : BitVec.ofNat 8 (16 * 10 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_3_10 ⟨line.toNat % 16, by omega⟩
  | 11 =>
    have eq : BitVec.ofNat 8 (16 * 11 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_3_11 ⟨line.toNat % 16, by omega⟩
  | 12 =>
    have eq : BitVec.ofNat 8 (16 * 12 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_3_12 ⟨line.toNat % 16, by omega⟩
  | 13 =>
    have eq : BitVec.ofNat 8 (16 * 13 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_3_13 ⟨line.toNat % 16, by omega⟩
  | 14 =>
    have eq : BitVec.ofNat 8 (16 * 14 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_3_14 ⟨line.toNat % 16, by omega⟩
  | 15 =>
    have eq : BitVec.ofNat 8 (16 * 15 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_3_15 ⟨line.toNat % 16, by omega⟩
  | q + 16 => omega

private theorem safe_4_0 : ∀ n : Fin 16, Safe ⟨4, by decide⟩ (BitVec.ofNat 8 (16 * 0 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_4_1 : ∀ n : Fin 16, Safe ⟨4, by decide⟩ (BitVec.ofNat 8 (16 * 1 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_4_2 : ∀ n : Fin 16, Safe ⟨4, by decide⟩ (BitVec.ofNat 8 (16 * 2 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_4_3 : ∀ n : Fin 16, Safe ⟨4, by decide⟩ (BitVec.ofNat 8 (16 * 3 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_4_4 : ∀ n : Fin 16, Safe ⟨4, by decide⟩ (BitVec.ofNat 8 (16 * 4 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_4_5 : ∀ n : Fin 16, Safe ⟨4, by decide⟩ (BitVec.ofNat 8 (16 * 5 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_4_6 : ∀ n : Fin 16, Safe ⟨4, by decide⟩ (BitVec.ofNat 8 (16 * 6 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_4_7 : ∀ n : Fin 16, Safe ⟨4, by decide⟩ (BitVec.ofNat 8 (16 * 7 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_4_8 : ∀ n : Fin 16, Safe ⟨4, by decide⟩ (BitVec.ofNat 8 (16 * 8 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_4_9 : ∀ n : Fin 16, Safe ⟨4, by decide⟩ (BitVec.ofNat 8 (16 * 9 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_4_10 : ∀ n : Fin 16, Safe ⟨4, by decide⟩ (BitVec.ofNat 8 (16 * 10 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_4_11 : ∀ n : Fin 16, Safe ⟨4, by decide⟩ (BitVec.ofNat 8 (16 * 11 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_4_12 : ∀ n : Fin 16, Safe ⟨4, by decide⟩ (BitVec.ofNat 8 (16 * 12 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_4_13 : ∀ n : Fin 16, Safe ⟨4, by decide⟩ (BitVec.ofNat 8 (16 * 13 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_4_14 : ∀ n : Fin 16, Safe ⟨4, by decide⟩ (BitVec.ofNat 8 (16 * 14 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_4_15 : ∀ n : Fin 16, Safe ⟨4, by decide⟩ (BitVec.ofNat 8 (16 * 15 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_4 (line : Byte) : Safe ⟨4, by decide⟩ line := by
  have bound := line.isLt
  generalize hq : line.toNat / 16 = q
  have small : q < 16 := by omega
  match q with
  | 0 =>
    have eq : BitVec.ofNat 8 (16 * 0 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_4_0 ⟨line.toNat % 16, by omega⟩
  | 1 =>
    have eq : BitVec.ofNat 8 (16 * 1 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_4_1 ⟨line.toNat % 16, by omega⟩
  | 2 =>
    have eq : BitVec.ofNat 8 (16 * 2 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_4_2 ⟨line.toNat % 16, by omega⟩
  | 3 =>
    have eq : BitVec.ofNat 8 (16 * 3 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_4_3 ⟨line.toNat % 16, by omega⟩
  | 4 =>
    have eq : BitVec.ofNat 8 (16 * 4 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_4_4 ⟨line.toNat % 16, by omega⟩
  | 5 =>
    have eq : BitVec.ofNat 8 (16 * 5 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_4_5 ⟨line.toNat % 16, by omega⟩
  | 6 =>
    have eq : BitVec.ofNat 8 (16 * 6 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_4_6 ⟨line.toNat % 16, by omega⟩
  | 7 =>
    have eq : BitVec.ofNat 8 (16 * 7 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_4_7 ⟨line.toNat % 16, by omega⟩
  | 8 =>
    have eq : BitVec.ofNat 8 (16 * 8 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_4_8 ⟨line.toNat % 16, by omega⟩
  | 9 =>
    have eq : BitVec.ofNat 8 (16 * 9 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_4_9 ⟨line.toNat % 16, by omega⟩
  | 10 =>
    have eq : BitVec.ofNat 8 (16 * 10 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_4_10 ⟨line.toNat % 16, by omega⟩
  | 11 =>
    have eq : BitVec.ofNat 8 (16 * 11 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_4_11 ⟨line.toNat % 16, by omega⟩
  | 12 =>
    have eq : BitVec.ofNat 8 (16 * 12 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_4_12 ⟨line.toNat % 16, by omega⟩
  | 13 =>
    have eq : BitVec.ofNat 8 (16 * 13 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_4_13 ⟨line.toNat % 16, by omega⟩
  | 14 =>
    have eq : BitVec.ofNat 8 (16 * 14 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_4_14 ⟨line.toNat % 16, by omega⟩
  | 15 =>
    have eq : BitVec.ofNat 8 (16 * 15 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_4_15 ⟨line.toNat % 16, by omega⟩
  | q + 16 => omega

private theorem safe_5_0 : ∀ n : Fin 16, Safe ⟨5, by decide⟩ (BitVec.ofNat 8 (16 * 0 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_5_1 : ∀ n : Fin 16, Safe ⟨5, by decide⟩ (BitVec.ofNat 8 (16 * 1 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_5_2 : ∀ n : Fin 16, Safe ⟨5, by decide⟩ (BitVec.ofNat 8 (16 * 2 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_5_3 : ∀ n : Fin 16, Safe ⟨5, by decide⟩ (BitVec.ofNat 8 (16 * 3 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_5_4 : ∀ n : Fin 16, Safe ⟨5, by decide⟩ (BitVec.ofNat 8 (16 * 4 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_5_5 : ∀ n : Fin 16, Safe ⟨5, by decide⟩ (BitVec.ofNat 8 (16 * 5 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_5_6 : ∀ n : Fin 16, Safe ⟨5, by decide⟩ (BitVec.ofNat 8 (16 * 6 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_5_7 : ∀ n : Fin 16, Safe ⟨5, by decide⟩ (BitVec.ofNat 8 (16 * 7 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_5_8 : ∀ n : Fin 16, Safe ⟨5, by decide⟩ (BitVec.ofNat 8 (16 * 8 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_5_9 : ∀ n : Fin 16, Safe ⟨5, by decide⟩ (BitVec.ofNat 8 (16 * 9 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_5_10 : ∀ n : Fin 16, Safe ⟨5, by decide⟩ (BitVec.ofNat 8 (16 * 10 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_5_11 : ∀ n : Fin 16, Safe ⟨5, by decide⟩ (BitVec.ofNat 8 (16 * 11 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_5_12 : ∀ n : Fin 16, Safe ⟨5, by decide⟩ (BitVec.ofNat 8 (16 * 12 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_5_13 : ∀ n : Fin 16, Safe ⟨5, by decide⟩ (BitVec.ofNat 8 (16 * 13 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_5_14 : ∀ n : Fin 16, Safe ⟨5, by decide⟩ (BitVec.ofNat 8 (16 * 14 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_5_15 : ∀ n : Fin 16, Safe ⟨5, by decide⟩ (BitVec.ofNat 8 (16 * 15 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_5 (line : Byte) : Safe ⟨5, by decide⟩ line := by
  have bound := line.isLt
  generalize hq : line.toNat / 16 = q
  have small : q < 16 := by omega
  match q with
  | 0 =>
    have eq : BitVec.ofNat 8 (16 * 0 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_5_0 ⟨line.toNat % 16, by omega⟩
  | 1 =>
    have eq : BitVec.ofNat 8 (16 * 1 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_5_1 ⟨line.toNat % 16, by omega⟩
  | 2 =>
    have eq : BitVec.ofNat 8 (16 * 2 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_5_2 ⟨line.toNat % 16, by omega⟩
  | 3 =>
    have eq : BitVec.ofNat 8 (16 * 3 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_5_3 ⟨line.toNat % 16, by omega⟩
  | 4 =>
    have eq : BitVec.ofNat 8 (16 * 4 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_5_4 ⟨line.toNat % 16, by omega⟩
  | 5 =>
    have eq : BitVec.ofNat 8 (16 * 5 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_5_5 ⟨line.toNat % 16, by omega⟩
  | 6 =>
    have eq : BitVec.ofNat 8 (16 * 6 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_5_6 ⟨line.toNat % 16, by omega⟩
  | 7 =>
    have eq : BitVec.ofNat 8 (16 * 7 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_5_7 ⟨line.toNat % 16, by omega⟩
  | 8 =>
    have eq : BitVec.ofNat 8 (16 * 8 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_5_8 ⟨line.toNat % 16, by omega⟩
  | 9 =>
    have eq : BitVec.ofNat 8 (16 * 9 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_5_9 ⟨line.toNat % 16, by omega⟩
  | 10 =>
    have eq : BitVec.ofNat 8 (16 * 10 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_5_10 ⟨line.toNat % 16, by omega⟩
  | 11 =>
    have eq : BitVec.ofNat 8 (16 * 11 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_5_11 ⟨line.toNat % 16, by omega⟩
  | 12 =>
    have eq : BitVec.ofNat 8 (16 * 12 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_5_12 ⟨line.toNat % 16, by omega⟩
  | 13 =>
    have eq : BitVec.ofNat 8 (16 * 13 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_5_13 ⟨line.toNat % 16, by omega⟩
  | 14 =>
    have eq : BitVec.ofNat 8 (16 * 14 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_5_14 ⟨line.toNat % 16, by omega⟩
  | 15 =>
    have eq : BitVec.ofNat 8 (16 * 15 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_5_15 ⟨line.toNat % 16, by omega⟩
  | q + 16 => omega

private theorem safe_6_0 : ∀ n : Fin 16, Safe ⟨6, by decide⟩ (BitVec.ofNat 8 (16 * 0 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_6_1 : ∀ n : Fin 16, Safe ⟨6, by decide⟩ (BitVec.ofNat 8 (16 * 1 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_6_2 : ∀ n : Fin 16, Safe ⟨6, by decide⟩ (BitVec.ofNat 8 (16 * 2 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_6_3 : ∀ n : Fin 16, Safe ⟨6, by decide⟩ (BitVec.ofNat 8 (16 * 3 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_6_4 : ∀ n : Fin 16, Safe ⟨6, by decide⟩ (BitVec.ofNat 8 (16 * 4 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_6_5 : ∀ n : Fin 16, Safe ⟨6, by decide⟩ (BitVec.ofNat 8 (16 * 5 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_6_6 : ∀ n : Fin 16, Safe ⟨6, by decide⟩ (BitVec.ofNat 8 (16 * 6 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_6_7 : ∀ n : Fin 16, Safe ⟨6, by decide⟩ (BitVec.ofNat 8 (16 * 7 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_6_8 : ∀ n : Fin 16, Safe ⟨6, by decide⟩ (BitVec.ofNat 8 (16 * 8 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_6_9 : ∀ n : Fin 16, Safe ⟨6, by decide⟩ (BitVec.ofNat 8 (16 * 9 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_6_10 : ∀ n : Fin 16, Safe ⟨6, by decide⟩ (BitVec.ofNat 8 (16 * 10 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_6_11 : ∀ n : Fin 16, Safe ⟨6, by decide⟩ (BitVec.ofNat 8 (16 * 11 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_6_12 : ∀ n : Fin 16, Safe ⟨6, by decide⟩ (BitVec.ofNat 8 (16 * 12 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_6_13 : ∀ n : Fin 16, Safe ⟨6, by decide⟩ (BitVec.ofNat 8 (16 * 13 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_6_14 : ∀ n : Fin 16, Safe ⟨6, by decide⟩ (BitVec.ofNat 8 (16 * 14 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_6_15 : ∀ n : Fin 16, Safe ⟨6, by decide⟩ (BitVec.ofNat 8 (16 * 15 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_6 (line : Byte) : Safe ⟨6, by decide⟩ line := by
  have bound := line.isLt
  generalize hq : line.toNat / 16 = q
  have small : q < 16 := by omega
  match q with
  | 0 =>
    have eq : BitVec.ofNat 8 (16 * 0 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_6_0 ⟨line.toNat % 16, by omega⟩
  | 1 =>
    have eq : BitVec.ofNat 8 (16 * 1 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_6_1 ⟨line.toNat % 16, by omega⟩
  | 2 =>
    have eq : BitVec.ofNat 8 (16 * 2 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_6_2 ⟨line.toNat % 16, by omega⟩
  | 3 =>
    have eq : BitVec.ofNat 8 (16 * 3 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_6_3 ⟨line.toNat % 16, by omega⟩
  | 4 =>
    have eq : BitVec.ofNat 8 (16 * 4 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_6_4 ⟨line.toNat % 16, by omega⟩
  | 5 =>
    have eq : BitVec.ofNat 8 (16 * 5 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_6_5 ⟨line.toNat % 16, by omega⟩
  | 6 =>
    have eq : BitVec.ofNat 8 (16 * 6 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_6_6 ⟨line.toNat % 16, by omega⟩
  | 7 =>
    have eq : BitVec.ofNat 8 (16 * 7 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_6_7 ⟨line.toNat % 16, by omega⟩
  | 8 =>
    have eq : BitVec.ofNat 8 (16 * 8 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_6_8 ⟨line.toNat % 16, by omega⟩
  | 9 =>
    have eq : BitVec.ofNat 8 (16 * 9 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_6_9 ⟨line.toNat % 16, by omega⟩
  | 10 =>
    have eq : BitVec.ofNat 8 (16 * 10 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_6_10 ⟨line.toNat % 16, by omega⟩
  | 11 =>
    have eq : BitVec.ofNat 8 (16 * 11 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_6_11 ⟨line.toNat % 16, by omega⟩
  | 12 =>
    have eq : BitVec.ofNat 8 (16 * 12 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_6_12 ⟨line.toNat % 16, by omega⟩
  | 13 =>
    have eq : BitVec.ofNat 8 (16 * 13 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_6_13 ⟨line.toNat % 16, by omega⟩
  | 14 =>
    have eq : BitVec.ofNat 8 (16 * 14 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_6_14 ⟨line.toNat % 16, by omega⟩
  | 15 =>
    have eq : BitVec.ofNat 8 (16 * 15 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_6_15 ⟨line.toNat % 16, by omega⟩
  | q + 16 => omega

private theorem safe_7_0 : ∀ n : Fin 16, Safe ⟨7, by decide⟩ (BitVec.ofNat 8 (16 * 0 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_7_1 : ∀ n : Fin 16, Safe ⟨7, by decide⟩ (BitVec.ofNat 8 (16 * 1 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_7_2 : ∀ n : Fin 16, Safe ⟨7, by decide⟩ (BitVec.ofNat 8 (16 * 2 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_7_3 : ∀ n : Fin 16, Safe ⟨7, by decide⟩ (BitVec.ofNat 8 (16 * 3 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_7_4 : ∀ n : Fin 16, Safe ⟨7, by decide⟩ (BitVec.ofNat 8 (16 * 4 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_7_5 : ∀ n : Fin 16, Safe ⟨7, by decide⟩ (BitVec.ofNat 8 (16 * 5 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_7_6 : ∀ n : Fin 16, Safe ⟨7, by decide⟩ (BitVec.ofNat 8 (16 * 6 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_7_7 : ∀ n : Fin 16, Safe ⟨7, by decide⟩ (BitVec.ofNat 8 (16 * 7 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_7_8 : ∀ n : Fin 16, Safe ⟨7, by decide⟩ (BitVec.ofNat 8 (16 * 8 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_7_9 : ∀ n : Fin 16, Safe ⟨7, by decide⟩ (BitVec.ofNat 8 (16 * 9 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_7_10 : ∀ n : Fin 16, Safe ⟨7, by decide⟩ (BitVec.ofNat 8 (16 * 10 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_7_11 : ∀ n : Fin 16, Safe ⟨7, by decide⟩ (BitVec.ofNat 8 (16 * 11 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_7_12 : ∀ n : Fin 16, Safe ⟨7, by decide⟩ (BitVec.ofNat 8 (16 * 12 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_7_13 : ∀ n : Fin 16, Safe ⟨7, by decide⟩ (BitVec.ofNat 8 (16 * 13 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_7_14 : ∀ n : Fin 16, Safe ⟨7, by decide⟩ (BitVec.ofNat 8 (16 * 14 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_7_15 : ∀ n : Fin 16, Safe ⟨7, by decide⟩ (BitVec.ofNat 8 (16 * 15 + n.val)) := by unfold Safe; decide +kernel

private theorem safe_7 (line : Byte) : Safe ⟨7, by decide⟩ line := by
  have bound := line.isLt
  generalize hq : line.toNat / 16 = q
  have small : q < 16 := by omega
  match q with
  | 0 =>
    have eq : BitVec.ofNat 8 (16 * 0 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_7_0 ⟨line.toNat % 16, by omega⟩
  | 1 =>
    have eq : BitVec.ofNat 8 (16 * 1 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_7_1 ⟨line.toNat % 16, by omega⟩
  | 2 =>
    have eq : BitVec.ofNat 8 (16 * 2 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_7_2 ⟨line.toNat % 16, by omega⟩
  | 3 =>
    have eq : BitVec.ofNat 8 (16 * 3 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_7_3 ⟨line.toNat % 16, by omega⟩
  | 4 =>
    have eq : BitVec.ofNat 8 (16 * 4 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_7_4 ⟨line.toNat % 16, by omega⟩
  | 5 =>
    have eq : BitVec.ofNat 8 (16 * 5 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_7_5 ⟨line.toNat % 16, by omega⟩
  | 6 =>
    have eq : BitVec.ofNat 8 (16 * 6 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_7_6 ⟨line.toNat % 16, by omega⟩
  | 7 =>
    have eq : BitVec.ofNat 8 (16 * 7 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_7_7 ⟨line.toNat % 16, by omega⟩
  | 8 =>
    have eq : BitVec.ofNat 8 (16 * 8 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_7_8 ⟨line.toNat % 16, by omega⟩
  | 9 =>
    have eq : BitVec.ofNat 8 (16 * 9 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_7_9 ⟨line.toNat % 16, by omega⟩
  | 10 =>
    have eq : BitVec.ofNat 8 (16 * 10 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_7_10 ⟨line.toNat % 16, by omega⟩
  | 11 =>
    have eq : BitVec.ofNat 8 (16 * 11 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_7_11 ⟨line.toNat % 16, by omega⟩
  | 12 =>
    have eq : BitVec.ofNat 8 (16 * 12 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_7_12 ⟨line.toNat % 16, by omega⟩
  | 13 =>
    have eq : BitVec.ofNat 8 (16 * 13 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_7_13 ⟨line.toNat % 16, by omega⟩
  | 14 =>
    have eq : BitVec.ofNat 8 (16 * 14 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_7_14 ⟨line.toNat % 16, by omega⟩
  | 15 =>
    have eq : BitVec.ofNat 8 (16 * 15 + line.toNat % 16) = line := by
      apply BitVec.eq_of_toNat_eq
      rw [BitVec.toNat_ofNat]
      omega
    simpa only [eq] using safe_7_15 ⟨line.toNat % 16, by omega⟩
  | q + 16 => omega

theorem all_safe (pos : Fin 8) (line : Byte) : Safe pos line := by
  rcases pos with ⟨p, hp⟩
  match p with
  | 0 => exact safe_0 line
  | 1 => exact safe_1 line
  | 2 => exact safe_2 line
  | 3 => exact safe_3 line
  | 4 => exact safe_4 line
  | 5 => exact safe_5 line
  | 6 => exact safe_6 line
  | 7 => exact safe_7 line
  | n + 8 => omega

end NeuralReversi.Solve1.LineSafety

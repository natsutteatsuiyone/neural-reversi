import Std

namespace NeuralReversi.Solve1.LineTable

abbrev Byte := BitVec 8
abbrev Usize := BitVec 64

def countFlip (pos : Fin 8) (line : Byte) : Byte :=
  let above : Usize := line.setWidth 64 >>> (pos.val + 1)
  let below : Usize := line.setWidth 64 &&& (((1 : Usize) <<< pos.val) - 1)
  (if above = 0 then 0 else 2 * above.ctz.setWidth 8) +
    (if below = 0 then 0 else
      2 * ((BitVec.ofNat 64 pos.val - 1 - (63 - below.clz)).setWidth 8))

def ray (line : Byte) (acc : Nat) : List Nat → Nat
  | [] => 0
  | i :: rest => if line.getLsbD i then acc else ray line (acc + 1) rest

def reference (pos : Fin 8) (line : Byte) : Nat :=
  ray line 0 ((List.range 8).filter (fun i => pos.val < i)) +
    ray line 0 ((List.range pos.val).reverse)

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
private theorem correct_0 (line : Byte) :
    (countFlip ⟨0, by decide⟩ line).toNat = 2 * reference ⟨0, by decide⟩ line := by
  have all : ∀ n : Fin 256,
      (countFlip ⟨0, by decide⟩ (BitVec.ofNat 8 n.val)).toNat =
        2 * reference ⟨0, by decide⟩ (BitVec.ofNat 8 n.val) := by decide +kernel
  simpa using all ⟨line.toNat, line.isLt⟩

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
private theorem correct_1 (line : Byte) :
    (countFlip ⟨1, by decide⟩ line).toNat = 2 * reference ⟨1, by decide⟩ line := by
  have all : ∀ n : Fin 256,
      (countFlip ⟨1, by decide⟩ (BitVec.ofNat 8 n.val)).toNat =
        2 * reference ⟨1, by decide⟩ (BitVec.ofNat 8 n.val) := by decide +kernel
  simpa using all ⟨line.toNat, line.isLt⟩

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
private theorem correct_2 (line : Byte) :
    (countFlip ⟨2, by decide⟩ line).toNat = 2 * reference ⟨2, by decide⟩ line := by
  have all : ∀ n : Fin 256,
      (countFlip ⟨2, by decide⟩ (BitVec.ofNat 8 n.val)).toNat =
        2 * reference ⟨2, by decide⟩ (BitVec.ofNat 8 n.val) := by decide +kernel
  simpa using all ⟨line.toNat, line.isLt⟩

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
private theorem correct_3 (line : Byte) :
    (countFlip ⟨3, by decide⟩ line).toNat = 2 * reference ⟨3, by decide⟩ line := by
  have all : ∀ n : Fin 256,
      (countFlip ⟨3, by decide⟩ (BitVec.ofNat 8 n.val)).toNat =
        2 * reference ⟨3, by decide⟩ (BitVec.ofNat 8 n.val) := by decide +kernel
  simpa using all ⟨line.toNat, line.isLt⟩

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
private theorem correct_4 (line : Byte) :
    (countFlip ⟨4, by decide⟩ line).toNat = 2 * reference ⟨4, by decide⟩ line := by
  have all : ∀ n : Fin 256,
      (countFlip ⟨4, by decide⟩ (BitVec.ofNat 8 n.val)).toNat =
        2 * reference ⟨4, by decide⟩ (BitVec.ofNat 8 n.val) := by decide +kernel
  simpa using all ⟨line.toNat, line.isLt⟩

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
private theorem correct_5 (line : Byte) :
    (countFlip ⟨5, by decide⟩ line).toNat = 2 * reference ⟨5, by decide⟩ line := by
  have all : ∀ n : Fin 256,
      (countFlip ⟨5, by decide⟩ (BitVec.ofNat 8 n.val)).toNat =
        2 * reference ⟨5, by decide⟩ (BitVec.ofNat 8 n.val) := by decide +kernel
  simpa using all ⟨line.toNat, line.isLt⟩

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
private theorem correct_6 (line : Byte) :
    (countFlip ⟨6, by decide⟩ line).toNat = 2 * reference ⟨6, by decide⟩ line := by
  have all : ∀ n : Fin 256,
      (countFlip ⟨6, by decide⟩ (BitVec.ofNat 8 n.val)).toNat =
        2 * reference ⟨6, by decide⟩ (BitVec.ofNat 8 n.val) := by decide +kernel
  simpa using all ⟨line.toNat, line.isLt⟩

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
private theorem correct_7 (line : Byte) :
    (countFlip ⟨7, by decide⟩ line).toNat = 2 * reference ⟨7, by decide⟩ line := by
  have all : ∀ n : Fin 256,
      (countFlip ⟨7, by decide⟩ (BitVec.ofNat 8 n.val)).toNat =
        2 * reference ⟨7, by decide⟩ (BitVec.ofNat 8 n.val) := by decide +kernel
  simpa using all ⟨line.toNat, line.isLt⟩

theorem countFlip_correct (pos : Fin 8) (line : Byte) :
    (countFlip pos line).toNat = 2 * reference pos line := by
  rcases pos with ⟨p, hp⟩
  match p with
  | 0 => exact correct_0 line
  | 1 => exact correct_1 line
  | 2 => exact correct_2 line
  | 3 => exact correct_3 line
  | 4 => exact correct_4 line
  | 5 => exact correct_5 line
  | 6 => exact correct_6 line
  | 7 => exact correct_7 line
  | n + 8 => omega

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
theorem reference_bound (pos : Fin 8) (line : Byte) : reference pos line ≤ 6 := by
  have all : ∀ p : Fin 8, ∀ n : Fin 256,
      reference p (BitVec.ofNat 8 n.val) ≤ 6 := by decide +kernel
  simpa using all pos ⟨line.toNat, line.isLt⟩

theorem countFlip_bound (pos : Fin 8) (line : Byte) :
    (countFlip pos line).toNat ≤ 12 := by
  rw [countFlip_correct]
  have := reference_bound pos line
  omega

theorem countFlip_even (pos : Fin 8) (line : Byte) :
    (countFlip pos line).toNat % 2 = 0 := by
  rw [countFlip_correct]
  omega

theorem countFlip_eq_ofNat (pos : Fin 8) (line : Byte) :
    countFlip pos line = BitVec.ofNat 8 (2 * reference pos line) := by
  apply BitVec.eq_of_toNat_eq
  rw [countFlip_correct, BitVec.toNat_ofNat]
  have := reference_bound pos line
  omega

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
theorem reference_toggle_square (pos : Fin 8) (line : Byte) :
    reference pos (line ^^^ ((1 : Byte) <<< pos.val)) = reference pos line := by
  have all : ∀ p : Fin 8, ∀ n : Fin 256,
      reference p (BitVec.ofNat 8 n.val ^^^ ((1 : Byte) <<< p.val)) =
        reference p (BitVec.ofNat 8 n.val) := by decide +kernel
  simpa using all pos ⟨line.toNat, line.isLt⟩

theorem countFlip_toggle_square (pos : Fin 8) (line : Byte) :
    countFlip pos (line ^^^ ((1 : Byte) <<< pos.val)) = countFlip pos line := by
  simp only [countFlip_eq_ofNat, reference_toggle_square]

end NeuralReversi.Solve1.LineTable

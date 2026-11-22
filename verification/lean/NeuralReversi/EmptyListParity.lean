import NeuralReversi.Bitboard.Model

namespace NeuralReversi.EmptyListParity

abbrev Square := Fin 64
abbrev Byte := BitVec 8

def quadrantId (sq : Square) : Byte :=
  let ids : Vector Byte 64 := #v[
    1, 1, 1, 1, 2, 2, 2, 2,
    1, 1, 1, 1, 2, 2, 2, 2,
    1, 1, 1, 1, 2, 2, 2, 2,
    1, 1, 1, 1, 2, 2, 2, 2,
    4, 4, 4, 4, 8, 8, 8, 8,
    4, 4, 4, 4, 8, 8, 8, 8,
    4, 4, 4, 4, 8, 8, 8, 8,
    4, 4, 4, 4, 8, 8, 8, 8]
  ids[sq]

def presorted : List Square := [
    0, 7, 56, 63, 2, 5, 16, 23,
    40, 47, 58, 61, 18, 21, 42, 45,
    3, 4, 24, 31, 32, 39, 59, 60,
    19, 20, 26, 29, 34, 37, 43, 44,
    11, 12, 25, 30, 33, 38, 51, 52,
    10, 13, 17, 22, 41, 46, 50, 53,
    1, 6, 8, 15, 48, 55, 57, 62,
    9, 14, 49, 54, 27, 28, 35, 36]

theorem scan_perm : presorted.Perm (List.finRange 64) := by
  apply List.perm_iff_count.mpr
  decide

private def quadrant (sq : Square) : Nat := 2 * (sq.val / 8 / 4) + sq.val % 8 / 4

private theorem id_spec : ∀ (sq : Square), (quadrantId sq).toNat < 16 ∧
    ∀ q : Fin 4, (quadrantId sq).getLsbD q.val = decide (quadrant sq = q.val) := by decide

def toggle (byte : Byte) (sq : Square) : Byte := byte ^^^ quadrantId sq

def fromEmpties (empties : Bitboard.Bits) : Byte :=
  (presorted.filter (fun sq => empties.getLsbD sq.val)).foldl toggle 0#8

inductive Reachable : List Square → Byte → Prop where
  | initial (empties : Bitboard.Bits) :
      Reachable ((List.finRange 64).filter (fun sq => empties.getLsbD sq.val)) (fromEmpties empties)
  | remove {cells byte} (sq : Square) : Reachable cells byte → sq ∈ cells →
      Reachable (cells.erase sq) (toggle byte sq)
  | restore {cells byte} (sq : Square) : Reachable cells byte → sq ∉ cells →
      Reachable (sq :: cells) (toggle byte sq)

private def xorIds : List Square → Byte
  | [] => 0
  | sq :: rest => quadrantId sq ^^^ xorIds rest

private theorem scan_xor (cells : List Square) (byte : Byte) :
    cells.foldl toggle byte = byte ^^^ xorIds cells := by
  induction cells generalizing byte with
  | nil => simp [xorIds]
  | cons sq rest ih => simp [ih, toggle, xorIds, BitVec.xor_assoc]

private theorem perm_xor {a b : List Square} (h : a.Perm b) : xorIds a = xorIds b := by
  induction h with
  | nil => rfl
  | cons _ _ ih => simp [xorIds, ih]
  | swap => simp only [xorIds]; ac_rfl
  | trans _ _ ih₁ ih₂ => exact ih₁.trans ih₂

private theorem reachable_xor {cells byte} (h : Reachable cells byte) : byte = xorIds cells := by
  induction h with
  | initial empties =>
    simp only [fromEmpties, scan_xor, BitVec.zero_xor]
    exact perm_xor (scan_perm.filter _)
  | remove sq _ present ih =>
    rw [ih, perm_xor (List.perm_cons_erase present)]
    simp only [toggle, xorIds]
    rw [BitVec.xor_comm _ (quadrantId sq), ← BitVec.xor_assoc, BitVec.xor_self, BitVec.zero_xor]
  | restore sq _ _ ih => simp [toggle, xorIds, ih, BitVec.xor_comm]

theorem parity_of_perm {cells order : List Square} {byte : Byte}
    (state : Reachable cells byte) (same : cells.Perm order) :
    byte = order.foldl toggle 0#8 := by
  rw [reachable_xor state, perm_xor same, scan_xor, BitVec.zero_xor]

def ParityContract : Prop :=
  ∀ cells byte, Reachable cells byte → byte.toNat < 16 ∧
    ∀ q : Fin 4, byte.getLsbD q.val = decide (cells.countP (fun sq => quadrant sq == q.val) % 2 = 1)

theorem parityContract_proven : ParityContract := by
  intro cells byte h
  rw [reachable_xor h]
  clear h byte
  induction cells with
  | nil => simp [xorIds]
  | cons sq rest ih =>
    refine ⟨Nat.xor_lt_two_pow (n := 4) (id_spec sq).1 ih.1, ?_⟩
    intro q
    simp only [xorIds, BitVec.getLsbD_xor, (id_spec sq).2 q, ih.2 q, List.countP_cons,
      beq_iff_eq]
    by_cases same : quadrant sq = q.val
    · simp [same]
      apply Bool.eq_iff_iff.mpr
      simp only [decide_eq_true_eq, Bool.not_eq_true', decide_eq_false_iff_not]
      omega
    · simp [same]

end NeuralReversi.EmptyListParity

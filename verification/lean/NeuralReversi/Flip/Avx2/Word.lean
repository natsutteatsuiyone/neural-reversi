import NeuralReversi.Flip.Avx2.Spec
import NeuralReversi.Flip.ScalarProperties

namespace NeuralReversi.Flip.Avx2
open NeuralReversi.Flip.Scalar

set_option maxRecDepth 8192
set_option maxHeartbeats 2000000

theorem blsmsk_bit (x : Word) (i : Nat) :
    ((x - 1#64) ^^^ x).getLsbD i = true ↔
      i < 64 ∧ ∀ k, k < i → x.getLsbD k = false := by
  rw [← BitVec.not_neg]
  simp only [BitVec.getLsbD_xor, BitVec.getLsbD_not, BitVec.getLsbD_neg]
  by_cases hi : i < 64
  · simp only [hi, decide_true, Bool.true_and, true_and]
    cases hx : x.getLsbD i <;>
      simp only [Bool.false_xor, Bool.true_xor, Bool.xor_false,
        Bool.xor_true, Bool.not_not, Bool.not_eq_true',
        decide_eq_false_iff_not, not_exists, not_and, Bool.not_eq_true]
  · have hx : x.getLsbD i = false := BitVec.getLsbD_of_ge x i (by omega)
    simp [hi, hx]

def Prefix (m opponent : Word) (i : Nat) : Prop :=
  m.getLsbD i = true ∧ ∀ k, k < i → (~~~opponent &&& m).getLsbD k = false

def leftPrefixWord (m opponent : Word) : Word :=
  let x := ~~~opponent &&& m
  ((x - 1#64) ^^^ x) &&& m

theorem left_prefix_bit (m opponent : Word) (i : Nat) :
    (leftPrefixWord m opponent).getLsbD i = true ↔ Prefix m opponent i := by
  simp only [leftPrefixWord, BitVec.getLsbD_and, Bool.and_eq_true, blsmsk_bit, Prefix]
  constructor
  · exact fun h => ⟨h.2, h.1.2⟩
  · exact fun h => ⟨⟨BitVec.lt_of_getLsbD h.1, h.2⟩, h.1⟩

theorem andnot_ne_self (player lo : Word) :
    ~~~player &&& lo ≠ lo ↔ ∃ j, lo.getLsbD j = true ∧ player.getLsbD j = true := by
  constructor
  · intro hn
    apply Classical.byContradiction
    intro hnone
    apply hn
    apply BitVec.eq_of_getLsbD_eq
    intro i hi
    have nh : ¬ (lo.getLsbD i = true ∧ player.getLsbD i = true) :=
      fun h => hnone ⟨i, h⟩
    simp only [BitVec.getLsbD_and, BitVec.getLsbD_not, hi, decide_true, Bool.true_and]
    cases hl : lo.getLsbD i <;> cases hp : player.getLsbD i <;> simp_all
  · rintro ⟨j, hl, hp⟩ heq
    have hj := BitVec.lt_of_getLsbD hl
    have hb := congrArg (fun w : Word => w.getLsbD j) heq
    simp only [BitVec.getLsbD_and, BitVec.getLsbD_not, hj, decide_true,
      Bool.true_and, hl, hp, Bool.not_true, Bool.false_and] at hb
    cases hb

def eraseWithoutAnchor (player lo : Word) : Word :=
  let lf := ~~~player &&& lo
  if lf = lo then 0 else lf

theorem erased_bit (player lo : Word) (i : Nat) :
    (eraseWithoutAnchor player lo).getLsbD i = true ↔
      lo.getLsbD i = true ∧ player.getLsbD i = false ∧
        ∃ j, lo.getLsbD j = true ∧ player.getLsbD j = true := by
  by_cases heq : ~~~player &&& lo = lo
  · have hn : ¬ ∃ j, lo.getLsbD j = true ∧ player.getLsbD j = true := by
      intro h
      exact ((andnot_ne_self player lo).mpr h) heq
    simp [eraseWithoutAnchor, heq, hn]
  · have hex := (andnot_ne_self player lo).mp heq
    simp only [eraseWithoutAnchor, heq, ↓reduceIte, BitVec.getLsbD_and,
      BitVec.getLsbD_not, Bool.and_eq_true, Bool.not_eq_true', decide_eq_true_eq,
      hex, and_true]
    constructor
    · exact fun h => ⟨h.2, h.1.2⟩
    · exact fun h => ⟨⟨BitVec.lt_of_getLsbD h.1, h.2⟩, h.1⟩

def leftWord (m player opponent : Word) : Word :=
  eraseWithoutAnchor player (leftPrefixWord m opponent)

theorem left_raw (m player opponent : Word) (i : Nat) :
    (leftWord m player opponent).getLsbD i = true ↔
      Prefix m opponent i ∧ player.getLsbD i = false ∧
        ∃ j, Prefix m opponent j ∧ player.getLsbD j = true := by
  simp only [leftWord, erased_bit, left_prefix_bit]

theorem nat_lt_of_highest_difference (a b j : Nat)
    (higher : ∀ k, j < k → a.testBit k = b.testBit k)
    (ha : a.testBit j = false) (hb : b.testBit j = true) : a < b := by
  have hquot : a / 2 ^ j / 2 = b / 2 ^ j / 2 := by
    apply Nat.eq_of_testBit_eq
    intro k
    simp only [Nat.testBit_div_two, Nat.testBit_div_two_pow]
    exact higher (k + 1 + j) (by omega)
  have hamod : a / 2 ^ j % 2 = 0 := by
    rw [Nat.testBit_eq_decide_div_mod_eq] at ha
    simp only [decide_eq_false_iff_not] at ha
    omega
  have hbmod : b / 2 ^ j % 2 = 1 := by
    rw [Nat.testBit_eq_decide_div_mod_eq] at hb
    exact of_decide_eq_true hb
  apply Nat.lt_of_div_lt_div (c := 2 ^ j)
  omega

theorem xor_comparison_at_greatest (x player : Word) (j : Nat)
    (hj : Greatest x j) :
    (x ^^^ player).toNat < player.toNat ↔ player.getLsbD j = true := by
  have higher : ∀ k, j < k → (x ^^^ player).toNat.testBit k = player.toNat.testBit k := by
    intro k hk
    change (x ^^^ player).getLsbD k = player.getLsbD k
    simp only [BitVec.getLsbD_xor, hj.2 k hk, Bool.false_xor]
  have atj : (x ^^^ player).getLsbD j = !player.getLsbD j := by
    simp only [BitVec.getLsbD_xor, hj.1, Bool.true_xor]
  cases hp : player.getLsbD j with
  | true =>
    have hlt := nat_lt_of_highest_difference (x ^^^ player).toNat player.toNat j higher
      (by change (x ^^^ player).getLsbD j = false; rw [atj, hp]; rfl) hp
    exact ⟨fun _ => rfl, fun _ => hlt⟩
  | false =>
    have hlt := nat_lt_of_highest_difference player.toNat (x ^^^ player).toNat j
      (fun k hk => (higher k hk).symm) hp
      (by change (x ^^^ player).getLsbD j = true; rw [atj, hp]; rfl)
    simp only [Bool.false_eq_true, iff_false]
    omega

theorem xor_unsigned_comparison (x player : Word) :
    (x ^^^ player).toNat < player.toNat ↔
      ∃ j, Greatest x j ∧ player.getLsbD j = true := by
  by_cases hx : x = 0
  · simp [hx, Greatest]
  · have hj := greatest_clz x hx
    rw [greatest_exists_iff x _ hj]
    exact xor_comparison_at_greatest x player _ hj

theorem xor_signed_comparison (x player : Word)
    (hx : x.msb = false) (hp : player.msb = false) :
    (x ^^^ player).toInt < player.toInt ↔
      ∃ j, Greatest x j ∧ player.getLsbD j = true := by
  have hxp : (x ^^^ player).msb = false := by simp [BitVec.msb_xor, hx, hp]
  rw [BitVec.toInt_eq_toNat_of_msb hxp, BitVec.toInt_eq_toNat_of_msb hp]
  exact (Int.ofNat_lt).trans (xor_unsigned_comparison x player)

def shadowFill (x : Word) (d : Nat) : Nat → Word
  | 0 => x
  | n + 1 =>
    let prev := shadowFill x d n
    prev ||| (prev >>> (2 ^ n * d))

theorem shadowFill_bit (x : Word) (d n i : Nat) :
    (shadowFill x d n).getLsbD i = true ↔
      ∃ k, k < 2 ^ n ∧ x.getLsbD (i + k * d) = true := by
  induction n generalizing i with
  | zero => simp [shadowFill]
  | succ n ih =>
    simp only [shadowFill, BitVec.getLsbD_or, Bool.or_eq_true,
      BitVec.getLsbD_ushiftRight, ih]
    have hpow : 2 ^ (n + 1) = 2 ^ n + 2 ^ n := by simp [Nat.pow_succ, Nat.mul_two]
    rw [hpow]
    constructor
    · rintro (⟨k, hk, hb⟩ | ⟨k, hk, hb⟩)
      · exact ⟨k, by omega, hb⟩
      · refine ⟨2 ^ n + k, by omega, ?_⟩
        simpa only [Nat.add_mul, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using hb
    · rintro ⟨k, hk, hb⟩
      by_cases hlo : k < 2 ^ n
      · exact .inl ⟨k, hlo, hb⟩
      · apply Or.inr
        refine ⟨k - 2 ^ n, by omega, ?_⟩
        have heq : k = 2 ^ n + (k - 2 ^ n) := by omega
        rw [heq] at hb
        simpa only [Nat.add_mul, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using hb

theorem shadow8_bit (x : Word) (d i : Nat) :
    (shadowFill x d 3).getLsbD i = true ↔
      ∃ k, k < 8 ∧ x.getLsbD (i + k * d) = true := by
  exact shadowFill_bit x d 3 i

theorem add_allOnes_eq_sub_one (x : Word) :
    x + BitVec.allOnes 64 = x - 1#64 := by
  rw [← BitVec.neg_one_eq_allOnes, BitVec.sub_eq_add_neg]

theorem cmpeq_erase (player lo : Word) :
    let lf := ~~~player &&& lo
    ~~~(if lf = lo then BitVec.allOnes 64 else 0) &&& lf =
      eraseWithoutAnchor player lo := by
  by_cases h : ~~~player &&& lo = lo
  · simp only [eraseWithoutAnchor, h, ↓reduceIte]
    rw [BitVec.not_allOnes, BitVec.zero_and]
    rfl
  · simp only [eraseWithoutAnchor, h, ↓reduceIte]
    change ~~~(0#64) &&& (~~~player &&& lo) = ~~~player &&& lo
    rw [BitVec.not_zero, BitVec.allOnes_and]

theorem cmpgt_gate (rs m rp re : Word) :
    ~~~rs &&& (m &&& (if re.toInt < rp.toInt then BitVec.allOnes 64 else 0)) =
      if re.toInt < rp.toInt then m &&& ~~~rs else 0 := by
  by_cases h : re.toInt < rp.toInt <;> simp only [h, ↓reduceIte]
  · rw [BitVec.and_allOnes, BitVec.and_comm]
  · simp

theorem shadow_three_stages (x : Word) (d : Nat) :
    shadowFill x d 3 =
      let first := x ||| (x >>> d)
      let second := first ||| (first >>> (2 * d))
      second ||| (second >>> (4 * d)) := by
  simp [shadowFill]


end NeuralReversi.Flip.Avx2

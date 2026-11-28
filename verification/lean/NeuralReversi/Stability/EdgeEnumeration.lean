module
public import NeuralReversi.Stability.EdgeModel

@[expose] public section

namespace NeuralReversi.Stability.Edge

def sourceEmptyOrder : Nat → Byte → List Nat
  | 0, _ => []
  | fuel + 1, bits => if bits = 0 then [] else
      bits.ctz.toNat :: sourceEmptyOrder fuel (bits &&& (bits - 1))

def emptyLoopStops : Nat → Byte → Bool
  | 0, bits => bits == 0
  | fuel + 1, bits => if bits == 0 then true else emptyLoopStops fuel (bits &&& (bits - 1))

def sourceSubsets (m p : Byte) : Nat → List Byte
  | 0 => []
  | fuel + 1 =>
      let next := (p - m) &&& m
      p :: if next == 0 then [] else sourceSubsets m next fuel

def subsetLoopStops (m p : Byte) : Nat → Bool
  | 0 => false
  | fuel + 1 =>
      let next := (p - m) &&& m
      if next == 0 then true else subsetLoopStops m next fuel

private def fastCtz8Aux (b : Byte) : Nat → Nat
  | 0 => if b.getLsbD 7 then 7 else 8
  | n + 1 => if b.getLsbD (7 - (n + 1)) then 7 - (n + 1) else fastCtz8Aux b n

private def fastCtz8 (b : Byte) : Nat := fastCtz8Aux b 7

private theorem ctzAuxRec_eq_fast (b : Byte) :
    ∀ n : Nat, n < 8 →
      (b.reverse.clzAuxRec n).toNat = fastCtz8Aux b n := by
  intro n hn
  induction n with
  | zero =>
    rw [BitVec.clzAuxRec.eq_1, fastCtz8Aux, BitVec.getLsbD_reverse, BitVec.getMsbD]
    cases b.getLsbD 7 <;> decide
  | succ n ih =>
    have hbit : b.reverse.getLsbD (n + 1) = b.getLsbD (7 - (n + 1)) := by
      simp only [BitVec.getLsbD_reverse, BitVec.getMsbD]
      have h1 : n + 1 < 8 := by omega
      simp [h1]
    rw [BitVec.clzAuxRec.eq_2, fastCtz8Aux, hbit]
    cases b.getLsbD (7 - (n + 1)) with
    | true =>
      simp only [↓reduceIte, BitVec.toNat_ofNat]
      omega
    | false =>
      simp only [Bool.false_eq_true, ↓reduceIte]
      exact ih (by omega)

private theorem ctz_toNat_eq_fast (b : Byte) : b.ctz.toNat = fastCtz8 b :=
  ctzAuxRec_eq_fast b 7 (by omega)

private def fastEmptyOrder : Nat → Byte → List Nat
  | 0, _ => []
  | fuel + 1, bits => if bits = 0 then [] else
      fastCtz8 bits :: fastEmptyOrder fuel (bits &&& (bits - 1))

private theorem sourceEmptyOrder_eq_fast (fuel : Nat) (bits : Byte) :
    sourceEmptyOrder fuel bits = fastEmptyOrder fuel bits := by
  induction fuel generalizing bits with
  | zero => rfl
  | succ fuel ih =>
    simp only [sourceEmptyOrder, fastEmptyOrder]
    split
    · rfl
    · rw [ctz_toNat_eq_fast, ih]

private theorem empty_order_finite : ∀ n : Fin 256,
    fastEmptyOrder 8 (BitVec.ofNat 8 n.val) =
      (List.range 8).filter (fun x => (BitVec.ofNat 8 n.val).getLsbD x) ∧
    emptyLoopStops 8 (BitVec.ofNat 8 n.val) = true := by decide +kernel

theorem source_empty_order (bits : Byte) :
    sourceEmptyOrder 8 bits = (List.range 8).filter bits.getLsbD ∧
    emptyLoopStops 8 bits = true := by
  rw [sourceEmptyOrder_eq_fast]
  simpa only [BitVec.ofNat_toNat, BitVec.setWidth_eq] using
    empty_order_finite ⟨bits.toNat, bits.isLt⟩

private theorem bv_bit (x : Byte) (i : Nat) : x.getLsbD i = x.toNat.testBit i := rfl

private theorem nat_le_of_testBit_imp :
    ∀ (n a b : Nat), a < 2 ^ n → b < 2 ^ n →
      (∀ i, i < n → a.testBit i = true → b.testBit i = true) → a ≤ b := by
  intro n
  induction n with
  | zero => intro a b ha hb _; omega
  | succ n ih =>
    intro a b ha hb himp
    have hdiv := ih (a / 2) (b / 2) (by omega) (by omega) (fun i hi hbit => by
      have := himp (i + 1) (by omega) (by simpa [Nat.testBit_succ] using hbit)
      simpa [Nat.testBit_succ] using this)
    have h0 : a % 2 ≤ b % 2 := by
      have ha2 : a % 2 = 0 ∨ a % 2 = 1 := by omega
      rcases ha2 with ha0 | ha1
      · omega
      · have ht : a.testBit 0 = true := by simp [Nat.testBit_zero, ha1]
        have hbt := himp 0 (by omega) ht
        simp only [Nat.testBit_zero, decide_eq_true_eq] at hbt
        omega
    omega

private theorem byte_le_of_sub (x m : Byte) (h : x &&& m = x) : x.toNat ≤ m.toNat := by
  apply nat_le_of_testBit_imp 8 x.toNat m.toNat x.isLt m.isLt
  intro i hi hx
  have h1 := congrArg (fun v : Byte => v.getLsbD i) h
  rw [BitVec.getLsbD_and, bv_bit, bv_bit] at h1
  rw [hx] at h1
  simpa using h1

private theorem nat_add_eq_or : ∀ (a b : Nat), a &&& b = 0 → a + b = a ||| b := by
  intro a
  induction a using Nat.strongRecOn with
  | ind a ih =>
    intro b hdisj
    by_cases ha : a = 0
    · subst ha; simp
    · by_cases hb : b = 0
      · subst hb; simp
      · have hmod : (a % 2) &&& (b % 2) = 0 := by
          have h := congrArg (· % 2) hdisj
          simpa [Nat.and_mod_two_pow (n := 1)] using h
        have hdiv : (a / 2) &&& (b / 2) = 0 := by
          have h := congrArg (· / 2) hdisj
          simpa [Nat.and_div_two_pow (n := 1)] using h
        have hrec := ih (a / 2) (by omega) (b / 2) hdiv
        have hor_div : (a ||| b) / 2 = (a / 2) ||| (b / 2) := by
          simpa using Nat.or_div_two_pow (a := a) (b := b) (n := 1)
        have hor_mod : (a ||| b) % 2 = (a % 2) ||| (b % 2) := by
          simpa using Nat.or_mod_two_pow (a := a) (b := b) (n := 1)
        have hbit : a % 2 + b % 2 = (a % 2) ||| (b % 2) := by
          have ha2 : a % 2 = 0 ∨ a % 2 = 1 := by omega
          have hb2 : b % 2 = 0 ∨ b % 2 = 1 := by omega
          rcases ha2 with ha0 | ha1 <;>
          rcases hb2 with hb0 | hb1 <;>
          simp_all
        omega

private theorem sub_eq_or_not_add_one (p m : Byte) (hsub : p &&& m = p) :
    p - m = (p ||| ~~~m) + 1 := by
  have hdisj : p &&& ~~~m = 0 := by
    apply BitVec.eq_of_getLsbD_eq
    intro i hi
    have h1 := congrArg (fun v : Byte => v.getLsbD i) hsub
    rw [BitVec.getLsbD_and] at h1
    rw [BitVec.getLsbD_and, BitVec.getLsbD_not, show (0 : Byte).getLsbD i = false from BitVec.getLsbD_zero]
    simp only [hi, decide_true, Bool.true_and]
    cases hp : p.getLsbD i <;> cases hm : m.getLsbD i <;> simp_all
  have hadd : p + ~~~m = p ||| ~~~m := by
    apply BitVec.eq_of_toNat_eq
    have h0 : p.toNat &&& (~~~m).toNat = 0 := by
      have ht := congrArg BitVec.toNat hdisj
      simpa [BitVec.toNat_and] using ht
    have hor := nat_add_eq_or p.toNat (~~~m).toNat h0
    have hlt := (p ||| ~~~m).isLt
    rw [BitVec.toNat_add, BitVec.toNat_or] at *
    rw [hor, Nat.mod_eq_of_lt hlt]
  have hneg : p - m = p + (~~~m + 1) := by
    rw [BitVec.sub_eq_add_neg, BitVec.neg_eq_not_add]; rfl
  rw [hneg, ← BitVec.add_assoc, hadd]

private theorem exists_lowest_set_bit (x : Byte) (hx : x ≠ 0) :
    ∃ k, k < 8 ∧ x.getLsbD k = true ∧ ∀ j, j < k → x.getLsbD j = false := by
  have aux : ∀ n : Nat, n ≤ 8 → (∀ j, j < n → x.getLsbD j = false) →
      ∃ k, k < 8 ∧ x.getLsbD k = true ∧ ∀ j, j < k → x.getLsbD j = false := by
    intro n hn
    induction hrem : 8 - n generalizing n with
    | zero =>
      intro hall
      exfalso; apply hx
      apply BitVec.eq_of_getLsbD_eq
      intro i hi
      simp [hall i (by omega)]
    | succ rem ih =>
      intro hall
      cases hb : x.getLsbD n with
      | true => exact ⟨n, by omega, hb, hall⟩
      | false =>
        exact ih (n + 1) (by omega) (by omega) (fun j hj => by
          by_cases heq : j = n
          · subst heq; exact hb
          · exact hall j (by omega))
  exact aux 0 (by omega) (fun _ hj => by omega)

private theorem nat_mod_pow2_of_all_ones :
    ∀ (k a : Nat), (∀ j, j < k → a.testBit j = true) → a % 2 ^ k = 2 ^ k - 1 := by
  intro k
  induction k with
  | zero => intro a _; omega
  | succ k ih =>
    intro a hall
    have h0 := hall 0 (by omega)
    simp only [Nat.testBit_zero, decide_eq_true_eq] at h0
    have hdiv := ih (a / 2) (fun j hj => by
      have := hall (j + 1) (by omega)
      simpa [Nat.testBit_succ] using this)
    have hpow : 2 ^ (k + 1) = 2 * 2 ^ k := by rw [Nat.pow_succ, Nat.mul_comm]
    have hmod : a % (2 * 2 ^ k) = 2 * ((a / 2) % 2 ^ k) + a % 2 := by
      have h1 := Nat.div_add_mod a 2
      have h2 := Nat.div_add_mod (a / 2) (2 ^ k)
      have h3 := Nat.div_add_mod a (2 * 2 ^ k)
      have h4 : a / (2 * 2 ^ k) = (a / 2) / 2 ^ k := by rw [← Nat.div_div_eq_div_mul]
      have hlt1 : 2 * ((a / 2) % 2 ^ k) + a % 2 < 2 * 2 ^ k := by
        have := Nat.mod_lt (a / 2) (Nat.two_pow_pos k); omega
      have hlt2 := Nat.mod_lt a (Nat.mul_pos (by decide : 0 < 2) (Nat.two_pow_pos k))
      rw [h4] at h3
      have hmul : 2 * 2 ^ k * (a / 2 / 2 ^ k) = 2 * (2 ^ k * (a / 2 / 2 ^ k)) := Nat.mul_assoc 2 _ _
      omega
    have hpos : 1 ≤ 2 ^ k := Nat.two_pow_pos k
    rw [hpow, hmod, hdiv, h0]
    omega

private theorem nat_testBit_add_low_ones (x k : Nat) (hk : k < 8) (hx : x < 256)
    (hlo : ∀ j, j < k → x.testBit j = true) (hzk : x.testBit k = false) :
    x + 1 < 256 ∧
    ∀ j, j < 8 → (x + 1).testBit j = if j < k then false else if j = k then true else x.testBit j := by
  have hmod : x % 2 ^ k = 2 ^ k - 1 := nat_mod_pow2_of_all_ones k x hlo
  have hkbit : (x / 2 ^ k) % 2 = 0 := by
    rw [Nat.testBit_eq_decide_div_mod_eq, decide_eq_false_iff_not] at hzk
    omega
  have hpow1 : 2 ^ (k + 1) = 2 ^ k * 2 := Nat.pow_succ 2 k
  have hpow_le : 2 ^ (k + 1) ≤ 256 := by
    have : ∀ m : Fin 8, 2 ^ (m.val + 1) ≤ 256 := by decide +kernel
    exact this ⟨k, hk⟩
  have hpos : 1 ≤ 2 ^ k := Nat.two_pow_pos k
  have hdiv_eq : x / 2 ^ k = 2 * (x / 2 ^ (k + 1)) := by
    have hd : x / 2 ^ (k + 1) = (x / 2 ^ k) / 2 := by rw [hpow1, ← Nat.div_div_eq_div_mul]
    omega
  have hx_decomp : x = 2 ^ (k + 1) * (x / 2 ^ (k + 1)) + (2 ^ k - 1) := by
    have h1 := Nat.div_add_mod x (2 ^ k)
    rw [hdiv_eq, hmod] at h1
    have hmul : 2 ^ k * (2 * (x / 2 ^ (k + 1))) = 2 ^ (k + 1) * (x / 2 ^ (k + 1)) := by
      rw [hpow1, Nat.mul_assoc]
    omega
  have hx1 : x + 1 = 2 ^ (k + 1) * (x / 2 ^ (k + 1)) + 2 ^ k := by omega
  have hdv : 256 % 2 ^ (k + 1) = 0 := by
    have : ∀ m : Fin 8, 256 % 2 ^ (m.val + 1) = 0 := by decide +kernel
    exact this ⟨k, hk⟩
  have hmul256 : 2 ^ (k + 1) * (256 / 2 ^ (k + 1)) = 256 := by
    have := Nat.div_add_mod 256 (2 ^ (k + 1)); omega
  have hq_lt : x / 2 ^ (k + 1) < 256 / 2 ^ (k + 1) := by
    apply Classical.byContradiction
    intro hge
    have h1 : 2 ^ (k + 1) * (256 / 2 ^ (k + 1)) ≤ 2 ^ (k + 1) * (x / 2 ^ (k + 1)) :=
      Nat.mul_le_mul_left _ (by omega)
    omega
  have hlt256 : x + 1 < 256 := by
    have h1 : 2 ^ (k + 1) * (x / 2 ^ (k + 1) + 1) ≤ 256 := by
      calc 2 ^ (k + 1) * (x / 2 ^ (k + 1) + 1)
        _ ≤ 2 ^ (k + 1) * (256 / 2 ^ (k + 1)) := Nat.mul_le_mul_left _ (by omega)
        _ = 256 := hmul256
    rw [Nat.mul_add, Nat.mul_one] at h1
    omega
  refine ⟨hlt256, ?_⟩
  intro j hj
  rw [Nat.testBit_eq_decide_div_mod_eq]
  by_cases hjk : j < k
  · simp only [hjk, ↓reduceIte]
    have hdiv_pow : 2 ^ k = 2 ^ j * 2 ^ (k - j) := by rw [← Nat.pow_add, Nat.add_sub_of_le (by omega)]
    have hdiv_pow1 : 2 ^ (k + 1) = 2 ^ j * 2 ^ (k + 1 - j) := by
      rw [← Nat.pow_add, Nat.add_sub_of_le (by omega)]
    have heven1 : 2 ∣ 2 ^ (k - j) := by
      have : k - j = (k - j - 1) + 1 := by omega
      rw [this, Nat.pow_succ]; exact Nat.dvd_mul_left 2 _
    have heven2 : 2 ∣ 2 ^ (k + 1 - j) := by
      have : k + 1 - j = (k - j) + 1 := by omega
      rw [this, Nat.pow_succ]; exact Nat.dvd_mul_left 2 _
    have hq : (x + 1) / 2 ^ j = 2 ^ (k + 1 - j) * (x / 2 ^ (k + 1)) + 2 ^ (k - j) := by
      have hposj : 0 < 2 ^ j := Nat.two_pow_pos j
      calc (x + 1) / 2 ^ j
        _ = (2 ^ (k + 1) * (x / 2 ^ (k + 1)) + 2 ^ k) / 2 ^ j := by rw [hx1]
        _ = (2 ^ j * (2 ^ (k + 1 - j) * (x / 2 ^ (k + 1)) + 2 ^ (k - j))) / 2 ^ j := by
          rw [hdiv_pow1, hdiv_pow, Nat.mul_assoc, ← Nat.mul_add]
        _ = 2 ^ (k + 1 - j) * (x / 2 ^ (k + 1)) + 2 ^ (k - j) := Nat.mul_div_right _ hposj
    rw [hq]
    obtain ⟨m1, hm1⟩ := heven1
    obtain ⟨m2, hm2⟩ := heven2
    rw [hm1, hm2]
    have hev : 2 * m2 * (x / 2 ^ (k + 1)) + 2 * m1 = 2 * (m2 * (x / 2 ^ (k + 1)) + m1) := by
      rw [Nat.mul_assoc, ← Nat.mul_add]
    rw [hev]; simp
  · by_cases heq : j = k
    · subst heq
      simp only [Nat.lt_irrefl, ↓reduceIte]
      have hq : (x + 1) / 2 ^ j = 2 * (x / 2 ^ (j + 1)) + 1 := by
        rw [hx1, hpow1]
        have hposj : 0 < 2 ^ j := Nat.two_pow_pos j
        have : 2 ^ j * 2 * (x / (2 ^ j * 2)) + 2 ^ j =
            2 ^ j * (2 * (x / (2 ^ j * 2)) + 1) := by
          rw [Nat.mul_add, Nat.mul_one, Nat.mul_assoc]
        rw [this, Nat.mul_div_right _ hposj]
      rw [hq]; simp
    · simp only [hjk, heq, ↓reduceIte, Nat.testBit_eq_decide_div_mod_eq]
      have hgt : k + 1 ≤ j := by omega
      have hdiv_pow : 2 ^ j = 2 ^ (k + 1) * 2 ^ (j - (k + 1)) := by
        rw [← Nat.pow_add, Nat.add_sub_of_le hgt]
      have hq1 : (x + 1) / 2 ^ j = (x / 2 ^ (k + 1)) / 2 ^ (j - (k + 1)) := by
        rw [hdiv_pow, ← Nat.div_div_eq_div_mul, hx1,
          Nat.mul_add_div (Nat.two_pow_pos (k + 1)),
          Nat.div_eq_of_lt (show 2 ^ k < 2 ^ (k + 1) by rw [hpow1]; omega), Nat.add_zero]
      have hq2 : x / 2 ^ j = (x / 2 ^ (k + 1)) / 2 ^ (j - (k + 1)) := by
        rw [hdiv_pow, ← Nat.div_div_eq_div_mul]
      rw [hq1, hq2]

private theorem nextSub_step (m p : Byte) (hsub : p &&& m = p) (hne : p ≠ m) :
    let q := (p - m) &&& m
    q ≠ 0 ∧ q &&& m = q ∧ p.toNat < q.toNat ∧
    ∀ r : Nat, p.toNat < r → r < q.toNat → BitVec.ofNat 8 r &&& m ≠ BitVec.ofNat 8 r := by
  let q := (p - m) &&& m
  have hqm : q &&& m = q := by simp [q, BitVec.and_assoc]
  have hdiff_ne : m &&& ~~~p ≠ 0 := by
    intro hz
    apply hne
    apply BitVec.eq_of_getLsbD_eq
    intro i hi
    have h1 := congrArg (fun v : Byte => v.getLsbD i) hsub
    have h2 := congrArg (fun v : Byte => v.getLsbD i) hz
    rw [BitVec.getLsbD_and] at h1
    rw [BitVec.getLsbD_and, BitVec.getLsbD_not, show (0 : Byte).getLsbD i = false from BitVec.getLsbD_zero] at h2
    simp only [hi, decide_true, Bool.true_and] at h2
    cases hp : p.getLsbD i <;> cases hm : m.getLsbD i <;> simp_all
  obtain ⟨k, hk, hk_bit, hk_low⟩ := exists_lowest_set_bit (m &&& ~~~p) hdiff_ne
  rw [BitVec.getLsbD_and, BitVec.getLsbD_not] at hk_bit
  simp only [hk, decide_true, Bool.true_and, Bool.and_eq_true, Bool.not_eq_true'] at hk_bit
  obtain ⟨hmk, hpk⟩ := hk_bit
  let x := p ||| ~~~m
  have hx_lo : ∀ j, j < k → x.toNat.testBit j = true := by
    intro j hj
    rw [← bv_bit]
    have hlj := hk_low j hj
    rw [BitVec.getLsbD_and, BitVec.getLsbD_not] at hlj
    rw [show x = p ||| ~~~m from rfl, BitVec.getLsbD_or, BitVec.getLsbD_not]
    simp only [show j < 8 by omega, decide_true, Bool.true_and] at hlj ⊢
    cases hp : p.getLsbD j <;> cases hm : m.getLsbD j <;> simp_all
  have hx_k : x.toNat.testBit k = false := by
    rw [← bv_bit, show x = p ||| ~~~m from rfl, BitVec.getLsbD_or, BitVec.getLsbD_not, hmk, hpk]
    simp [hk]
  obtain ⟨hx1_lt, hx1_bit⟩ := nat_testBit_add_low_ones x.toNat k hk x.isLt hx_lo hx_k
  have hq_bit : ∀ j, j < 8 → q.getLsbD j =
      if j < k then false else if j = k then true else p.getLsbD j := by
    intro j hj
    have hsub_eq := sub_eq_or_not_add_one p m hsub
    have hadd_toNat : (x + 1).toNat = x.toNat + 1 := by
      rw [BitVec.toNat_add]; change (x.toNat + 1) % 256 = x.toNat + 1; omega
    have hbit_add : (x + 1).getLsbD j =
        if j < k then false else if j = k then true else x.getLsbD j := by
      rw [bv_bit, hadd_toNat, hx1_bit j hj, ← bv_bit]
    change ((p - m) &&& m).getLsbD j = _
    rw [hsub_eq, BitVec.getLsbD_and, hbit_add]
    by_cases hjk : j < k
    · simp [hjk]
    · by_cases heq : j = k
      · subst heq; simp [hmk]
      · simp only [hjk, heq, ↓reduceIte]
        rw [show x = p ||| ~~~m from rfl, BitVec.getLsbD_or, BitVec.getLsbD_not]
        simp only [hj, decide_true, Bool.true_and]
        have hp_sub := congrArg (fun v : Byte => v.getLsbD j) hsub
        rw [BitVec.getLsbD_and] at hp_sub
        cases hp : p.getLsbD j <;> cases hm : m.getLsbD j <;> simp_all
  have hq_ne : q ≠ 0 := by
    intro hz
    have := hq_bit k hk
    rw [hz] at this; simp at this
  have hdiv_eq : ∀ a b : Byte, (∀ j, k < j → j < 8 → a.getLsbD j = b.getLsbD j) →
      a.toNat / 2 ^ (k + 1) = b.toNat / 2 ^ (k + 1) := by
    intro a b hhi
    apply Nat.eq_of_testBit_eq
    intro j
    rw [Nat.testBit_div_two_pow, Nat.testBit_div_two_pow]
    by_cases hlt : k + 1 + j < 8
    · rw [← bv_bit, ← bv_bit, Nat.add_comm j (k + 1)]
      exact hhi (k + 1 + j) (by omega) hlt
    · rw [Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le a.isLt (Nat.pow_le_pow_right (by decide) (by omega))),
          Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le b.isLt (Nat.pow_le_pow_right (by decide) (by omega)))]
  have hq_div : q.toNat / 2 ^ (k + 1) = p.toNat / 2 ^ (k + 1) :=
    hdiv_eq q p (fun j hgt hj => by rw [hq_bit j hj]; simp [show ¬ j < k by omega, show j ≠ k by omega])
  have hq_mod : q.toNat % 2 ^ (k + 1) = 2 ^ k := by
    apply Nat.eq_of_testBit_eq
    intro j
    rw [Nat.testBit_mod_two_pow, Nat.testBit_two_pow]
    by_cases hlt : j < k + 1
    · simp only [hlt, decide_true, Bool.true_and]
      rw [← bv_bit, hq_bit j (by omega)]
      by_cases hjk : j < k
      · simp [hjk, show k ≠ j by omega]
      · have heq : j = k := by omega
        subst heq; simp
    · simp [hlt, show k ≠ j by omega]
  have hp_mod_lt : p.toNat % 2 ^ (k + 1) < 2 ^ k := by
    have hpow1 : 2 ^ (k + 1) = 2 ^ k * 2 := Nat.pow_succ 2 k
    have hbitk : (p.toNat / 2 ^ k) % 2 = 0 := by
      rw [bv_bit, Nat.testBit_eq_decide_div_mod_eq, decide_eq_false_iff_not] at hpk
      omega
    have h1 := Nat.div_add_mod p.toNat (2 ^ k)
    have h2 := Nat.div_add_mod p.toNat (2 ^ (k + 1))
    have h3 : p.toNat / 2 ^ (k + 1) = (p.toNat / 2 ^ k) / 2 := by
      rw [hpow1, ← Nat.div_div_eq_div_mul]
    have hdiv2 : 2 * ((p.toNat / 2 ^ k) / 2) = p.toNat / 2 ^ k := by omega
    have hmul : 2 ^ (k + 1) * (p.toNat / 2 ^ (k + 1)) = 2 ^ k * (p.toNat / 2 ^ k) := by
      rw [h3, hpow1, Nat.mul_assoc, hdiv2]
    have hlt := Nat.mod_lt p.toNat (Nat.two_pow_pos k)
    omega
  have hlt_pq : p.toNat < q.toNat := by
    have hp_eq := Nat.div_add_mod p.toNat (2 ^ (k + 1))
    have hq_eq := Nat.div_add_mod q.toNat (2 ^ (k + 1))
    rw [hq_div, hq_mod] at hq_eq
    omega
  refine ⟨hq_ne, hqm, hlt_pq, ?_⟩
  intro r hpr hrq hrm
  change r < q.toNat at hrq
  have hr_lt : r < 256 := Nat.lt_trans hrq q.isLt
  have hrb_toNat : (BitVec.ofNat 8 r).toNat = r := by rw [BitVec.toNat_ofNat]; omega
  have hr_div : r / 2 ^ (k + 1) = p.toNat / 2 ^ (k + 1) := by
    have hp_eq := Nat.div_add_mod p.toNat (2 ^ (k + 1))
    have hq_eq := Nat.div_add_mod q.toNat (2 ^ (k + 1))
    have hr_eq := Nat.div_add_mod r (2 ^ (k + 1))
    rw [hq_div, hq_mod] at hq_eq
    have hpow1 : 2 ^ (k + 1) = 2 * 2 ^ k := by rw [Nat.pow_succ, Nat.mul_comm]
    have hr_mod := Nat.mod_lt r (Nat.two_pow_pos (k + 1))
    apply Classical.byContradiction
    intro hne_div
    rcases Nat.lt_or_gt_of_ne hne_div with hlt | hgt
    · have : 2 ^ (k + 1) * (r / 2 ^ (k + 1) + 1) ≤ 2 ^ (k + 1) * (p.toNat / 2 ^ (k + 1)) :=
        Nat.mul_le_mul_left _ (by omega)
      rw [Nat.mul_add, Nat.mul_one] at this; omega
    · have : 2 ^ (k + 1) * (p.toNat / 2 ^ (k + 1) + 1) ≤ 2 ^ (k + 1) * (r / 2 ^ (k + 1)) :=
        Nat.mul_le_mul_left _ (by omega)
      rw [Nat.mul_add, Nat.mul_one] at this; omega
  have hr_mod_lt : r % 2 ^ (k + 1) < 2 ^ k := by
    have hq_eq := Nat.div_add_mod q.toNat (2 ^ (k + 1))
    have hr_eq := Nat.div_add_mod r (2 ^ (k + 1))
    rw [hq_div, hq_mod] at hq_eq
    rw [hr_div] at hr_eq
    omega
  have hp_mod_lt_r : p.toNat % 2 ^ k < r % 2 ^ k := by
    have hp_eq := Nat.div_add_mod p.toNat (2 ^ (k + 1))
    have hr_eq := Nat.div_add_mod r (2 ^ (k + 1))
    rw [hr_div] at hr_eq
    have h1 : p.toNat % 2 ^ (k + 1) = p.toNat % 2 ^ k := by
      have hpow1 : 2 ^ (k + 1) = 2 ^ k * 2 := Nat.pow_succ 2 k
      rw [← Nat.mod_eq_of_lt hp_mod_lt, hpow1, Nat.mod_mul_right_mod]
    have h2 : r % 2 ^ (k + 1) = r % 2 ^ k := by
      have hpow1 : 2 ^ (k + 1) = 2 ^ k * 2 := Nat.pow_succ 2 k
      rw [← Nat.mod_eq_of_lt hr_mod_lt, hpow1, Nat.mod_mul_right_mod]
    omega
  have hle_rp : r % 2 ^ k ≤ p.toNat % 2 ^ k := by
    apply nat_le_of_testBit_imp k (r % 2 ^ k) (p.toNat % 2 ^ k)
      (Nat.mod_lt _ (Nat.two_pow_pos k)) (Nat.mod_lt _ (Nat.two_pow_pos k))
    intro j hj hrj
    rw [Nat.testBit_mod_two_pow] at hrj ⊢
    simp only [hj, decide_true, Bool.true_and] at hrj ⊢
    have hrm_j := congrArg (fun v : Byte => v.getLsbD j) hrm
    rw [BitVec.getLsbD_and, bv_bit, hrb_toNat, hrj, Bool.true_and] at hrm_j
    have hlj := hk_low j hj
    rw [BitVec.getLsbD_and, BitVec.getLsbD_not] at hlj
    simp only [show j < 8 by omega, decide_true, Bool.true_and, hrm_j, Bool.true_and,
      Bool.not_eq_false'] at hlj
    exact hlj
  omega

private def subsetSpec (m : Byte) (lo : Nat) : List Byte :=
  ((List.range 256).filter (fun r => decide (lo ≤ r) && (BitVec.ofNat 8 r &&& m == BitVec.ofNat 8 r))).map
    (BitVec.ofNat 8)

private theorem filter_range_split (pred : Nat → Bool) (lo hi : Nat) (hlt : lo < hi) (hhi : hi ≤ 256)
    (hpred_lo : pred lo = true) (hnone : ∀ r, lo < r → r < hi → pred r = false) :
    (List.range 256).filter (fun r => decide (lo ≤ r) && pred r) =
      lo :: (List.range 256).filter (fun r => decide (hi ≤ r) && pred r) := by
  have aux : ∀ n : Nat, n ≤ 256 →
      (List.range n).filter (fun r => decide (lo ≤ r) && pred r) =
        (if lo < n then [lo] else []) ++ (List.range n).filter (fun r => decide (hi ≤ r) && pred r) := by
    intro n hn
    induction n with
    | zero => simp
    | succ n ih =>
      rw [List.range_succ, List.filter_append, List.filter_append, ih (by omega)]
      simp only [List.filter_cons, List.filter_nil]
      by_cases hln : lo < n
      · have hln1 : lo < n + 1 := by omega
        have hlo_le : lo ≤ n := by omega
        by_cases hhn : hi ≤ n
        · simp [hln, hln1, hlo_le, hhn]
        · have hfalse : pred n = false := hnone n hln (by omega)
          simp [hln, hln1, hlo_le, hhn, hfalse]
      · by_cases heq : n = lo
        · subst heq
          have hlt1 : n < n + 1 := Nat.lt_succ_self n
          have hhn : ¬ hi ≤ n := by omega
          have hnil : (List.range n).filter (fun r => decide (hi ≤ r) && pred r) = [] := by
            apply List.filter_eq_nil_iff.mpr
            intro r hr; simp only [List.mem_range] at hr; simp; omega
          simp [hlt1, hpred_lo, hhn, hnil]
        · have hnot1 : ¬ lo < n + 1 := by omega
          have hnotle : ¬ lo ≤ n := by omega
          have hnot_hi : ¬ hi ≤ n := by omega
          simp [hln, hnot1, hnotle, hnot_hi]
  have h := aux 256 (Nat.le_refl 256)
  have hlt256 : lo < 256 := by omega
  simp [hlt256] at h
  exact h

private theorem filter_range_empty_above (m : Byte) :
    (List.range 256).filter (fun r =>
      decide (m.toNat + 1 ≤ r) && (BitVec.ofNat 8 r &&& m == BitVec.ofNat 8 r)) = [] := by
  apply List.filter_eq_nil_iff.mpr
  intro r hr
  simp only [List.mem_range] at hr
  simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq, not_and]
  intro hge hsub
  have hle := byte_le_of_sub (BitVec.ofNat 8 r) m hsub
  rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt hr] at hle
  omega

private theorem sourceSubsets_and_stops (m : Byte) :
    ∀ (fuel : Nat) (p : Byte), p &&& m = p → 256 - p.toNat ≤ fuel →
      subsetLoopStops m p fuel = true ∧ sourceSubsets m p fuel = subsetSpec m p.toNat := by
  intro fuel
  induction fuel with
  | zero =>
    intro p _ hfuel
    have := p.isLt; omega
  | succ fuel ih =>
    intro p hsub hfuel
    unfold subsetLoopStops sourceSubsets
    dsimp only
    by_cases hpm : p = m
    · subst hpm
      have h0 : (p - p) &&& p = 0 := by simp
      simp only [h0, beq_self_eq_true, ↓reduceIte, true_and]
      unfold subsetSpec
      have hfilt := filter_range_split
        (fun r => BitVec.ofNat 8 r &&& p == BitVec.ofNat 8 r)
        p.toNat (p.toNat + 1) (by omega) (by have := p.isLt; omega)
        (by simp [BitVec.ofNat_toNat, BitVec.setWidth_eq, hsub])
        (fun r h1 h2 => by omega)
      rw [hfilt, filter_range_empty_above p]
      simp [BitVec.ofNat_toNat, BitVec.setWidth_eq]
    · obtain ⟨hq_ne, hq_sub, hlt_pq, hgap⟩ := nextSub_step m p hsub hpm
      have hne_beq : (((p - m) &&& m) == 0) = false := beq_false_of_ne hq_ne
      simp only [hne_beq, Bool.false_eq_true, ↓reduceIte]
      have ⟨hstop, hlist⟩ := ih ((p - m) &&& m) hq_sub (by omega)
      refine ⟨hstop, ?_⟩
      rw [hlist]
      unfold subsetSpec
      have hfilt := filter_range_split
        (fun r => BitVec.ofNat 8 r &&& m == BitVec.ofNat 8 r)
        p.toNat ((p - m) &&& m).toNat hlt_pq (by have := ((p - m) &&& m).isLt; omega)
        (by simp [BitVec.ofNat_toNat, BitVec.setWidth_eq, hsub])
        (fun r h1 h2 => beq_false_of_ne (hgap r h1 h2))
      rw [hfilt]
      simp [BitVec.ofNat_toNat, BitVec.setWidth_eq]

theorem source_subset_loop_stops (mask : Byte) : subsetLoopStops mask 0 256 = true :=
  (sourceSubsets_and_stops mask 256 0 (by simp) (by simp)).1

theorem source_subsets_exact (mask : Byte) :
    sourceSubsets mask 0 256 =
      ((List.range 256).filter (fun p =>
        BitVec.ofNat 8 p &&& mask == BitVec.ofNat 8 p)).map (BitVec.ofNat 8) := by
  have h := (sourceSubsets_and_stops mask 256 0 (by simp) (by simp)).2
  simpa [subsetSpec] using h

end NeuralReversi.Stability.Edge

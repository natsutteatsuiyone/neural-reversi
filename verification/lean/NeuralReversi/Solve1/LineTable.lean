module

@[expose] public section

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

/-- Fast structural 8-bit leading-zero count (no `Nat.brecOn`). -/
def fastClz8Aux (b : Byte) : Nat → Nat
  | 0 => if b.getLsbD 0 then 7 else 8
  | n + 1 => if b.getLsbD (n + 1) then 7 - (n + 1) else fastClz8Aux b n

def fastClz8 (b : Byte) : Nat := fastClz8Aux b 7

/-- Fast structural 8-bit trailing-zero count (no `Nat.brecOn`). -/
def fastCtz8Aux (b : Byte) : Nat → Nat
  | 0 => if b.getLsbD 7 then 7 else 8
  | n + 1 => if b.getLsbD (7 - (n + 1)) then 7 - (n + 1) else fastCtz8Aux b n

def fastCtz8 (b : Byte) : Nat := fastCtz8Aux b 7

private theorem clzAuxRec_hi_zero {w : Nat} (x : BitVec w) (m : Nat) :
    ∀ n : Nat, m ≤ n → (∀ k, m < k → k ≤ n → x.getLsbD k = false) →
      x.clzAuxRec n = x.clzAuxRec m := by
  intro n hle hzero
  induction n with
  | zero =>
    have hm : m = 0 := by omega
    subst hm; rfl
  | succ n ih =>
    by_cases h : m = n + 1
    · subst h; rfl
    · rw [BitVec.clzAuxRec.eq_2, hzero (n + 1) (by omega) (Nat.le_refl _)]
      exact ih (by omega) (fun k hk1 hk2 => hzero k hk1 (by omega))

private theorem clzAuxRec_setWidth64_fast (b : Byte) :
    ∀ n : Nat, n < 8 →
      ((b.setWidth 64).clzAuxRec n).toNat =
        if fastClz8Aux b n = 8 then 64 else 56 + fastClz8Aux b n := by
  intro n hn
  induction n with
  | zero =>
    rw [BitVec.clzAuxRec.eq_1, fastClz8Aux, BitVec.getLsbD_setWidth]
    cases b.getLsbD 0 <;> decide
  | succ n ih =>
    rw [BitVec.clzAuxRec.eq_2, fastClz8Aux, BitVec.getLsbD_setWidth]
    have hlt : n + 1 < 64 := by omega
    simp only [hlt, decide_true, Bool.true_and]
    cases b.getLsbD (n + 1) with
    | false => exact ih (by omega)
    | true =>
      have hne : 7 - (n + 1) ≠ 8 := by omega
      simp only [↓reduceIte, hne, BitVec.toNat_ofNat]
      omega

private theorem ray_bound (line : Byte) (acc : Nat) (xs : List Nat) :
    ray line acc xs ≤ if xs.isEmpty then 0 else acc + xs.length - 1 := by
  induction xs generalizing acc with
  | nil => simp [ray]
  | cons i rest ih =>
    unfold ray
    split
    · cases rest <;> simp <;> omega
    · have h := ih (acc + 1)
      cases rest with
      | nil => simp [ray]
      | cons j rest' =>
        simp only [List.isEmpty_cons, Bool.false_eq_true, ↓reduceIte, List.length_cons] at h ⊢
        omega

private theorem ray_congr (line1 line2 : Byte) (acc : Nat) (xs : List Nat)
    (h : ∀ i ∈ xs, line1.getLsbD i = line2.getLsbD i) :
    ray line1 acc xs = ray line2 acc xs := by
  induction xs generalizing acc with
  | nil => rfl
  | cons i rest ih =>
    simp only [ray, h i (by simp), ih (acc + 1) (fun j hj => h j (List.mem_cons_of_mem _ hj))]

private theorem filter_gt_len : ∀ pos : Fin 8,
    ((List.range 8).filter (fun i => pos.val < i)).length = 7 - pos.val := by decide +kernel

private theorem fastClz8Aux_bound (b : Byte) (n : Nat) :
    fastClz8Aux b n = 8 ∨ (7 - n ≤ fastClz8Aux b n ∧ fastClz8Aux b n ≤ 7) := by
  induction n with
  | zero =>
    unfold fastClz8Aux
    split <;> omega
  | succ n ih =>
    unfold fastClz8Aux
    split
    · right; omega
    · rcases ih with h | ⟨h1, h2⟩
      · exact Or.inl h
      · exact Or.inr ⟨by omega, h2⟩

private theorem fastClz8Aux_eq_eight (b : Byte) :
    ∀ n : Nat, fastClz8Aux b n = 8 ↔ ∀ k, k ≤ n → b.getLsbD k = false := by
  intro n
  induction n with
  | zero =>
    unfold fastClz8Aux
    cases hb : b.getLsbD 0 with
    | true =>
      simp only [↓reduceIte, Nat.reduceEqDiff, false_iff]
      intro hall
      have := hall 0 (Nat.le_refl 0)
      rw [hb] at this; exact Bool.noConfusion this
    | false =>
      simp only [Bool.false_eq_true, ↓reduceIte, true_iff]
      intro k hk
      have hk0 : k = 0 := by omega
      subst hk0; exact hb
  | succ n ih =>
    unfold fastClz8Aux
    cases hb : b.getLsbD (n + 1) with
    | true =>
      have hne : 7 - (n + 1) ≠ 8 := by omega
      simp only [↓reduceIte, hne, false_iff]
      intro hall
      have := hall (n + 1) (Nat.le_refl _)
      rw [hb] at this; exact Bool.noConfusion this
    | false =>
      simp only [Bool.false_eq_true, ↓reduceIte, ih]
      constructor
      · intro h k hk
        by_cases heq : k = n + 1
        · subst heq; exact hb
        · exact h k (by omega)
      · intro h k hk
        exact h k (by omega)

private theorem fastClz8_lt_eight (b : Byte) (hb : b ≠ 0) : fastClz8 b ≤ 7 := by
  rcases fastClz8Aux_bound b 7 with h8 | ⟨_, h7⟩
  · exfalso
    have hall := (fastClz8Aux_eq_eight b 7).mp h8
    apply hb
    apply BitVec.eq_of_getLsbD_eq
    intro i hi
    simpa using hall i (by omega)
  · exact h7

theorem setWidth64_clz_toNat (b : Byte) (hb : b ≠ 0) :
    (b.setWidth 64).clz.toNat = 56 + fastClz8 b := by
  change ((b.setWidth 64).clzAuxRec 63).toNat = 56 + fastClz8Aux b 7
  rw [clzAuxRec_hi_zero (b.setWidth 64) 7 63 (by omega) (fun k hk1 _ => by
    rw [BitVec.getLsbD_setWidth, BitVec.getLsbD_of_ge b k (by omega), Bool.and_false])]
  rw [clzAuxRec_setWidth64_fast b 7 (by omega)]
  have hle := fastClz8_lt_eight b hb
  have hne : fastClz8Aux b 7 ≠ 8 := by unfold fastClz8 at hle; omega
  simp only [hne, ↓reduceIte]

private theorem ctzAuxRec_setWidth64_fast (b : Byte) :
    ∀ n : Nat, n < 8 →
      (∃ k, k ≤ n ∧ b.getLsbD (7 - k) = true) →
      ((b.setWidth 64).reverse.clzAuxRec (56 + n)).toNat = fastCtz8Aux b n := by
  intro n hn hex
  induction n with
  | zero =>
    obtain ⟨k, hk, hbit⟩ := hex
    have hk0 : k = 0 := by omega
    subst hk0
    have hbit64 : (b.setWidth 64).reverse.getLsbD 56 = true := by
      simp only [BitVec.getLsbD_reverse, BitVec.getMsbD, BitVec.getLsbD_setWidth]
      simpa using hbit
    rw [Nat.add_zero, BitVec.clzAuxRec.eq_2, fastCtz8Aux, hbit, hbit64]
    simp only [↓reduceIte]
    rfl
  | succ n ih =>
    have hbit_eq : (b.setWidth 64).reverse.getLsbD (56 + n).succ = b.getLsbD (7 - (n + 1)) := by
      simp only [BitVec.getLsbD_reverse, BitVec.getMsbD, BitVec.getLsbD_setWidth]
      have h1 : (56 + n).succ < 64 := by omega
      have h2 : 63 - (56 + n).succ = 7 - (n + 1) := by omega
      have h3 : 7 - (n + 1) < 64 := by omega
      simp only [h1, h2, h3, decide_true, Bool.true_and]
    rw [show 56 + (n + 1) = (56 + n).succ by omega, BitVec.clzAuxRec.eq_2, fastCtz8Aux, hbit_eq]
    cases hb : b.getLsbD (7 - (n + 1)) with
    | true =>
      simp only [↓reduceIte, BitVec.toNat_ofNat]
      omega
    | false =>
      simp only [Bool.false_eq_true, ↓reduceIte]
      apply ih (by omega)
      obtain ⟨k, hk, hbit⟩ := hex
      by_cases hkn : k = n + 1
      · subst hkn
        rw [hb] at hbit
        exact Bool.noConfusion hbit
      · exact ⟨k, by omega, hbit⟩

private theorem byte_nonzero_bit (b : Byte) (hb : b ≠ 0) :
    ∃ k, k ≤ 7 ∧ b.getLsbD (7 - k) = true := by
  apply Classical.byContradiction
  intro hall
  apply hb
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  rw [show (0 : Byte).getLsbD i = false from BitVec.getLsbD_zero]
  cases h : b.getLsbD i with
  | false => rfl
  | true =>
    exfalso
    apply hall
    exact ⟨7 - i, by omega, by simpa [show 7 - (7 - i) = i by omega] using h⟩

theorem setWidth64_ctz_toNat (b : Byte) (hb : b ≠ 0) :
    (b.setWidth 64).ctz.toNat = fastCtz8 b :=
  ctzAuxRec_setWidth64_fast b 7 (by omega) (byte_nonzero_bit b hb)

theorem above_eq_setWidth (pos : Fin 8) (line : Byte) :
    line.setWidth 64 >>> (pos.val + 1) = (line >>> (pos.val + 1)).setWidth 64 := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [BitVec.getLsbD_ushiftRight, BitVec.getLsbD_setWidth, hi, decide_true, Bool.true_and]
  by_cases h : pos.val + 1 + i < 64
  · simp [h]
  · simp [h, BitVec.getLsbD_of_ge line (pos.val + 1 + i) (by omega)]

private theorem belowMask_eq : ∀ p : Fin 8,
    (((1 : Usize) <<< p.val) - 1) = ((((1 : Byte) <<< p.val) - 1)).setWidth 64 := by
  decide +kernel

theorem below_eq_setWidth (pos : Fin 8) (line : Byte) :
    line.setWidth 64 &&& (((1 : Usize) <<< pos.val) - 1) =
      (line &&& (((1 : Byte) <<< pos.val) - 1)).setWidth 64 := by
  rw [belowMask_eq pos]
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [BitVec.getLsbD_and, BitVec.getLsbD_setWidth, hi, decide_true, Bool.true_and]

theorem setWidth64_eq_zero_iff (b : Byte) : b.setWidth 64 = (0 : Usize) ↔ b = (0 : Byte) := by
  constructor
  · intro h
    apply BitVec.eq_of_toNat_eq
    have ht := congrArg BitVec.toNat h
    have h0 : (0 : Usize).toNat = 0 := rfl
    have h0b : (0 : Byte).toNat = 0 := rfl
    rw [BitVec.toNat_setWidth, h0] at ht
    have hlt := b.isLt
    rw [h0b]
    omega
  · intro h; subst h; rfl

private theorem belowMask_bit : ∀ p : Fin 8, ∀ i : Fin 8,
    ((((1 : Byte) <<< p.val) - 1)).getLsbD i.val = decide (i.val < p.val) := by decide +kernel

private theorem fastCtz8Aux_eq_ray (line : Byte) (pos : Nat) (hpos : pos < 8) :
    ∀ n : Nat, n < 8 →
      (if fastCtz8Aux (line >>> (pos + 1)) n = 8 then 0 else fastCtz8Aux (line >>> (pos + 1)) n) =
        ray line (7 - n) ((List.range 8).filter (fun i => pos + 1 + (7 - n) ≤ i)) := by
  intro n hn
  induction n with
  | zero =>
    have h2 : (List.range 8).filter (fun i => pos + 1 + 7 ≤ i) = [] := by
      apply List.filter_eq_nil_iff.mpr
      intro i hi
      simp only [List.mem_range] at hi
      simp; omega
    have h1 : (line >>> (pos + 1)).getLsbD 7 = false := by
      rw [BitVec.getLsbD_ushiftRight]
      exact BitVec.getLsbD_of_ge line (pos + 1 + 7) (by omega)
    rw [fastCtz8Aux, h1]
    simp [h2, ray]
  | succ n ih =>
    let start := 7 - (n + 1)
    have hstart_next : 7 - n = start + 1 := by omega
    by_cases hlt : pos + 1 + start < 8
    · have hfilter : (List.range 8).filter (fun i => pos + 1 + start ≤ i) =
          (pos + 1 + start) :: (List.range 8).filter (fun i => pos + 1 + (start + 1) ≤ i) := by
        have h : ∀ p : Fin 8, ∀ s : Fin 8, p.val + 1 + s.val < 8 →
            (List.range 8).filter (fun i => p.val + 1 + s.val ≤ i) =
            (p.val + 1 + s.val) :: (List.range 8).filter (fun i => p.val + 1 + (s.val + 1) ≤ i) := by
          decide +kernel
        exact h ⟨pos, hpos⟩ ⟨start, by omega⟩ hlt
      change (if fastCtz8Aux (line >>> (pos + 1)) (n + 1) = 8 then 0
        else fastCtz8Aux (line >>> (pos + 1)) (n + 1)) =
        ray line start ((List.range 8).filter (fun i => pos + 1 + start ≤ i))
      rw [hfilter, ray, fastCtz8Aux]
      have hbit : (line >>> (pos + 1)).getLsbD start = line.getLsbD (pos + 1 + start) := by
        rw [BitVec.getLsbD_ushiftRight]
      change (if (if (line >>> (pos + 1)).getLsbD start = true then start else fastCtz8Aux (line >>> (pos + 1)) n) = 8
        then 0 else if (line >>> (pos + 1)).getLsbD start = true then start else fastCtz8Aux (line >>> (pos + 1)) n) = _
      rw [hbit]
      cases line.getLsbD (pos + 1 + start) with
      | true =>
        have hne : start ≠ 8 := by omega
        simp only [↓reduceIte, hne]
      | false =>
        simp only [Bool.false_eq_true, ↓reduceIte]
        simpa only [hstart_next] using ih (by omega)
    · have hbit : (line >>> (pos + 1)).getLsbD start = false := by
        rw [BitVec.getLsbD_ushiftRight]
        exact BitVec.getLsbD_of_ge line (pos + 1 + start) (by omega)
      have hfilt1 : (List.range 8).filter (fun i => pos + 1 + start ≤ i) = [] := by
        apply List.filter_eq_nil_iff.mpr
        intro i hi
        simp only [List.mem_range] at hi
        simp; omega
      have hfilt2 : (List.range 8).filter (fun i => pos + 1 + (7 - n) ≤ i) = [] := by
        apply List.filter_eq_nil_iff.mpr
        intro i hi
        simp only [List.mem_range] at hi
        simp; omega
      change (if fastCtz8Aux (line >>> (pos + 1)) (n + 1) = 8 then 0
        else fastCtz8Aux (line >>> (pos + 1)) (n + 1)) =
        ray line start ((List.range 8).filter (fun i => pos + 1 + start ≤ i))
      rw [fastCtz8Aux]
      change (if (if (line >>> (pos + 1)).getLsbD start = true then start else fastCtz8Aux (line >>> (pos + 1)) n) = 8
        then 0 else if (line >>> (pos + 1)).getLsbD start = true then start else fastCtz8Aux (line >>> (pos + 1)) n) = _
      rw [hbit, hfilt1, ray]
      simp only [Bool.false_eq_true, ↓reduceIte]
      have ih' := ih (by omega)
      rw [hfilt2, ray] at ih'
      exact ih'

private theorem fastCtz8_eq_eight (a : Byte) :
    fastCtz8 a = 8 ↔ a = 0 := by
  have aux : ∀ n : Nat, n < 8 →
      (fastCtz8Aux a n = 8 ↔ ∀ k, 7 - n ≤ k → k < 8 → a.getLsbD k = false) := by
    intro n hn
    induction n with
    | zero =>
      unfold fastCtz8Aux
      cases hb : a.getLsbD 7 with
      | true =>
        simp only [↓reduceIte, Nat.reduceEqDiff, false_iff]
        intro hall
        have := hall 7 (by omega) (by omega)
        rw [hb] at this; exact Bool.noConfusion this
      | false =>
        simp only [Bool.false_eq_true, ↓reduceIte, true_iff]
        intro k hk1 hk2
        have : k = 7 := by omega
        subst this; exact hb
    | succ n ih =>
      unfold fastCtz8Aux
      cases hb : a.getLsbD (7 - (n + 1)) with
      | true =>
        have hne : 7 - (n + 1) ≠ 8 := by omega
        simp only [↓reduceIte, hne, false_iff]
        intro hall
        have := hall (7 - (n + 1)) (by omega) (by omega)
        rw [hb] at this; exact Bool.noConfusion this
      | false =>
        simp only [Bool.false_eq_true, ↓reduceIte, ih (by omega)]
        constructor
        · intro h k hk1 hk2
          by_cases heq : k = 7 - (n + 1)
          · subst heq; exact hb
          · exact h k (by omega) hk2
        · intro h k hk1 hk2
          exact h k (by omega) hk2
  rw [fastCtz8, aux 7 (by omega)]
  constructor
  · intro h
    apply BitVec.eq_of_getLsbD_eq
    intro i hi
    simpa using h i (by omega) hi
  · intro h k _ _
    subst h; simp

private theorem fastCtz8_above_le (pos : Fin 8) (line : Byte)
    (ha : line >>> (pos.val + 1) ≠ 0) : fastCtz8 (line >>> (pos.val + 1)) ≤ 6 := by
  let a := line >>> (pos.val + 1)
  have h7 : a.getLsbD 7 = false := by
    rw [BitVec.getLsbD_ushiftRight]
    exact BitVec.getLsbD_of_ge line (pos.val + 1 + 7) (by omega)
  have hne : fastCtz8 a ≠ 8 := fun h => ha ((fastCtz8_eq_eight a).mp h)
  have aux : ∀ n : Nat, n < 8 →
      (fastCtz8Aux a n = 8 ∨ fastCtz8Aux a n ≤ 6) := by
    intro n hn
    induction n with
    | zero =>
      unfold fastCtz8Aux
      rw [h7]; left; rfl
    | succ n ih =>
      unfold fastCtz8Aux
      split
      · right; omega
      · exact ih (by omega)
  rcases aux 7 (by omega) with h8 | h6
  · exact absurd h8 hne
  · exact h6

private theorem fastClz8Aux_eq_ray (line : Byte) (pos : Fin 8) :
    ∀ n : Nat, n < pos.val →
      let b : Byte := line &&& (((1 : Byte) <<< pos.val) - 1)
      (if fastClz8Aux b n = 8 then 0 else pos.val - 1 - (7 - fastClz8Aux b n)) =
        ray line (pos.val - 1 - n) (List.range (n + 1)).reverse := by
  intro n hn
  induction n with
  | zero =>
    dsimp only
    have hbit : (line &&& (((1 : Byte) <<< pos.val) - 1)).getLsbD 0 = line.getLsbD 0 := by
      rw [BitVec.getLsbD_and, belowMask_bit pos ⟨0, by decide⟩]
      have h0 : 0 < pos.val := by omega
      simp [h0]
    simp only [fastClz8Aux, hbit, List.range_succ, List.range_zero, List.nil_append,
      List.reverse_cons, List.reverse_nil, ray]
    cases line.getLsbD 0 <;> simp
  | succ n ih =>
    dsimp only
    have hbit : (line &&& (((1 : Byte) <<< pos.val) - 1)).getLsbD (n + 1) = line.getLsbD (n + 1) := by
      rw [BitVec.getLsbD_and, belowMask_bit pos ⟨n + 1, by omega⟩]
      have hlt : n + 1 < pos.val := by omega
      simp [hlt]
    rw [List.range_succ, List.reverse_append, List.reverse_singleton, List.singleton_append,
      ray, fastClz8Aux, hbit]
    cases line.getLsbD (n + 1) with
    | true =>
      have hne : 7 - (n + 1) ≠ 8 := by omega
      have heq : 7 - (7 - (n + 1)) = n + 1 := by omega
      simp only [↓reduceIte, hne, heq]
    | false =>
      simp only [Bool.false_eq_true, ↓reduceIte]
      have hacc : pos.val - 1 - (n + 1) + 1 = pos.val - 1 - n := by omega
      rw [hacc]
      exact ih (by omega)

private theorem fastClz8_below_skip (line : Byte) (pos : Fin 8) :
    ∀ n : Nat, pos.val ≤ n → n < 8 →
      fastClz8Aux (line &&& (((1 : Byte) <<< pos.val) - 1)) n =
        if pos.val = 0 then 8 else fastClz8Aux (line &&& (((1 : Byte) <<< pos.val) - 1)) (pos.val - 1) := by
  intro n hge hn
  induction n with
  | zero =>
    have hp0 : pos.val = 0 := by omega
    have hbit : (line &&& (((1 : Byte) <<< pos.val) - 1)).getLsbD 0 = false := by
      rw [BitVec.getLsbD_and, belowMask_bit pos ⟨0, by decide⟩, hp0]; simp
    rw [fastClz8Aux, hbit]; simp [hp0]
  | succ n ih =>
    have hbit : (line &&& (((1 : Byte) <<< pos.val) - 1)).getLsbD (n + 1) = false := by
      rw [BitVec.getLsbD_and, belowMask_bit pos ⟨n + 1, by omega⟩]
      have hnot : ¬ (n + 1 < pos.val) := by omega
      simp [hnot]
    rw [fastClz8Aux, hbit]
    simp only [Bool.false_eq_true, ↓reduceIte]
    by_cases heq : pos.val = n + 1
    · simp [heq]
    · exact ih (by omega) (by omega)

theorem reference_bound (pos : Fin 8) (line : Byte) : reference pos line ≤ 6 := by
  unfold reference
  have h1 := ray_bound line 0 ((List.range 8).filter (fun i => pos.val < i))
  have h2 := ray_bound line 0 (List.range pos.val).reverse
  have hlen1 := filter_gt_len pos
  have hlen2 : (List.range pos.val).reverse.length = pos.val := by simp
  have hpos := pos.isLt
  have hb1 : ray line 0 ((List.range 8).filter (fun i => pos.val < i)) ≤ if pos.val = 7 then 0 else 6 - pos.val := by
    cases h : ((List.range 8).filter (fun i => pos.val < i)).isEmpty with
    | true => simp [h] at h1; omega
    | false =>
      have hne : ((List.range 8).filter (fun i => pos.val < i)).length ≠ 0 := by
        intro hz
        have := List.eq_nil_of_length_eq_zero hz
        simp [this] at h
      simp [h, hlen1] at h1
      split <;> omega
  have hb2 : ray line 0 (List.range pos.val).reverse ≤ if pos.val = 0 then 0 else pos.val - 1 := by
    cases h : (List.range pos.val).reverse.isEmpty with
    | true => simp [h] at h2; omega
    | false =>
      have hne : (List.range pos.val).reverse.length ≠ 0 := by
        intro hz
        have := List.eq_nil_of_length_eq_zero hz
        simp [this] at h
      simp [h, hlen2] at h2
      split <;> omega
  by_cases h0 : pos.val = 0
  · rw [ite_eq_left h0] at hb2
    have hne7 : pos.val ≠ 7 := by omega
    rw [ite_eq_right hne7] at hb1
    omega
  · by_cases h7 : pos.val = 7
    · rw [ite_eq_left h7] at hb1
      rw [ite_eq_right h0] at hb2
      omega
    · rw [ite_eq_right h7] at hb1
      rw [ite_eq_right h0] at hb2
      omega

theorem countFlip_correct (pos : Fin 8) (line : Byte) :
    (countFlip pos line).toNat = 2 * reference pos line := by
  let a : Byte := line >>> (pos.val + 1)
  let b : Byte := line &&& (((1 : Byte) <<< pos.val) - 1)
  have h_above : (if a = 0 then 0 else fastCtz8 a) =
      ray line 0 ((List.range 8).filter (fun i => pos.val < i)) := by
    have hraw := fastCtz8Aux_eq_ray line pos.val pos.isLt 7 (by omega)
    have hfilt : (List.range 8).filter (fun i => pos.val + 1 + (7 - 7) ≤ i) =
        (List.range 8).filter (fun i => pos.val < i) := by
      apply congrArg (fun f => (List.range 8).filter f)
      funext i
      exact decide_eq_decide.mpr (by omega)
    rw [hfilt, Nat.sub_self] at hraw
    by_cases ha : a = 0
    · have h8 : fastCtz8Aux (line >>> (pos.val + 1)) 7 = 8 := (fastCtz8_eq_eight a).mpr ha
      rw [ite_eq_left ha]
      rw [ite_eq_left h8] at hraw
      exact hraw
    · have hne : fastCtz8Aux (line >>> (pos.val + 1)) 7 ≠ 8 := fun h => ha ((fastCtz8_eq_eight a).mp h)
      rw [ite_eq_right ha]
      rw [ite_eq_right hne] at hraw
      exact hraw
  have h_below_bnd : b ≠ 0 → 1 ≤ pos.val ∧ fastClz8 b ≤ 7 ∧ 7 - fastClz8 b < pos.val := by
    intro hb
    have hclz7 := fastClz8_lt_eight b hb
    have hpos1 : 1 ≤ pos.val := by
      apply Classical.byContradiction
      intro hlt
      have hp0 : pos.val = 0 := by omega
      apply hb
      simp [b, hp0]
    have hskip := fastClz8_below_skip line pos 7 (by omega) (by omega)
    have hpne : pos.val ≠ 0 := by omega
    simp only [hpne, ↓reduceIte] at hskip
    have hclz_eq : fastClz8 b = fastClz8Aux b (pos.val - 1) := hskip
    rcases fastClz8Aux_bound b (pos.val - 1) with h8 | ⟨hlo, _⟩
    · rw [← hclz_eq] at h8; omega
    · rw [← hclz_eq] at hlo
      exact ⟨hpos1, hclz7, by omega⟩
  have h_below : (if b = 0 then 0 else pos.val - 1 - (7 - fastClz8 b)) =
      ray line 0 (List.range pos.val).reverse := by
    by_cases hp0 : pos.val = 0
    · have hb0 : b = 0 := by simp [b, hp0]
      simp [hb0, hp0, ray]
    · have hskip := fastClz8_below_skip line pos 7 (by omega) (by omega)
      simp only [hp0, ↓reduceIte] at hskip
      have hraw := fastClz8Aux_eq_ray line pos (pos.val - 1) (by omega)
      dsimp only at hraw
      have hsucc : pos.val - 1 + 1 = pos.val := by omega
      have hzero : pos.val - 1 - (pos.val - 1) = 0 := by omega
      rw [hsucc, hzero] at hraw
      by_cases hb : b = 0
      · have h8 : fastClz8Aux (line &&& (((1 : Byte) <<< pos.val) - 1)) (pos.val - 1) = 8 := by
          rw [← hskip]
          exact (fastClz8Aux_eq_eight b 7).mpr (fun k _ => by rw [hb]; exact BitVec.getLsbD_zero)
        rw [ite_eq_left hb]
        rw [ite_eq_left h8] at hraw
        exact hraw
      · have hle := fastClz8_lt_eight b hb
        have hne : fastClz8Aux (line &&& (((1 : Byte) <<< pos.val) - 1)) (pos.val - 1) ≠ 8 := by
          rw [← hskip]; change fastClz8Aux b 7 ≠ 8; unfold fastClz8 at hle; omega
        rw [ite_eq_right hb, fastClz8]
        change pos.val - 1 - (7 - fastClz8Aux (line &&& (((1 : Byte) <<< pos.val) - 1)) 7) = _
        rw [hskip]
        rw [ite_eq_right hne] at hraw
        exact hraw
  unfold countFlip
  dsimp only
  rw [above_eq_setWidth, below_eq_setWidth]
  have h1 : (if a.setWidth 64 = (0 : Usize) then (0 : Byte) else 2 * (a.setWidth 64).ctz.setWidth 8).toNat =
      if a = 0 then 0 else 2 * fastCtz8 a := by
    by_cases ha : a = 0
    · have ha64 : a.setWidth 64 = (0 : Usize) := (setWidth64_eq_zero_iff a).mpr ha
      rw [ite_eq_left ha64, ite_eq_left ha]; rfl
    · have ha64 : a.setWidth 64 ≠ (0 : Usize) := fun h => ha ((setWidth64_eq_zero_iff a).mp h)
      have hctz := setWidth64_ctz_toNat a ha
      have hle : fastCtz8 a ≤ 6 := fastCtz8_above_le pos line ha
      rw [ite_eq_right ha64, ite_eq_right ha, BitVec.toNat_mul, BitVec.toNat_setWidth, hctz]
      have h2 : (2 : Byte).toNat = 2 := rfl
      rw [h2]
      omega
  have h2 : (if b.setWidth 64 = (0 : Usize) then (0 : Byte) else
      2 * ((BitVec.ofNat 64 pos.val - 1 - (63 - (b.setWidth 64).clz)).setWidth 8)).toNat =
      if b = 0 then 0 else 2 * (pos.val - 1 - (7 - fastClz8 b)) := by
    by_cases hb : b = 0
    · have hb64 : b.setWidth 64 = (0 : Usize) := (setWidth64_eq_zero_iff b).mpr hb
      rw [ite_eq_left hb64, ite_eq_left hb]; rfl
    · have hb64 : b.setWidth 64 ≠ (0 : Usize) := fun h => hb ((setWidth64_eq_zero_iff b).mp h)
      have hclz := setWidth64_clz_toNat b hb
      have ⟨hpos1, hclz7, hlt⟩ := h_below_bnd hb
      have hpos8 := pos.isLt
      rw [ite_eq_right hb64, ite_eq_right hb, BitVec.toNat_mul, BitVec.toNat_setWidth,
        BitVec.toNat_sub, BitVec.toNat_sub, BitVec.toNat_sub, hclz]
      have h2b : (2 : Byte).toNat = 2 := rfl
      have h1u : (1 : Usize).toNat = 1 := rfl
      have h63u : (63 : Usize).toNat = 63 := rfl
      rw [h2b, h1u, h63u, BitVec.toNat_ofNat]
      omega
  have hsum : (if a = 0 then 0 else 2 * fastCtz8 a) +
      (if b = 0 then 0 else 2 * (pos.val - 1 - (7 - fastClz8 b))) = 2 * reference pos line := by
    unfold reference
    rw [← h_above, ← h_below]
    split <;> split <;> omega
  rw [BitVec.toNat_add, h1, h2, hsum]
  have hrb := reference_bound pos line
  omega

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

theorem reference_toggle_square (pos : Fin 8) (line : Byte) :
    reference pos (line ^^^ ((1 : Byte) <<< pos.val)) = reference pos line := by
  have hbit : ∀ i, i < 8 → i ≠ pos.val →
      (line ^^^ ((1 : Byte) <<< pos.val)).getLsbD i = line.getLsbD i := by
    intro i hi hne
    rw [BitVec.getLsbD_xor, BitVec.getLsbD_shiftLeft]
    by_cases hlt : i < pos.val
    · simp [hi, hlt]
    · have hsub : i - pos.val ≠ 0 := by omega
      simp [hi, hlt, BitVec.getLsbD_one, hsub]
  unfold reference
  congr 1
  · apply ray_congr
    intro i hi
    simp only [List.mem_filter, List.mem_range, decide_eq_true_eq] at hi
    exact hbit i hi.1 (by omega)
  · apply ray_congr
    intro i hi
    simp only [List.mem_reverse, List.mem_range] at hi
    have hpos := pos.isLt
    exact hbit i (by omega) (by omega)

theorem countFlip_toggle_square (pos : Fin 8) (line : Byte) :
    countFlip pos (line ^^^ ((1 : Byte) <<< pos.val)) = countFlip pos line := by
  simp only [countFlip_eq_ofNat, reference_toggle_square]

end NeuralReversi.Solve1.LineTable

module
public import NeuralReversi.Stability.EdgeCapture

@[expose] public section

namespace NeuralReversi.Stability.Edge
open Semantics

def MoveProgress (p o : Byte) (x : Fin 8) : Prop :=
  p &&& o = 0 → emptyAt p o x = true →
  let a := applyEdgeMove x p o
  a.1 &&& a.2 = 0 ∧ (a.1 ||| a.2).cpop.toNat = (p ||| o).cpop.toNat + 1

def MoveCorrect (p o : Byte) (x : Fin 8) : Prop :=
  p &&& o = 0 → emptyAt p o x = true →
  let a := applyEdgeMove x p o
  a.1 &&& a.2 = 0 ∧ (a.1 ||| a.2).cpop.toNat = (p ||| o).cpop.toNat + 1 ∧
    a = moveSpec p o x

private theorem and_bit8_bne_zero (v : Byte) (k : Nat) (hk : k < 8) :
    (v &&& bit8 k != (0 : Byte)) = v.getLsbD k := by
  have hbit : ∀ i, (v &&& bit8 k).getLsbD i = (v.getLsbD k && decide (i = k)) := by
    intro i
    by_cases hi : i < 8
    · rw [BitVec.getLsbD_and, bit8_bit ⟨k, hk⟩ ⟨i, hi⟩]
      by_cases heq : i = k
      · subst heq; simp
      · have hne : (⟨i, hi⟩ : Fin 8) ≠ ⟨k, hk⟩ := fun h => heq (congrArg Fin.val h)
        simp [heq, hne]
    · rw [BitVec.getLsbD_of_ge _ _ (by omega)]
      have hne : i ≠ k := by omega
      simp [hne]
  cases hv : v.getLsbD k with
  | false =>
    have hz : v &&& bit8 k = (0 : Byte) := by
      apply BitVec.eq_of_getLsbD_eq
      intro i _
      simp [hbit i, hv]
    simp [hz]
  | true =>
    have hnz : v &&& bit8 k ≠ (0 : Byte) := by
      intro h
      have := congrArg (fun w : Byte => w.getLsbD k) h
      simp [hbit k, hv] at this
    exact bne_iff_ne.mpr hnz

private theorem scanLeft_spec (o : Byte) : ∀ k : Nat, k < 8 →
    scanLeft o k ≤ k ∧
    (∀ y, scanLeft o k < y → y ≤ k → o.getLsbD y = true) ∧
    (scanLeft o k = 0 ∨ o.getLsbD (scanLeft o k) = false) := by
  intro k hk
  induction k with
  | zero => exact ⟨Nat.le_refl 0, fun y h1 h2 => by omega, Or.inl rfl⟩
  | succ k ih =>
    simp only [scanLeft, and_bit8_bne_zero o (k + 1) hk]
    cases ho : o.getLsbD (k + 1) with
    | false =>
      simp only [Bool.false_eq_true, ↓reduceIte]
      exact ⟨Nat.le_refl _, fun y h1 h2 => by omega, Or.inr ho⟩
    | true =>
      simp only [↓reduceIte]
      obtain ⟨hle, hrun, hstop⟩ := ih (by omega)
      refine ⟨by omega, ?_, hstop⟩
      intro y h1 h2
      by_cases hy : y = k + 1
      · subst hy; exact ho
      · exact hrun y h1 (by omega)

private theorem flipLeft_bits (p o : Byte) : ∀ k : Nat, k < 8 → ∀ i : Nat, i < 8 →
    (flipLeft p o k).1.getLsbD i = (p.getLsbD i ^^ decide (scanLeft o k < i ∧ i ≤ k)) ∧
    (flipLeft p o k).2.getLsbD i = (o.getLsbD i ^^ decide (scanLeft o k < i ∧ i ≤ k)) := by
  intro k hk
  induction k generalizing p o with
  | zero =>
    intro i _
    have hnot : ¬(0 < i ∧ i ≤ 0) := by omega
    simp only [flipLeft, scanLeft, hnot, decide_false, Bool.xor_false, and_self]
  | succ k ih =>
    intro i hi
    simp only [flipLeft, scanLeft, and_bit8_bne_zero o (k + 1) hk]
    cases ho : o.getLsbD (k + 1) with
    | false =>
      have hnot : ¬(k + 1 < i ∧ i ≤ k + 1) := by omega
      simp [hnot]
    | true =>
      simp only [↓reduceIte]
      have h_o_eq : ∀ y, y ≤ k → (o ^^^ bit8 (k + 1)).getLsbD y = o.getLsbD y := by
        intro y hy
        rw [BitVec.getLsbD_xor, bit8_bit ⟨k + 1, hk⟩ ⟨y, by omega⟩]
        have hne : (⟨y, by omega⟩ : Fin 8) ≠ ⟨k + 1, hk⟩ := fun h => by
          have := congrArg Fin.val h; dsimp only at this; omega
        simp [hne]
      have hscan_eq : scanLeft (o ^^^ bit8 (k + 1)) k = scanLeft o k := by
        have gen : ∀ m, m ≤ k → scanLeft (o ^^^ bit8 (k + 1)) m = scanLeft o m := by
          intro m hm
          induction m with
          | zero => rfl
          | succ m ihm =>
            simp only [scanLeft, and_bit8_bne_zero _ (m + 1) (by omega), h_o_eq (m + 1) hm]
            cases o.getLsbD (m + 1) <;> simp [ihm (by omega)]
        exact gen k (Nat.le_refl k)
      have ⟨h1, h2⟩ := ih (p ^^^ bit8 (k + 1)) (o ^^^ bit8 (k + 1)) (by omega) i hi
      rw [hscan_eq] at h1 h2
      have hle := (scanLeft_spec o k (by omega)).1
      rw [h1, h2, BitVec.getLsbD_xor, BitVec.getLsbD_xor, bit8_bit ⟨k + 1, hk⟩ ⟨i, hi⟩]
      by_cases heq : i = k + 1
      · subst heq
        have hlt : scanLeft o k < k + 1 := by omega
        have hnot : ¬(k + 1 ≤ k) := by omega
        simp [hlt, hnot]
      · have hne : (⟨i, hi⟩ : Fin 8) ≠ ⟨k + 1, hk⟩ := fun h => heq (congrArg Fin.val h)
        have hiff : (scanLeft o k < i ∧ i ≤ k) ↔ (scanLeft o k < i ∧ i ≤ k + 1) := by omega
        simp [hne, hiff]

private theorem scanRight_spec (o : Byte) : ∀ fuel y : Nat, y + fuel = 8 →
    y ≤ scanRight o y fuel ∧ scanRight o y fuel ≤ 8 ∧
    (∀ z, y ≤ z → z < scanRight o y fuel → o.getLsbD z = true) ∧
    (scanRight o y fuel = 8 ∨ o.getLsbD (scanRight o y fuel) = false) := by
  intro fuel
  induction fuel with
  | zero =>
    intro y hy
    simp only [scanRight]
    exact ⟨by omega, by omega, fun z h1 h2 => by omega, Or.inl (by omega)⟩
  | succ fuel ih =>
    intro y hy
    simp only [scanRight, and_bit8_bne_zero o y (by omega)]
    cases ho : o.getLsbD y with
    | false =>
      simp only [Bool.false_eq_true, ↓reduceIte]
      exact ⟨Nat.le_refl _, by omega, fun z h1 h2 => by omega, Or.inr ho⟩
    | true =>
      simp only [↓reduceIte]
      obtain ⟨hge, hle, hrun, hstop⟩ := ih (y + 1) (by omega)
      refine ⟨by omega, hle, ?_, hstop⟩
      intro z h1 h2
      by_cases hz : z = y
      · subst hz; exact ho
      · exact hrun z (by omega) h2

private theorem flipRight_bits (p o : Byte) : ∀ fuel y : Nat, y + fuel = 8 → ∀ i : Nat, i < 8 →
    (flipRight p o y fuel).1.getLsbD i = (p.getLsbD i ^^ decide (y ≤ i ∧ i < scanRight o y fuel)) ∧
    (flipRight p o y fuel).2.getLsbD i = (o.getLsbD i ^^ decide (y ≤ i ∧ i < scanRight o y fuel)) := by
  intro fuel
  induction fuel generalizing p o with
  | zero =>
    intro y _ i _
    simp only [flipRight, scanRight]
    have hnot : ¬(y ≤ i ∧ i < y) := by omega
    simp [hnot]
  | succ fuel ih =>
    intro y hy i hi
    simp only [flipRight, scanRight, and_bit8_bne_zero o y (by omega)]
    cases ho : o.getLsbD y with
    | false =>
      have hnot : ¬(y ≤ i ∧ i < y) := by omega
      simp [hnot]
    | true =>
      simp only [↓reduceIte]
      have h_o_eq : ∀ z, y + 1 ≤ z → z < 8 → (o ^^^ bit8 y).getLsbD z = o.getLsbD z := by
        intro z hz1 hz2
        rw [BitVec.getLsbD_xor, bit8_bit ⟨y, by omega⟩ ⟨z, hz2⟩]
        have hne : (⟨z, hz2⟩ : Fin 8) ≠ ⟨y, by omega⟩ := fun h => by
          have := congrArg Fin.val h; dsimp only at this; omega
        simp [hne]
      have hscan_eq : scanRight (o ^^^ bit8 y) (y + 1) fuel = scanRight o (y + 1) fuel := by
        have gen : ∀ f w, y + 1 ≤ w → w + f = 8 →
            scanRight (o ^^^ bit8 y) w f = scanRight o w f := by
          intro f
          induction f with
          | zero => intro _ _ _; rfl
          | succ f ihf =>
            intro w hw1 hw2
            simp only [scanRight, and_bit8_bne_zero _ w (by omega), h_o_eq w hw1 (by omega)]
            cases o.getLsbD w <;> simp [ihf (w + 1) (by omega) (by omega)]
        exact gen fuel (y + 1) (Nat.le_refl _) (by omega)
      have ⟨h1, h2⟩ := ih (p ^^^ bit8 y) (o ^^^ bit8 y) (y + 1) (by omega) i hi
      rw [hscan_eq] at h1 h2
      have hge := (scanRight_spec o fuel (y + 1) (by omega)).1
      rw [h1, h2, BitVec.getLsbD_xor, BitVec.getLsbD_xor, bit8_bit ⟨y, by omega⟩ ⟨i, hi⟩]
      by_cases heq : i = y
      · subst heq
        have hlt : i < scanRight o (i + 1) fuel := by omega
        have hnot : ¬(i + 1 ≤ i) := by omega
        simp [hlt, hnot]
      · have hne : (⟨i, hi⟩ : Fin 8) ≠ ⟨y, by omega⟩ := fun h => heq (congrArg Fin.val h)
        have hiff : (y + 1 ≤ i ∧ i < scanRight o (y + 1) fuel) ↔ (y ≤ i ∧ i < scanRight o (y + 1) fuel) := by omega
        simp [hne, hiff]

private theorem cpop_or_bit8 : ∀ m : Fin 256, ∀ x : Fin 8,
    BitVec.ofNat 8 m.val &&& bit8 x.val = (0 : Byte) →
    (BitVec.ofNat 8 m.val ||| bit8 x.val).cpop.toNat = (BitVec.ofNat 8 m.val).cpop.toNat + 1 := by
  decide +kernel

theorem moveCorrect_all (p o : Byte) (x : Fin 8) : MoveCorrect p o x := by
  intro hdisj hempty
  have hdisj_bit : ∀ k, p.getLsbD k = true → o.getLsbD k = false := by
    intro k hk
    have h0 := congrArg (fun v : Byte => v.getLsbD k) hdisj
    simp [BitVec.getLsbD_and, hk] at h0
    exact h0
  let p1 := p ||| bit8 x.val
  have hp1_bit : ∀ k, k < 8 → p1.getLsbD k = (p.getLsbD k || decide (k = x.val)) := by
    intro k hk
    simp only [p1, BitVec.getLsbD_or, bit8_bit x ⟨k, hk⟩]
    by_cases h : k = x.val
    · subst h; simp
    · have hne : (⟨k, hk⟩ : Fin 8) ≠ x := fun e => h (congrArg Fin.val e)
      simp [h, hne]
  let yL := scanLeft o (x.val - 1)
  let doLeft : Bool := decide (x.val > 1) && p.getLsbD yL
  let left := if x.val > 1 then
    if p1 &&& bit8 yL != (0 : Byte) then flipLeft p1 o (x.val - 1) else (p1, o)
    else (p1, o)
  have hleft_bits : ∀ i, i < 8 →
      left.1.getLsbD i = (p1.getLsbD i ^^ (doLeft && decide (yL < i ∧ i < x.val))) ∧
      left.2.getLsbD i = (o.getLsbD i ^^ (doLeft && decide (yL < i ∧ i < x.val))) := by
    intro i hi
    by_cases hx : x.val > 1
    · have hyL_le := (scanLeft_spec o (x.val - 1) (by omega)).1
      have hcond : (p1 &&& bit8 yL != (0 : Byte)) = p.getLsbD yL := by
        rw [and_bit8_bne_zero p1 yL (by omega), hp1_bit yL (by omega)]
        have hne : yL ≠ x.val := by omega
        simp [hne]
      simp only [left, hx, ↓reduceIte, hcond, doLeft, decide_true, Bool.true_and]
      cases hpL : p.getLsbD yL with
      | false => simp
      | true =>
        simp only [↓reduceIte, Bool.true_and]
        have ⟨h1, h2⟩ := flipLeft_bits p1 o (x.val - 1) (by omega) i hi
        have hiff : (yL < i ∧ i ≤ x.val - 1) ↔ (yL < i ∧ i < x.val) := by omega
        simpa only [yL, hiff] using And.intro h1 h2
    · simp [left, hx, doLeft]
  have hleft_gt : ∀ i, x.val < i → i < 8 → left.1.getLsbD i = p.getLsbD i ∧ left.2.getLsbD i = o.getLsbD i := by
    intro i h1 h2
    have ⟨hl1, hl2⟩ := hleft_bits i h2
    have hnot : ¬(yL < i ∧ i < x.val) := by omega
    have hne : i ≠ x.val := by omega
    simp [hl1, hl2, hnot, hp1_bit i h2, hne]
  have hscanR_eq : x.val < 6 →
      scanRight left.2 (x.val + 1) (8 - (x.val + 1)) = scanRight o (x.val + 1) (8 - (x.val + 1)) := by
    intro hx
    have gen : ∀ f w, x.val + 1 ≤ w → w + f = 8 → scanRight left.2 w f = scanRight o w f := by
      intro f
      induction f with
      | zero => intro _ _ _; rfl
      | succ f ihf =>
        intro w hw1 hw2
        simp only [scanRight, and_bit8_bne_zero _ w (by omega), (hleft_gt w (by omega) (by omega)).2]
        cases o.getLsbD w <;> simp [ihf (w + 1) (by omega) (by omega)]
    exact gen (8 - (x.val + 1)) (x.val + 1) (Nat.le_refl _) (by omega)
  let yR := scanRight o (x.val + 1) (8 - (x.val + 1))
  let doRight : Bool := decide (x.val < 6) && decide (yR < 8) && p.getLsbD yR
  have happly_bits : ∀ i, i < 8 →
      (applyEdgeMove x p o).1.getLsbD i =
        (p1.getLsbD i ^^ (doLeft && decide (yL < i ∧ i < x.val)) ^^
          (doRight && decide (x.val < i ∧ i < yR))) ∧
      (applyEdgeMove x p o).2.getLsbD i =
        (o.getLsbD i ^^ (doLeft && decide (yL < i ∧ i < x.val)) ^^
          (doRight && decide (x.val < i ∧ i < yR))) := by
    intro i hi
    have happly_def : applyEdgeMove x p o =
        if x.val < 6 then
          let y := scanRight left.2 (x.val + 1) (8 - (x.val + 1))
          if y < 8 && (left.1 &&& bit8 y != (0 : Byte)) then
            flipRight left.1 left.2 (x.val + 1) (8 - (x.val + 1)) else left
        else left := rfl
    rw [happly_def]
    have ⟨hl1, hl2⟩ := hleft_bits i hi
    by_cases hx : x.val < 6
    · have hsR := hscanR_eq hx
      have ⟨hyR_ge, _, _, _⟩ := scanRight_spec o (8 - (x.val + 1)) (x.val + 1) (by omega)
      simp only [hx, ↓reduceIte, hsR]
      by_cases hyR : yR < 8
      · have hcond : (left.1 &&& bit8 yR != (0 : Byte)) = p.getLsbD yR := by
          rw [and_bit8_bne_zero left.1 yR hyR, (hleft_gt yR (by omega) hyR).1]
        have hdoR : doRight = p.getLsbD yR := by simp [doRight, hx, hyR]
        change (if yR < 8 && (left.1 &&& bit8 yR != (0 : Byte)) then
          flipRight left.1 left.2 (x.val + 1) (8 - (x.val + 1)) else left).1.getLsbD i = _ ∧
          (if yR < 8 && (left.1 &&& bit8 yR != (0 : Byte)) then
          flipRight left.1 left.2 (x.val + 1) (8 - (x.val + 1)) else left).2.getLsbD i = _
        simp only [hyR, decide_true, Bool.true_and, hcond, hdoR]
        cases hpR : p.getLsbD yR with
        | false =>
          exact ⟨by simpa using hl1, by simpa using hl2⟩
        | true =>
          simp only [↓reduceIte, Bool.true_and]
          have ⟨fr1, fr2⟩ := flipRight_bits left.1 left.2 (8 - (x.val + 1)) (x.val + 1) (by omega) i hi
          rw [hsR] at fr1 fr2
          rw [hl1] at fr1
          rw [hl2] at fr2
          have hiff : (x.val + 1 ≤ i ∧ i < yR) ↔ (x.val < i ∧ i < yR) := by omega
          simpa only [yR, hiff] using And.intro fr1 fr2
      · have hdoR : doRight = false := by simp [doRight, hyR]
        change (if yR < 8 && (left.1 &&& bit8 yR != (0 : Byte)) then _ else left).1.getLsbD i = _ ∧
          (if yR < 8 && (left.1 &&& bit8 yR != (0 : Byte)) then _ else left).2.getLsbD i = _
        simp only [hyR, decide_false, Bool.false_and, Bool.false_eq_true, ↓reduceIte, hdoR, Bool.xor_false]
        exact ⟨hl1, hl2⟩
    · have hdoR : doRight = false := by simp [doRight, hx]
      simp only [hx, ↓reduceIte, hdoR, Bool.false_and, Bool.xor_false]
      exact ⟨hl1, hl2⟩
  have hflip_eq : ∀ i : Fin 8,
      ((doLeft && decide (yL < i.val ∧ i.val < x.val)) ||
       (doRight && decide (x.val < i.val ∧ i.val < yR))) = captureSpec p o x i := by
    intro i
    apply Bool.eq_iff_iff.mpr
    simp only [captureSpec, decide_eq_true_eq, BracketBits, Bool.or_eq_true, Bool.and_eq_true,
      doLeft, doRight]
    constructor
    · rintro (⟨⟨hx, hpL⟩, hlo, hhi⟩ | ⟨⟨⟨hx, hyR⟩, hpR⟩, hlo, hhi⟩)
      · have ⟨hyL_le, hyL_run, _⟩ := scanLeft_spec o (x.val - 1) (by omega)
        refine ⟨⟨yL, by omega⟩, hpL, Or.inr ⟨hlo, hhi, fun y hy1 hy2 => hyL_run y.val hy1 (by omega)⟩⟩
      · have ⟨_, _, hyR_run, _⟩ := scanRight_spec o (8 - (x.val + 1)) (x.val + 1) (by omega)
        refine ⟨⟨yR, hyR⟩, hpR, Or.inl ⟨hlo, hhi, fun y hy1 hy2 => hyR_run y.val (by omega) hy2⟩⟩
    · rintro ⟨z, hpz, ⟨hlo, hhi, hrun⟩ | ⟨hlo, hhi, hrun⟩⟩
      · right
        have hx : x.val < 6 := by omega
        have ⟨hyR_ge, hyR_le, hyR_run, hyR_stop⟩ := scanRight_spec o (8 - (x.val + 1)) (x.val + 1) (by omega)
        have hoz : o.getLsbD z.val = false := hdisj_bit z.val hpz
        have hz_eq : yR = z.val := by
          by_cases hlt : yR < z.val
          · rcases hyR_stop with h8 | hfalse
            · omega
            · have htrue := hrun ⟨yR, by omega⟩ (by dsimp only; omega) (by dsimp only; omega)
              rw [hfalse] at htrue; exact Bool.noConfusion htrue
          · by_cases hgt : z.val < yR
            · have htrue := hyR_run z.val (by omega) hgt
              rw [hoz] at htrue; exact Bool.noConfusion htrue
            · omega
        refine ⟨⟨⟨hx, by omega⟩, by simpa [hz_eq] using hpz⟩, hlo, by omega⟩
      · left
        have hx : x.val > 1 := by omega
        have ⟨hyL_le, hyL_run, hyL_stop⟩ := scanLeft_spec o (x.val - 1) (by omega)
        have hoz : o.getLsbD z.val = false := hdisj_bit z.val hpz
        have hz_eq : yL = z.val := by
          by_cases hgt : z.val < yL
          · rcases hyL_stop with h0 | hfalse
            · omega
            · have htrue := hrun ⟨yL, by omega⟩ (by dsimp only; omega) (by dsimp only; omega)
              rw [hfalse] at htrue; exact Bool.noConfusion htrue
          · by_cases hlt : yL < z.val
            · have htrue := hyL_run z.val hlt (by omega)
              rw [hoz] at htrue; exact Bool.noConfusion htrue
            · omega
        refine ⟨⟨hx, by simpa [hz_eq] using hpz⟩, by omega, hhi⟩
  have heq : applyEdgeMove x p o = moveSpec p o x := by
    apply Prod.ext
    · apply BitVec.eq_of_getLsbD_eq
      intro i hi
      have hb := (happly_bits i hi).1
      have hf := hflip_eq ⟨i, hi⟩
      have hdisj_lr : ¬((doLeft && decide (yL < i ∧ i < x.val)) = true ∧
          (doRight && decide (x.val < i ∧ i < yR)) = true) := by
        rintro ⟨h1, h2⟩
        simp only [Bool.and_eq_true, decide_eq_true_eq] at h1 h2
        omega
      rw [hb]
      simp only [moveSpec, BitVec.getLsbD_or, encodeBits_bit _ ⟨i, hi⟩, ← hf]
      cases hL : (doLeft && decide (yL < i ∧ i < x.val)) <;>
        cases hR : (doRight && decide (x.val < i ∧ i < yR))
      · simp [p1]
      · have hcap : captureSpec p o x ⟨i, hi⟩ = true := by rw [← hf, hL, hR]; rfl
        have hoi : o.getLsbD i = true := captureSpec_opponent p o x ⟨i, hi⟩ hcap
        have hpi : p.getLsbD i = false := by
          cases hp : p.getLsbD i
          · rfl
          · have := hdisj_bit i hp; simp [hoi] at this
        have hne : i ≠ x.val := by
          simp only [Bool.and_eq_true, decide_eq_true_eq] at hR; omega
        simp [hp1_bit i hi, hpi, hne]
      · have hcap : captureSpec p o x ⟨i, hi⟩ = true := by rw [← hf, hL, hR]; rfl
        have hoi : o.getLsbD i = true := captureSpec_opponent p o x ⟨i, hi⟩ hcap
        have hpi : p.getLsbD i = false := by
          cases hp : p.getLsbD i
          · rfl
          · have := hdisj_bit i hp; simp [hoi] at this
        have hne : i ≠ x.val := by
          simp only [Bool.and_eq_true, decide_eq_true_eq] at hL; omega
        simp [hp1_bit i hi, hpi, hne]
      · exact False.elim (hdisj_lr ⟨hL, hR⟩)
    · apply BitVec.eq_of_getLsbD_eq
      intro i hi
      have hb := (happly_bits i hi).2
      have hf := hflip_eq ⟨i, hi⟩
      have hdisj_lr : ¬((doLeft && decide (yL < i ∧ i < x.val)) = true ∧
          (doRight && decide (x.val < i ∧ i < yR)) = true) := by
        rintro ⟨h1, h2⟩
        simp only [Bool.and_eq_true, decide_eq_true_eq] at h1 h2
        omega
      rw [hb]
      simp only [moveSpec, BitVec.getLsbD_and, BitVec.getLsbD_not, hi, decide_true, Bool.true_and,
        encodeBits_bit _ ⟨i, hi⟩, ← hf]
      cases hL : (doLeft && decide (yL < i ∧ i < x.val)) <;>
        cases hR : (doRight && decide (x.val < i ∧ i < yR))
      · simp
      · have hcap : captureSpec p o x ⟨i, hi⟩ = true := by rw [← hf, hL, hR]; rfl
        have hoi : o.getLsbD i = true := captureSpec_opponent p o x ⟨i, hi⟩ hcap
        simp [hoi]
      · have hcap : captureSpec p o x ⟨i, hi⟩ = true := by rw [← hf, hL, hR]; rfl
        have hoi : o.getLsbD i = true := captureSpec_opponent p o x ⟨i, hi⟩ hcap
        simp [hoi]
      · exact False.elim (hdisj_lr ⟨hL, hR⟩)
  refine ⟨?_, ?_, heq⟩
  · rw [heq]; exact moveSpec_disjoint p o x hdisj hempty
  · rw [heq]
    have hor : (moveSpec p o x).1 ||| (moveSpec p o x).2 = (p ||| o) ||| bit8 x.val := by
      apply BitVec.eq_of_getLsbD_eq
      intro i hi
      simp only [moveSpec, BitVec.getLsbD_or, BitVec.getLsbD_and, BitVec.getLsbD_not, hi,
        decide_true, Bool.true_and, encodeBits_bit _ ⟨i, hi⟩]
      cases hc : captureSpec p o x ⟨i, hi⟩ with
      | false => simp [Bool.or_assoc, Bool.or_comm, Bool.or_left_comm]
      | true =>
        have hoi := captureSpec_opponent p o x ⟨i, hi⟩ hc
        simp [hoi]
    rw [hor]
    have hzero : (p ||| o) &&& bit8 x.val = (0 : Byte) := by simpa [emptyAt] using hempty
    have hpop := cpop_or_bit8 ⟨(p ||| o).toNat, (p ||| o).isLt⟩ x
    simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hpop
    exact hpop hzero

end NeuralReversi.Stability.Edge

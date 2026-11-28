module

@[expose] public section

namespace NeuralReversi.Solve2

def NWS (alpha value result : Int) : Prop :=
  (result ≤ alpha → value ≤ result) ∧ (alpha < result → result ≤ value)

def OneEmpty (alpha value result : Int) : Prop :=
  value ≤ result ∧ (alpha < result → result = value)

theorem negate_oneEmpty {alpha value result : Int}
    (h : OneEmpty (-(alpha + 1)) value result) : NWS alpha (-value) (-result) := by
  obtain ⟨upper, exactHigh⟩ := h
  constructor
  · intro low
    have eq := exactHigh (by omega)
    omega
  · intro high
    omega

theorem oneEmpty_nws {alpha value result : Int}
    (h : OneEmpty alpha value result) : NWS alpha value result := by
  exact ⟨fun _ => h.1, fun high => Int.le_of_eq (h.2 high)⟩

theorem nws_exact (alpha value : Int) : NWS alpha value value := by
  exact ⟨fun _ => Int.le_refl _, fun _ => Int.le_refl _⟩

def OptionalRel (R : Int → Int → Prop) : Option Int → Option Int → Prop
  | none, none => True
  | some v, some r => R v r
  | _, _ => False

def maxSearch (alpha : Int) (r1 r2 : Option Int) (fallback : Int) : Int :=
  match r1 with
  | some r => if alpha + 1 ≤ r then r else
      match r2 with | some q => max q r | none => r
  | none => match r2 with | some q => q | none => fallback

def minSearch (alpha : Int) (r1 r2 : Option Int) (fallback : Int) : Int :=
  match r1 with
  | some r => if r ≤ alpha then r else
      match r2 with | some q => min q r | none => r
  | none => match r2 with | some q => q | none => fallback

def maxValue (v1 v2 : Option Int) (fallback : Int) : Int :=
  match v1, v2 with
  | some v, some w => max w v
  | some v, none => v
  | none, some w => w
  | none, none => fallback

def minValue (v1 v2 : Option Int) (fallback : Int) : Int :=
  match v1, v2 with
  | some v, some w => min w v
  | some v, none => v
  | none, some w => w
  | none, none => fallback

theorem max_pair_nws {alpha v w r q : Int}
    (hr : NWS alpha v r) (hq : NWS alpha w q) :
    NWS alpha (max w v) (if alpha + 1 ≤ r then r else max q r) := by
  obtain ⟨rl,rh⟩ := hr
  obtain ⟨ql,qh⟩ := hq
  unfold NWS
  split
  · rename_i cut
    constructor
    · omega
    · intro _
      have hv := rh (by omega)
      have hm : v ≤ max w v := Int.le_max_right _ _
      omega
  · constructor
    · intro low
      have rv : r ≤ max q r := Int.le_max_right _ _
      have qv : q ≤ max q r := Int.le_max_left _ _
      have vl := rl (by omega)
      have wl := ql (by omega)
      exact Int.max_le.mpr ⟨by omega, by omega⟩
    · intro high
      by_cases h : q ≤ r
      · rw [Int.max_eq_right h] at high ⊢
        have hv := rh high
        exact Int.le_trans hv (Int.le_max_right _ _)
      · rw [Int.max_eq_left (by omega : r ≤ q)] at high ⊢
        have hw := qh high
        exact Int.le_trans hw (Int.le_max_left _ _)

theorem min_pair_nws {alpha v w r q : Int}
    (hr : NWS alpha v r) (hq : NWS alpha w q) :
    NWS alpha (min w v) (if r ≤ alpha then r else min q r) := by
  obtain ⟨rl,rh⟩ := hr
  obtain ⟨ql,qh⟩ := hq
  unfold NWS
  split
  · rename_i cut
    constructor
    · intro _
      exact Int.le_trans (Int.min_le_right _ _) (rl cut)
    · omega
  · constructor
    · intro low
      by_cases h : q ≤ r
      · rw [Int.min_eq_left h] at low ⊢
        exact Int.le_trans (Int.min_le_left _ _) (ql low)
      · rw [Int.min_eq_right (by omega : r ≤ q)] at low ⊢
        exact Int.le_trans (Int.min_le_right _ _) (rl low)
    · intro high
      have rv : min q r ≤ r := Int.min_le_right _ _
      have qv : min q r ≤ q := Int.min_le_left _ _
      have vh := rh (by omega)
      have wh := qh (by omega)
      exact Int.le_min.mpr ⟨by omega, by omega⟩

theorem maxSearch_nws {alpha fv fr : Int} {v1 v2 r1 r2 : Option Int}
    (h1 : OptionalRel (NWS alpha) v1 r1) (h2 : OptionalRel (NWS alpha) v2 r2)
    (fallback : NWS alpha fv fr) :
    NWS alpha (maxValue v1 v2 fv) (maxSearch alpha r1 r2 fr) := by
  cases v1 <;> cases r1 <;> cases v2 <;> cases r2 <;>
    simp only [OptionalRel] at h1 h2 <;> try contradiction
  · exact fallback
  · exact h2
  · simpa only [maxValue, maxSearch, ite_self] using h1
  · exact max_pair_nws h1 h2

theorem minSearch_nws {alpha fv fr : Int} {v1 v2 r1 r2 : Option Int}
    (h1 : OptionalRel (NWS alpha) v1 r1) (h2 : OptionalRel (NWS alpha) v2 r2)
    (fallback : NWS alpha fv fr) :
    NWS alpha (minValue v1 v2 fv) (minSearch alpha r1 r2 fr) := by
  cases v1 <;> cases r1 <;> cases v2 <;> cases r2 <;>
    simp only [OptionalRel] at h1 h2 <;> try contradiction
  · exact fallback
  · exact h2
  · simpa only [minValue, minSearch, ite_self] using h1
  · exact min_pair_nws h1 h2

def solve (alpha : Int) (p1 p2 o1 o2 : Option Int) (terminal : Int) : Int :=
  maxSearch alpha (p1.map (- ·)) (p2.map (- ·)) (minSearch alpha o1 o2 terminal)

def value (p1 p2 o1 o2 : Option Int) (terminal : Int) : Int :=
  maxValue (p1.map (- ·)) (p2.map (- ·)) (minValue o1 o2 terminal)

theorem optional_negate {alpha : Int} {v r : Option Int}
    (h : OptionalRel (OneEmpty (-(alpha + 1))) v r) :
    OptionalRel (NWS alpha) (v.map (- ·)) (r.map (- ·)) := by
  cases v <;> cases r <;> simp only [OptionalRel, Option.map] at * <;> try contradiction
  exact negate_oneEmpty h

theorem optional_oneEmpty {alpha : Int} {v r : Option Int}
    (h : OptionalRel (OneEmpty alpha) v r) : OptionalRel (NWS alpha) v r := by
  cases v <;> cases r <;> simp only [OptionalRel] at * <;> try contradiction
  exact oneEmpty_nws h

theorem solve_nws {alpha terminal : Int} {p1 p2 o1 o2 r1 r2 s1 s2 : Option Int}
    (hp1 : OptionalRel (OneEmpty (-(alpha + 1))) p1 r1)
    (hp2 : OptionalRel (OneEmpty (-(alpha + 1))) p2 r2)
    (ho1 : OptionalRel (OneEmpty alpha) o1 s1)
    (ho2 : OptionalRel (OneEmpty alpha) o2 s2) :
    NWS alpha (value p1 p2 o1 o2 terminal) (solve alpha r1 r2 s1 s2 terminal) :=
  maxSearch_nws (optional_negate hp1) (optional_negate hp2)
    (minSearch_nws (optional_oneEmpty ho1) (optional_oneEmpty ho2) (nws_exact _ _))

abbrev Score := BitVec 32

def childAlpha (alpha : Score) : Score := -(alpha + 1)

def maxWord (a b : Score) : Score := if a.toInt ≤ b.toInt then b else a
def minWord (a b : Score) : Score := if a.toInt ≤ b.toInt then a else b

def maxSearchWord (beta : Score) (r1 r2 : Option Score) (fallback : Score) : Score :=
  match r1 with
  | some r => if beta.toInt ≤ r.toInt then r else
      match r2 with | some q => maxWord q r | none => r
  | none => match r2 with | some q => q | none => fallback

def minSearchWord (alpha : Score) (r1 r2 : Option Score) (fallback : Score) : Score :=
  match r1 with
  | some r => if r.toInt ≤ alpha.toInt then r else
      match r2 with | some q => minWord q r | none => r
  | none => match r2 with | some q => q | none => fallback

def solveWord (alpha : Score) (p1 p2 o1 o2 : Option Score) (terminal : Score) : Score :=
  maxSearchWord (alpha + 1) (p1.map (- ·)) (p2.map (- ·))
    (minSearchWord alpha o1 o2 terminal)

theorem add_one_safe (alpha : Score) (safe : alpha.toInt < 2147483647) :
    (alpha + 1).toInt = alpha.toInt + 1 := by
  have lower := BitVec.le_toInt alpha
  rw [BitVec.toInt_add]
  change (alpha.toInt + 1).bmod 4294967296 = alpha.toInt + 1
  apply Int.bmod_eq_of_le <;> simp_all <;> omega

theorem neg_safe (score : Score) (safe : -2147483648 < score.toInt) :
    (-score).toInt = -score.toInt := by
  have upper := BitVec.toInt_lt (x := score)
  rw [BitVec.toInt_neg]
  apply Int.bmod_eq_of_le <;> simp_all <;> omega

theorem childAlpha_safe (alpha : Score) (safe : alpha.toInt < 2147483647) :
    (childAlpha alpha).toInt = -(alpha.toInt + 1) := by
  have lower := BitVec.le_toInt alpha
  have beta := add_one_safe alpha safe
  unfold childAlpha
  rw [neg_safe (alpha + 1) (by rw [beta]; simp_all; omega), beta]

theorem maxWord_toInt (a b : Score) : (maxWord a b).toInt = max a.toInt b.toInt := by
  unfold maxWord
  split <;> rename_i h
  · exact (Int.max_eq_right h).symm
  · exact (Int.max_eq_left (by omega)).symm

theorem minWord_toInt (a b : Score) : (minWord a b).toInt = min a.toInt b.toInt := by
  unfold minWord
  split <;> rename_i h
  · exact (Int.min_eq_left h).symm
  · exact (Int.min_eq_right (by omega)).symm

theorem maxSearchWord_toInt (alpha beta : Score) (r1 r2 : Option Score) (fallback : Score)
    (window : beta.toInt = alpha.toInt + 1) :
    (maxSearchWord beta r1 r2 fallback).toInt =
      maxSearch alpha.toInt (r1.map BitVec.toInt) (r2.map BitVec.toInt) fallback.toInt := by
  cases r1 <;> cases r2 <;> simp only [maxSearchWord, maxSearch, Option.map, window]
  all_goals split <;> simp only [maxWord_toInt]

theorem minSearchWord_toInt (alpha : Score) (r1 r2 : Option Score) (fallback : Score) :
    (minSearchWord alpha r1 r2 fallback).toInt =
      minSearch alpha.toInt (r1.map BitVec.toInt) (r2.map BitVec.toInt) fallback.toInt := by
  cases r1 <;> cases r2 <;> simp only [minSearchWord, minSearch, Option.map]
  all_goals split <;> simp only [minWord_toInt]

def NegSafe (x : Option Score) : Prop := ∀ s, x = some s → -2147483648 < s.toInt

theorem map_neg_toInt (x : Option Score) (safe : NegSafe x) :
    (x.map (- ·)).map BitVec.toInt = (x.map BitVec.toInt).map (- ·) := by
  cases x with
  | none => rfl
  | some s => simp only [Option.map, neg_safe s (safe s rfl)]

theorem solveWord_toInt (alpha terminal : Score) (p1 p2 o1 o2 : Option Score)
    (windowSafe : alpha.toInt < 2147483647) (safe1 : NegSafe p1) (safe2 : NegSafe p2) :
    (solveWord alpha p1 p2 o1 o2 terminal).toInt =
      solve alpha.toInt (p1.map BitVec.toInt) (p2.map BitVec.toInt)
        (o1.map BitVec.toInt) (o2.map BitVec.toInt) terminal.toInt := by
  unfold solveWord solve
  rw [maxSearchWord_toInt alpha _ _ _ _ (add_one_safe alpha windowSafe),
    minSearchWord_toInt, map_neg_toInt p1 safe1, map_neg_toInt p2 safe2]

theorem solveWord_nws {alpha terminal : Score} {p1 p2 o1 o2 : Option Score}
    {v1 v2 w1 w2 : Option Int}
    (windowSafe : alpha.toInt < 2147483647) (safe1 : NegSafe p1) (safe2 : NegSafe p2)
    (hp1 : OptionalRel (OneEmpty (childAlpha alpha).toInt) v1 (p1.map BitVec.toInt))
    (hp2 : OptionalRel (OneEmpty (childAlpha alpha).toInt) v2 (p2.map BitVec.toInt))
    (ho1 : OptionalRel (OneEmpty alpha.toInt) w1 (o1.map BitVec.toInt))
    (ho2 : OptionalRel (OneEmpty alpha.toInt) w2 (o2.map BitVec.toInt)) :
    NWS alpha.toInt (value v1 v2 w1 w2 terminal.toInt)
      (solveWord alpha p1 p2 o1 o2 terminal).toInt := by
  rw [solveWord_toInt alpha terminal p1 p2 o1 o2 windowSafe safe1 safe2]
  rw [childAlpha_safe alpha windowSafe] at hp1 hp2
  exact solve_nws hp1 hp2 ho1 ho2

theorem nws_classification {alpha value result : Int} (h : NWS alpha value result) :
    (result ≤ alpha ↔ value ≤ alpha) ∧ (alpha < result ↔ alpha < value) := by
  obtain ⟨low, high⟩ := h
  constructor <;> constructor <;> intro h <;> by_cases side : result ≤ alpha
  all_goals first | have hl := low side; omega | have hh := high (by omega); omega

theorem beta_no_signed_overflow (alpha : Score) (safe : alpha.toInt < 2147483647) :
    BitVec.saddOverflow alpha 1 = false := by
  have lower := BitVec.le_toInt alpha
  simp only [BitVec.saddOverflow, Bool.or_eq_false_iff,
    decide_eq_false_iff_not]
  constructor <;> simp_all <;> omega

theorem score_no_neg_overflow (score : Score) (safe : -2147483648 < score.toInt) :
    BitVec.negOverflow score = false := by
  simp only [BitVec.negOverflow, beq_eq_false_iff_ne]
  omega

theorem childAlpha_no_neg_overflow (alpha : Score) (safe : alpha.toInt < 2147483647) :
    BitVec.negOverflow (alpha + 1) = false := by
  apply score_no_neg_overflow
  rw [add_one_safe alpha safe]
  have lower := BitVec.le_toInt alpha
  simp_all
  omega

theorem control_first_cut_not_exact :
    solve 0 (some (-2)) (some (-10)) none none 0 = 2 ∧
    value (some (-2)) (some (-10)) none none 0 = 10 := by decide

theorem control_child_inexact :
    OneEmpty (-1) (-10) (-2) ∧ NWS 0 10 2 ∧ (2 : Int) ≠ 10 := by
  simp [OneEmpty, NWS]

theorem control_pass_cut_not_exact :
    solve 0 none none (some (-2)) (some (-10)) 0 = -2 ∧
    value none none (some (-2)) (some (-10)) 0 = -10 := by decide

theorem control_both_pass (alpha terminal : Int) :
    solve alpha none none none none terminal = terminal := rfl

theorem control_max_alpha_wrap :
    ((2147483647 : Score) + 1).toInt = -2147483648 ∧
    (childAlpha 2147483647).toInt = -2147483648 := by decide

theorem control_min_score_wrap :
    (-(2147483648 : Score)).toInt = -2147483648 := by decide

end NeuralReversi.Solve2

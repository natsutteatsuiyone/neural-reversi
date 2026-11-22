import NeuralReversi.Solve2.Control

namespace NeuralReversi.Solve34
open Solve2
abbrev Score := BitVec 32

inductive ListRel (R : α → β → Prop) : List α → List β → Prop
  | nil : ListRel R [] []
  | cons {a b as bs} : R a b → ListRel R as bs → ListRel R (a :: as) (b :: bs)

def inf : Int := 127

def mergeMax : Option Int → Option Int → Option Int
  | none, b => b
  | a, none => a
  | some a, some b => some (max b a)

def mergeMin : Option Int → Option Int → Option Int
  | none, b => b
  | a, none => a
  | some a, some b => some (min b a)

@[simp] theorem mergeMax_none (a : Option Int) : mergeMax a none = a := by cases a <;> rfl
@[simp] theorem mergeMin_none (a : Option Int) : mergeMin a none = a := by cases a <;> rfl

def foldMax (best : Option Int) : List (Option Int) → Option Int
  | [] => best
  | x :: xs => foldMax (mergeMax best x) xs

def foldMin (best : Option Int) : List (Option Int) → Option Int
  | [] => best
  | x :: xs => foldMin (mergeMin best x) xs

def maxRun (alpha : Int) (best : Option Int) : List (Option Int) → Option Int
  | [] => best
  | none :: xs => maxRun alpha best xs
  | some r :: xs => if alpha + 1 ≤ r then some r else maxRun alpha (mergeMax best (some r)) xs

def minRun (alpha : Int) (best : Option Int) : List (Option Int) → Option Int
  | [] => best
  | none :: xs => minRun alpha best xs
  | some r :: xs => if r ≤ alpha then some r else minRun alpha (mergeMin best (some r)) xs

def sourceMax (alpha : Int) (first : Bool) (best : Int)
    (xs : List (Option Int)) (fallback : Int) : Int :=
  match xs with
  | [] => if best = -inf then fallback else best
  | [some r] => max r best
  | none :: rest => sourceMax alpha false best rest fallback
  | some r :: rest => if alpha + 1 ≤ r then r else
      sourceMax alpha false (if first then r else max r best) rest fallback

def sourceMin (alpha : Int) (first : Bool) (best : Int)
    (xs : List (Option Int)) (fallback : Int) : Int :=
  match xs with
  | [] => if best = inf then fallback else best
  | [some r] => min r best
  | none :: rest => sourceMin alpha false best rest fallback
  | some r :: rest => if r ≤ alpha then r else
      sourceMin alpha false (if first then r else min r best) rest fallback

theorem negate_nws {alpha value result : Int} (h : NWS (-(alpha + 1)) value result) :
    NWS alpha (-value) (-result) := by
  obtain ⟨lo,hi⟩ := h
  constructor
  · intro h
    have hr := hi (by omega)
    omega
  · intro h
    have hr := lo (by omega)
    omega

theorem max_nws {alpha v w r q : Int} (hr : NWS alpha v r) (hq : NWS alpha w q) :
    NWS alpha (max w v) (max q r) := by
  obtain ⟨rl,rh⟩ := hr
  obtain ⟨ql,qh⟩ := hq
  constructor
  · intro low
    have rv := Int.le_max_right q r
    have qv := Int.le_max_left q r
    have vl := rl (by omega)
    have wl := ql (by omega)
    exact Int.max_le.mpr ⟨by omega, by omega⟩
  · intro high
    by_cases h : q ≤ r
    · rw [Int.max_eq_right h] at high ⊢
      exact Int.le_trans (rh high) (Int.le_max_right _ _)
    · rw [Int.max_eq_left (by omega : r ≤ q)] at high ⊢
      exact Int.le_trans (qh high) (Int.le_max_left _ _)

theorem min_nws {alpha v w r q : Int} (hr : NWS alpha v r) (hq : NWS alpha w q) :
    NWS alpha (min w v) (min q r) := by
  obtain ⟨rl,rh⟩ := hr
  obtain ⟨ql,qh⟩ := hq
  constructor
  · intro low
    by_cases h : q ≤ r
    · rw [Int.min_eq_left h] at low ⊢
      exact Int.le_trans (Int.min_le_left _ _) (ql low)
    · rw [Int.min_eq_right (by omega : r ≤ q)] at low ⊢
      exact Int.le_trans (Int.min_le_right _ _) (rl low)
  · intro high
    have rv := Int.min_le_right q r
    have qv := Int.min_le_left q r
    have vh := rh (by omega)
    have wh := qh (by omega)
    exact Int.le_min.mpr ⟨by omega, by omega⟩

theorem mergeMax_nws {alpha : Int} {v w r q : Option Int}
    (hr : OptionalRel (NWS alpha) v r) (hq : OptionalRel (NWS alpha) w q) :
    OptionalRel (NWS alpha) (mergeMax v w) (mergeMax r q) := by
  cases v <;> cases r <;> cases w <;> cases q <;>
    simp only [OptionalRel, mergeMax] at * <;> try contradiction <;> try trivial
  · exact hq
  · exact hr
  · exact max_nws hr hq

theorem mergeMin_nws {alpha : Int} {v w r q : Option Int}
    (hr : OptionalRel (NWS alpha) v r) (hq : OptionalRel (NWS alpha) w q) :
    OptionalRel (NWS alpha) (mergeMin v w) (mergeMin r q) := by
  cases v <;> cases r <;> cases w <;> cases q <;>
    simp only [OptionalRel, mergeMin] at * <;> try contradiction <;> try trivial
  · exact hq
  · exact hr
  · exact min_nws hr hq

theorem foldMax_ge (v : Int) (xs : List (Option Int)) :
    ∃ w, foldMax (some v) xs = some w ∧ v ≤ w := by
  induction xs generalizing v with
  | nil => exact ⟨v,rfl,Int.le_refl _⟩
  | cons x xs ih =>
    cases x with
    | none => exact ih v
    | some q =>
      obtain ⟨w,eq,le⟩ := ih (max q v)
      exact ⟨w,eq,Int.le_trans (Int.le_max_right _ _) le⟩

theorem foldMin_le (v : Int) (xs : List (Option Int)) :
    ∃ w, foldMin (some v) xs = some w ∧ w ≤ v := by
  induction xs generalizing v with
  | nil => exact ⟨v,rfl,Int.le_refl _⟩
  | cons x xs ih =>
    cases x with
    | none => exact ih v
    | some q =>
      obtain ⟨w,eq,le⟩ := ih (min q v)
      exact ⟨w,eq,Int.le_trans le (Int.min_le_right _ _)⟩

theorem max_cut_safe {alpha v r : Int} (best : Option Int) (xs : List (Option Int))
    (h : NWS alpha v r) (cut : alpha + 1 ≤ r) :
    OptionalRel (NWS alpha) (foldMax (mergeMax best (some v)) xs) (some r) := by
  have high := h.2 (by omega)
  cases best with
  | none =>
    obtain ⟨w,eq,le⟩ := foldMax_ge v xs
    rw [show mergeMax none (some v) = some v from rfl, eq]
    exact ⟨by omega, by omega⟩
  | some b =>
    obtain ⟨w,eq,le⟩ := foldMax_ge (max v b) xs
    rw [show mergeMax (some b) (some v) = some (max v b) from rfl, eq]
    have hv := Int.le_max_left v b
    exact ⟨by omega, by omega⟩

theorem min_cut_safe {alpha v r : Int} (best : Option Int) (xs : List (Option Int))
    (h : NWS alpha v r) (cut : r ≤ alpha) :
    OptionalRel (NWS alpha) (foldMin (mergeMin best (some v)) xs) (some r) := by
  have low := h.1 cut
  cases best with
  | none =>
    obtain ⟨w,eq,le⟩ := foldMin_le v xs
    rw [show mergeMin none (some v) = some v from rfl, eq]
    exact ⟨by omega, by omega⟩
  | some b =>
    obtain ⟨w,eq,le⟩ := foldMin_le (min v b) xs
    rw [show mergeMin (some b) (some v) = some (min v b) from rfl, eq]
    have hv := Int.min_le_left v b
    exact ⟨by omega, by omega⟩

theorem maxRun_nws {alpha : Int} {vb rb : Option Int} {vs rs : List (Option Int)}
    (heads : OptionalRel (NWS alpha) vb rb)
    (children : ListRel (OptionalRel (NWS alpha)) vs rs) :
    OptionalRel (NWS alpha) (foldMax vb vs) (maxRun alpha rb rs) := by
  induction children generalizing vb rb with
  | nil => exact heads
  | @cons v r vs rs hr rest ih =>
    cases v <;> cases r <;> simp only [OptionalRel] at hr <;> try contradiction
    · simpa only [foldMax, foldMin, maxRun, minRun, mergeMax_none, mergeMin_none] using ih heads
    · simp only [maxRun, foldMax]
      split
      · exact max_cut_safe vb vs hr (by assumption)
      · exact ih (mergeMax_nws heads hr)

theorem minRun_nws {alpha : Int} {vb rb : Option Int} {vs rs : List (Option Int)}
    (heads : OptionalRel (NWS alpha) vb rb)
    (children : ListRel (OptionalRel (NWS alpha)) vs rs) :
    OptionalRel (NWS alpha) (foldMin vb vs) (minRun alpha rb rs) := by
  induction children generalizing vb rb with
  | nil => exact heads
  | @cons v r vs rs hr rest ih =>
    cases v <;> cases r <;> simp only [OptionalRel] at hr <;> try contradiction
    · simpa only [foldMax, foldMin, maxRun, minRun, mergeMax_none, mergeMin_none] using ih heads
    · simp only [minRun, foldMin]
      split
      · exact min_cut_safe vb vs hr (by assumption)
      · exact ih (mergeMin_nws heads hr)

def Within (x : Option Int) : Prop := ∀ r, x = some r → -inf < r ∧ r < inf
def AllWithin (xs : List (Option Int)) : Prop := ∀ x ∈ xs, Within x
def Below (alpha : Int) (x : Option Int) : Prop := ∀ r, x = some r → r ≤ alpha
def Above (alpha : Int) (x : Option Int) : Prop := ∀ r, x = some r → alpha < r

theorem mergeMax_within {a b : Option Int} (ha : Within a) (hb : Within b) :
    Within (mergeMax a b) := by
  cases a <;> cases b <;> simp only [mergeMax]
  · exact hb
  · exact hb
  · exact ha
  · intro r eq
    cases eq
    have h1 := ha _ rfl
    have h2 := hb _ rfl
    simp only [Int.max_def]
    split <;> assumption

theorem mergeMin_within {a b : Option Int} (ha : Within a) (hb : Within b) :
    Within (mergeMin a b) := by
  cases a <;> cases b <;> simp only [mergeMin]
  · exact hb
  · exact hb
  · exact ha
  · intro r eq
    cases eq
    have h1 := ha _ rfl
    have h2 := hb _ rfl
    simp only [Int.min_def]
    split <;> assumption

theorem mergeMax_below {alpha : Int} {a b : Option Int} (ha : Below alpha a) (hb : Below alpha b) :
    Below alpha (mergeMax a b) := by
  cases a <;> cases b <;> simp only [mergeMax]
  · exact hb
  · exact hb
  · exact ha
  · intro r eq
    cases eq
    exact Int.max_le.mpr ⟨hb _ rfl,ha _ rfl⟩

theorem mergeMin_above {alpha : Int} {a b : Option Int} (ha : Above alpha a) (hb : Above alpha b) :
    Above alpha (mergeMin a b) := by
  cases a <;> cases b <;> simp only [mergeMin]
  · exact hb
  · exact hb
  · exact ha
  · intro r eq
    cases eq
    have h1 := ha _ rfl
    have h2 := hb _ rfl
    simp only [Int.min_def]
    split <;> assumption

theorem max_getD (best : Option Int) {r : Int} (bound : -inf < r) :
    (mergeMax best (some r)).getD (-inf) = max r (best.getD (-inf)) := by
  cases best with
  | none => simpa [mergeMax] using (Int.max_eq_left (Int.le_of_lt bound)).symm
  | some b => rfl

theorem min_getD (best : Option Int) {r : Int} (bound : r < inf) :
    (mergeMin best (some r)).getD inf = min r (best.getD inf) := by
  cases best with
  | none => simpa [mergeMin] using (Int.min_eq_left (Int.le_of_lt bound)).symm
  | some b => rfl

theorem sourceMax_eq_run (alpha : Int) (first : Bool) (best : Option Int)
    (xs : List (Option Int)) (fallback : Int)
    (bound : Within best) (bounds : AllWithin xs) (low : Below alpha best)
    (initial : first = true → best = none) :
    sourceMax alpha first (best.getD (-inf)) xs fallback =
      (maxRun alpha best xs).getD fallback := by
  induction xs generalizing first best with
  | nil =>
    cases best with
    | none => simp [sourceMax,maxRun]
    | some b =>
      have hb := bound b rfl
      simp [sourceMax,maxRun,show b ≠ -inf by omega]
  | cons x xs ih =>
    have head := bounds x (by simp)
    have tail : AllWithin xs := fun y hy => bounds y (by simp [hy])
    cases x with
    | none =>
      simpa only [sourceMax,maxRun] using ih false best bound tail low (by simp)
    | some r =>
      have hr := head r rfl
      cases xs with
      | nil =>
        simp only [sourceMax,maxRun]
        split
        · rename_i cut
          cases best with
          | none => simp [Int.max_eq_left (Int.le_of_lt hr.1)]
          | some b =>
            have hb := low b rfl
            simp [Int.max_eq_left (by omega : b ≤ r)]
        · cases best <;> simp [mergeMax,Int.max_eq_left (Int.le_of_lt hr.1)]
      | cons y ys =>
        simp only [sourceMax,maxRun]
        split
        · rfl
        · rename_i noCut
          have encoded : (if first then r else max r (best.getD (-inf))) =
              (mergeMax best (some r)).getD (-inf) := by
            cases first with
            | false => exact (max_getD best hr.1).symm
            | true => rw [initial rfl]; rfl
          rw [encoded]
          apply ih false (mergeMax best (some r)) (mergeMax_within bound head) tail
          · apply mergeMax_below low
            intro v eq
            cases eq
            omega
          · simp

theorem sourceMin_eq_run (alpha : Int) (first : Bool) (best : Option Int)
    (xs : List (Option Int)) (fallback : Int)
    (bound : Within best) (bounds : AllWithin xs) (high : Above alpha best)
    (initial : first = true → best = none) :
    sourceMin alpha first (best.getD inf) xs fallback =
      (minRun alpha best xs).getD fallback := by
  induction xs generalizing first best with
  | nil =>
    cases best with
    | none => simp [sourceMin,minRun]
    | some b =>
      have hb := bound b rfl
      simp [sourceMin,minRun,show b ≠ inf by omega]
  | cons x xs ih =>
    have head := bounds x (by simp)
    have tail : AllWithin xs := fun y hy => bounds y (by simp [hy])
    cases x with
    | none =>
      simpa only [sourceMin,minRun] using ih false best bound tail high (by simp)
    | some r =>
      have hr := head r rfl
      cases xs with
      | nil =>
        simp only [sourceMin,minRun]
        split
        · rename_i cut
          cases best with
          | none => simp [Int.min_eq_left (Int.le_of_lt hr.2)]
          | some b =>
            have hb := high b rfl
            simp [Int.min_eq_left (by omega : r ≤ b)]
        · cases best <;> simp [mergeMin,Int.min_eq_left (Int.le_of_lt hr.2)]
      | cons y ys =>
        simp only [sourceMin,minRun]
        split
        · rfl
        · rename_i noCut
          have encoded : (if first then r else min r (best.getD inf)) =
              (mergeMin best (some r)).getD inf := by
            cases first with
            | false => exact (min_getD best hr.2).symm
            | true => rw [initial rfl]; rfl
          rw [encoded]
          apply ih false (mergeMin best (some r)) (mergeMin_within bound head) tail
          · apply mergeMin_above high
            intro v eq
            cases eq
            omega
          · simp

def sourceMaxWord (alpha : Score) (first : Bool) (best : Score)
    (xs : List (Option Score)) (fallback : Score) : Score :=
  match xs with
  | [] => if best = -(127 : Score) then fallback else best
  | [some r] => Solve2.maxWord r best
  | none :: rest => sourceMaxWord alpha false best rest fallback
  | some r :: rest => if (alpha + 1).toInt ≤ r.toInt then r else
      sourceMaxWord alpha false (if first then r else Solve2.maxWord r best) rest fallback

def sourceMinWord (alpha : Score) (first : Bool) (best : Score)
    (xs : List (Option Score)) (fallback : Score) : Score :=
  match xs with
  | [] => if best = (127 : Score) then fallback else best
  | [some r] => Solve2.minWord r best
  | none :: rest => sourceMinWord alpha false best rest fallback
  | some r :: rest => if r.toInt ≤ alpha.toInt then r else
      sourceMinWord alpha false (if first then r else Solve2.minWord r best) rest fallback

def solve (alpha : Int) (ps os : List (Option Int)) (terminal : Int) : Int :=
  sourceMax alpha true (-inf) (ps.map (Option.map (- ·)))
    (sourceMin alpha true inf os terminal)

def value (ps os : List (Option Int)) (terminal : Int) : Int :=
  (foldMax none (ps.map (Option.map (- ·)))).getD ((foldMin none os).getD terminal)

def solveWord (alpha : Score) (ps os : List (Option Score)) (terminal : Score) : Score :=
  sourceMaxWord alpha true (-127) (ps.map (Option.map (- ·)))
    (sourceMinWord alpha true 127 os terminal)

theorem optional_getD_nws {alpha vf rf : Int} {vo ro : Option Int}
    (h : OptionalRel (NWS alpha) vo ro) (fallback : NWS alpha vf rf) :
    NWS alpha (vo.getD vf) (ro.getD rf) := by
  cases vo <;> cases ro <;> simp only [OptionalRel] at h <;> try contradiction
  · exact fallback
  · exact h

theorem list_negate_nws {alpha : Int} {vs rs : List (Option Int)}
    (h : ListRel (OptionalRel (NWS (-(alpha+1)))) vs rs) :
    ListRel (OptionalRel (NWS alpha)) (vs.map (Option.map (- ·))) (rs.map (Option.map (- ·))) := by
  induction h with
  | nil => exact .nil
  | @cons v r vs rs head tail ih =>
    apply ListRel.cons ?_ ih
    cases v <;> cases r <;> simp only [OptionalRel,Option.map] at * <;> try contradiction
    exact negate_nws head

theorem within_negate {x : Option Int} (h : Within x) : Within (x.map (- ·)) := by
  cases x with
  | none => intro r eq; cases eq
  | some v =>
    intro r eq
    cases eq
    have hv := h v rfl
    dsimp
    constructor <;> omega

theorem allWithin_negate {xs : List (Option Int)} (h : AllWithin xs) :
    AllWithin (xs.map (Option.map (- ·))) := by
  intro x hx
  obtain ⟨v,hv,rfl⟩ := List.mem_map.mp hx
  exact within_negate (h v hv)

theorem solve_eq_runs (alpha : Int) (ps os : List (Option Int)) (terminal : Int)
    (hp : AllWithin ps) (ho : AllWithin os) :
    solve alpha ps os terminal =
      (maxRun alpha none (ps.map (Option.map (- ·)))).getD
        ((minRun alpha none os).getD terminal) := by
  have maxeq := sourceMax_eq_run alpha true none _ (sourceMin alpha true inf os terminal)
    (by intro r h; cases h) (allWithin_negate hp) (by intro r h; cases h) (by simp)
  have mineq := sourceMin_eq_run alpha true none os terminal
    (by intro r h; cases h) ho (by intro r h; cases h) (by simp)
  change solve alpha ps os terminal = _ at maxeq
  rw [show sourceMin alpha true inf os terminal = _ from mineq] at maxeq
  exact maxeq

theorem solve_nws {alpha terminal : Int} {pv ov ps os : List (Option Int)}
    (hp : ListRel (OptionalRel (NWS (-(alpha+1)))) pv ps)
    (ho : ListRel (OptionalRel (NWS alpha)) ov os)
    (psWithin : AllWithin ps) (osWithin : AllWithin os) :
    NWS alpha (value pv ov terminal) (solve alpha ps os terminal) := by
  rw [solve_eq_runs alpha ps os terminal psWithin osWithin]
  exact optional_getD_nws (maxRun_nws (show OptionalRel (NWS alpha) none none from trivial)
    (list_negate_nws hp)) (optional_getD_nws (minRun_nws
      (show OptionalRel (NWS alpha) none none from trivial) ho) (nws_exact _ _))

theorem word_negInf_iff (s : Score) : s = -(127 : Score) ↔ s.toInt = -inf := by
  constructor
  · intro h; subst s; rfl
  · intro h
    apply BitVec.eq_of_toInt_eq
    exact h

theorem word_inf_iff (s : Score) : s = (127 : Score) ↔ s.toInt = inf := by
  constructor
  · intro h; subst s; rfl
  · intro h
    apply BitVec.eq_of_toInt_eq
    exact h

theorem sourceMaxWord_toInt (alpha : Score) (first : Bool) (best : Score)
    (xs : List (Option Score)) (fallback : Score) (safe : alpha.toInt < 2147483647) :
    (sourceMaxWord alpha first best xs fallback).toInt =
      sourceMax alpha.toInt first best.toInt (xs.map (Option.map BitVec.toInt)) fallback.toInt := by
  induction xs generalizing first best with
  | nil =>
    simp only [sourceMaxWord, sourceMax, List.map_nil, word_negInf_iff]
    split <;> rfl
  | cons x xs ih =>
    cases x with
    | none => simpa only [sourceMaxWord,sourceMax,List.map_cons,Option.map_none] using ih false best
    | some r =>
      cases xs with
      | nil => exact Solve2.maxWord_toInt r best
      | cons y ys =>
        simp only [sourceMaxWord,sourceMax,List.map_cons,Option.map_some, Solve2.add_one_safe alpha safe]
        split
        · rfl
        · have h := ih false (if first then r else Solve2.maxWord r best)
          cases first <;> simpa only [Bool.false_eq_true, ↓reduceIte, Solve2.maxWord_toInt, List.map_cons] using h

theorem sourceMinWord_toInt (alpha : Score) (first : Bool) (best : Score)
    (xs : List (Option Score)) (fallback : Score) :
    (sourceMinWord alpha first best xs fallback).toInt =
      sourceMin alpha.toInt first best.toInt (xs.map (Option.map BitVec.toInt)) fallback.toInt := by
  induction xs generalizing first best with
  | nil =>
    simp only [sourceMinWord, sourceMin, List.map_nil, word_inf_iff]
    split <;> rfl
  | cons x xs ih =>
    cases x with
    | none => simpa only [sourceMinWord,sourceMin,List.map_cons,Option.map_none] using ih false best
    | some r =>
      cases xs with
      | nil => exact Solve2.minWord_toInt r best
      | cons y ys =>
        simp only [sourceMinWord,sourceMin,List.map_cons,Option.map_some]
        split
        · rfl
        · have h := ih false (if first then r else Solve2.minWord r best)
          cases first <;> simpa only [Bool.false_eq_true, ↓reduceIte, Solve2.minWord_toInt, List.map_cons] using h

theorem list_neg_toInt (ps : List (Option Score))
    (within : AllWithin (ps.map (Option.map BitVec.toInt))) :
    (ps.map (Option.map (- ·))).map (Option.map BitVec.toInt) =
      (ps.map (Option.map BitVec.toInt)).map (Option.map (- ·)) := by
  apply List.ext_getElem
  · simp
  · intro i h1 h2
    have hi : i < ps.length := by simpa using h1
    simp only [List.getElem_map]
    apply Solve2.map_neg_toInt
    intro s hs
    have hmem : (ps[i]).map BitVec.toInt ∈ ps.map (Option.map BitVec.toInt) :=
      List.mem_map.mpr ⟨ps[i], List.getElem_mem _, rfl⟩
    have bound := within _ hmem s.toInt (by simp [hs])
    dsimp [inf] at bound
    omega

theorem solveWord_toInt (alpha : Score) (ps os : List (Option Score)) (terminal : Score)
    (safe : alpha.toInt < 2147483647)
    (within : AllWithin (ps.map (Option.map BitVec.toInt))) :
    (solveWord alpha ps os terminal).toInt =
      solve alpha.toInt (ps.map (Option.map BitVec.toInt))
        (os.map (Option.map BitVec.toInt)) terminal.toInt := by
  unfold solveWord solve
  rw [sourceMaxWord_toInt _ _ _ _ _ safe, sourceMinWord_toInt, list_neg_toInt ps within]
  rfl

theorem solveWord_nws {alpha terminal : Score} {ps os : List (Option Score)}
    {pv ov : List (Option Int)} (safe : alpha.toInt < 2147483647)
    (hp : ListRel (OptionalRel (NWS (Solve2.childAlpha alpha).toInt))
      pv (ps.map (Option.map BitVec.toInt)))
    (ho : ListRel (OptionalRel (NWS alpha.toInt)) ov (os.map (Option.map BitVec.toInt)))
    (psWithin : AllWithin (ps.map (Option.map BitVec.toInt)))
    (osWithin : AllWithin (os.map (Option.map BitVec.toInt))) :
    NWS alpha.toInt (value pv ov terminal.toInt) (solveWord alpha ps os terminal).toInt := by
  rw [solveWord_toInt alpha ps os terminal safe psWithin]
  rw [Solve2.childAlpha_safe alpha safe] at hp
  exact solve_nws hp ho psWithin osWithin

end NeuralReversi.Solve34

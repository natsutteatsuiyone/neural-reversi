import NeuralReversi.ScoreRefresh.BoardBridge
import NeuralReversi.Solve34.Control

namespace NeuralReversi.ScoreRefresh.SolveBridge

def neg (s : DiscScore) : DiscScore := DiscScore.new (-s.raw)
def addOne (s : DiscScore) : DiscScore := DiscScore.new (s.raw + 1)
def maximum (a b : DiscScore) : DiscScore := if DiscScore.le a b then b else a
def minimum (a b : DiscScore) : DiscScore := if DiscScore.le a b then a else b

theorem neg_release (s : DiscScore) : DiscScore.negate false s = .ok (neg s) := rfl
theorem addOne_release (s : DiscScore) : DiscScore.addInt false s 1 = .ok (addOne s) := rfl

theorem maximum_raw (a b : DiscScore) : (maximum a b).raw = Solve2.maxWord a.raw b.raw := by
  unfold maximum DiscScore.le Solve2.maxWord
  split <;> simp_all <;> omega

theorem minimum_raw (a b : DiscScore) : (minimum a b).raw = Solve2.minWord a.raw b.raw := by
  unfold minimum DiscScore.le Solve2.minWord
  split <;> simp_all <;> omega

def sourceMax (alpha : DiscScore) (first : Bool) (best : DiscScore)
    (xs : List (Option DiscScore)) (fallback : DiscScore) : DiscScore :=
  match xs with
  | [] => if best = neg discInf then fallback else best
  | [some r] => maximum r best
  | none :: rest => sourceMax alpha false best rest fallback
  | some r :: rest => if DiscScore.le (addOne alpha) r then r else
      sourceMax alpha false (if first then r else maximum r best) rest fallback

def sourceMin (alpha : DiscScore) (first : Bool) (best : DiscScore)
    (xs : List (Option DiscScore)) (fallback : DiscScore) : DiscScore :=
  match xs with
  | [] => if best = discInf then fallback else best
  | [some r] => minimum r best
  | none :: rest => sourceMin alpha false best rest fallback
  | some r :: rest => if DiscScore.le r alpha then r else
      sourceMin alpha false (if first then r else minimum r best) rest fallback

theorem sourceMax_raw (alpha : DiscScore) (first : Bool) (best : DiscScore)
    (xs : List (Option DiscScore)) (fallback : DiscScore) :
    (sourceMax alpha first best xs fallback).raw =
      Solve34.sourceMaxWord alpha.raw first best.raw (xs.map (Option.map DiscScore.value)) fallback.raw := by
  induction xs generalizing first best with
  | nil =>
    have he : best = neg discInf ↔ best.raw = -(127 : Word) := by cases best; simp [neg, discInf, DiscScore.new]
    simp only [sourceMax, Solve34.sourceMaxWord, List.map_nil, he]
    split <;> rfl
  | cons x xs ih =>
    cases x with
    | none => simpa only [sourceMax, Solve34.sourceMaxWord, List.map_cons, Option.map_none] using ih false best
    | some r =>
      cases xs with
      | nil => exact maximum_raw r best
      | cons y ys =>
        simp only [sourceMax, Solve34.sourceMaxWord, List.map_cons, Option.map_some,
          DiscScore.le, addOne, DiscScore.new, DiscScore.value]
        by_cases cut : (alpha.raw + 1).toInt ≤ r.raw.toInt
        · simp only [cut, decide_true, ↓reduceIte]
        · simp only [cut, decide_false, Bool.false_eq_true, ↓reduceIte]
          have h := ih false (if first then r else maximum r best)
          cases first <;> simpa only [Bool.false_eq_true, ↓reduceIte, maximum_raw,
            List.map_cons, DiscScore.value] using h

theorem sourceMin_raw (alpha : DiscScore) (first : Bool) (best : DiscScore)
    (xs : List (Option DiscScore)) (fallback : DiscScore) :
    (sourceMin alpha first best xs fallback).raw =
      Solve34.sourceMinWord alpha.raw first best.raw (xs.map (Option.map DiscScore.value)) fallback.raw := by
  induction xs generalizing first best with
  | nil =>
    have he : best = discInf ↔ best.raw = (127 : Word) := by cases best; simp [discInf]
    simp only [sourceMin, Solve34.sourceMinWord, List.map_nil, he]
    split <;> rfl
  | cons x xs ih =>
    cases x with
    | none => simpa only [sourceMin, Solve34.sourceMinWord, List.map_cons, Option.map_none] using ih false best
    | some r =>
      cases xs with
      | nil => exact minimum_raw r best
      | cons y ys =>
        simp only [sourceMin, Solve34.sourceMinWord, List.map_cons, Option.map_some,
          DiscScore.le, DiscScore.value, decide_eq_true_eq]
        split
        · rfl
        · have h := ih false (if first then r else minimum r best)
          cases first <;> simpa only [Bool.false_eq_true, ↓reduceIte, minimum_raw,
            List.map_cons, DiscScore.value] using h

def solve (alpha : DiscScore) (ps os : List (Option DiscScore)) (terminal : DiscScore) : DiscScore :=
  sourceMax alpha true (neg discInf) (ps.map (Option.map neg))
    (sourceMin alpha true discInf os terminal)

theorem solve_raw (alpha : DiscScore) (ps os : List (Option DiscScore)) (terminal : DiscScore) :
    (solve alpha ps os terminal).raw =
      Solve34.solveWord alpha.raw (ps.map (Option.map DiscScore.value))
        (os.map (Option.map DiscScore.value)) terminal.raw := by
  unfold solve Solve34.solveWord
  rw [sourceMax_raw, sourceMin_raw]
  simp only [List.map_map, Option.map_map, Function.comp_def, neg, DiscScore.new, DiscScore.value, discInf]

theorem solve_nws (alpha terminal : DiscScore) (ps os : List (Option DiscScore))
    (pv ov : List (Option Int)) (safe : alpha.raw.toInt < 2147483647)
    (hp : Solve34.ListRel (Solve2.OptionalRel (Solve2.NWS (Solve2.childAlpha alpha.raw).toInt))
      pv ((ps.map (Option.map DiscScore.value)).map (Option.map BitVec.toInt)))
    (ho : Solve34.ListRel (Solve2.OptionalRel (Solve2.NWS alpha.raw.toInt))
      ov ((os.map (Option.map DiscScore.value)).map (Option.map BitVec.toInt)))
    (psWithin : Solve34.AllWithin ((ps.map (Option.map DiscScore.value)).map (Option.map BitVec.toInt)))
    (osWithin : Solve34.AllWithin ((os.map (Option.map DiscScore.value)).map (Option.map BitVec.toInt))) :
    Solve2.NWS alpha.raw.toInt (Solve34.value pv ov terminal.raw.toInt) (solve alpha ps os terminal).raw.toInt := by
  rw [solve_raw]
  exact Solve34.solveWord_nws safe hp ho psWithin osWithin

end NeuralReversi.ScoreRefresh.SolveBridge

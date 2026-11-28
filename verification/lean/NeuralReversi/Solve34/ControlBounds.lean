module
public import NeuralReversi.Solve34.Control
public import NeuralReversi.Solve34.BaseBounds

@[expose] public section

namespace NeuralReversi.Solve34

def AllBounded (lo hi : Int) (xs : List (Option Int)) : Prop :=
  ∀ x ∈ xs, BoundedOption lo hi x

theorem mergeMax_bounds {lo hi : Int} {a b : Option Int}
    (ha : BoundedOption lo hi a) (hb : BoundedOption lo hi b) :
    BoundedOption lo hi (mergeMax a b) := by
  cases a <;> cases b <;> simp_all [BoundedOption, mergeMax, Int.max_def]
  split <;> simp_all

theorem mergeMin_bounds {lo hi : Int} {a b : Option Int}
    (ha : BoundedOption lo hi a) (hb : BoundedOption lo hi b) :
    BoundedOption lo hi (mergeMin a b) := by
  cases a <;> cases b <;> simp_all [BoundedOption, mergeMin, Int.min_def]
  split <;> simp_all

theorem maxRun_bounds {lo hi : Int} (alpha : Int) (best : Option Int)
    (xs : List (Option Int)) (hb : BoundedOption lo hi best) (hx : AllBounded lo hi xs) :
    BoundedOption lo hi (maxRun alpha best xs) := by
  induction xs generalizing best with
  | nil => exact hb
  | cons x xs ih =>
    have head := hx x (by simp)
    have tail : AllBounded lo hi xs := fun y hy => hx y (by simp [hy])
    cases x with
    | none => exact ih best hb tail
    | some r =>
      simp only [maxRun]
      split
      · exact head
      · exact ih _ (mergeMax_bounds hb head) tail

theorem minRun_bounds {lo hi : Int} (alpha : Int) (best : Option Int)
    (xs : List (Option Int)) (hb : BoundedOption lo hi best) (hx : AllBounded lo hi xs) :
    BoundedOption lo hi (minRun alpha best xs) := by
  induction xs generalizing best with
  | nil => exact hb
  | cons x xs ih =>
    have head := hx x (by simp)
    have tail : AllBounded lo hi xs := fun y hy => hx y (by simp [hy])
    cases x with
    | none => exact ih best hb tail
    | some r =>
      simp only [minRun]
      split
      · exact head
      · exact ih _ (mergeMin_bounds hb head) tail

theorem getD_bounds {lo hi : Int} {o : Option Int} {fallback : Int}
    (ho : BoundedOption lo hi o) (hf : lo ≤ fallback ∧ fallback ≤ hi) :
    lo ≤ o.getD fallback ∧ o.getD fallback ≤ hi := by
  cases o with
  | none => exact hf
  | some r => exact ho r rfl

theorem bounded114_within {xs : List (Option Int)} (h : AllBounded (-114) 114 xs) :
    AllWithin xs := by
  intro x hx r hr
  have hh := h x hx r hr
  dsimp [inf]
  constructor <;> omega

theorem bounded114_negate {xs : List (Option Int)} (h : AllBounded (-114) 114 xs) :
    AllBounded (-114) 114 (xs.map (Option.map (- ·))) := by
  intro x hx
  obtain ⟨v,hv,rfl⟩ := List.mem_map.mp hx
  exact option_neg_bounds (h v hv)

theorem solve_bounds (alpha : Int) (ps os : List (Option Int)) (terminal : Int)
    (hp : AllBounded (-114) 114 ps) (ho : AllBounded (-114) 114 os)
    (ht : -114 ≤ terminal ∧ terminal ≤ 114) :
    -114 ≤ solve alpha ps os terminal ∧ solve alpha ps os terminal ≤ 114 := by
  rw [solve_eq_runs alpha ps os terminal (bounded114_within hp) (bounded114_within ho)]
  apply getD_bounds (maxRun_bounds _ _ _ (by intro s hs; cases hs) (bounded114_negate hp))
  exact getD_bounds (minRun_bounds _ _ _ (by intro s hs; cases hs) ho) ht

theorem solveWord_bounds (alpha : Score) (ps os : List (Option Score)) (terminal : Score)
    (safe : alpha.toInt < 2147483647)
    (hp : AllBounded (-114) 114 (ps.map (Option.map BitVec.toInt)))
    (ho : AllBounded (-114) 114 (os.map (Option.map BitVec.toInt)))
    (ht : -114 ≤ terminal.toInt ∧ terminal.toInt ≤ 114) :
    -114 ≤ (solveWord alpha ps os terminal).toInt ∧
      (solveWord alpha ps os terminal).toInt ≤ 114 := by
  rw [solveWord_toInt alpha ps os terminal safe (bounded114_within hp)]
  exact solve_bounds _ _ _ _ hp ho ht

end NeuralReversi.Solve34

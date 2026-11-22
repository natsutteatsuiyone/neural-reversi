import NeuralReversi.TranspositionTable.Packing

namespace NeuralReversi.TranspositionTable.Penalty

open Packing

theorem penalized_depth (e : EntryData) (positive : 0 < e.depth.toNat) :
    (penalized e).depth.toNat + 1 = e.depth.toNat := by
  have bound : e.depth.toNat < 256 := e.depth.toFin.isLt
  dsimp [penalized]
  change ((255 + e.depth.toNat) % 256) + 1 = e.depth.toNat
  omega

theorem penalized_canCut (e : EntryData) (positive : 0 < e.depth.toNat)
    (checked : Bool) (beta : Score) (depth : Depth)
    (selectivity : Selectivity) (requireEndgame : Bool) :
    canCut checked (penalized e) beta depth selectivity requireEndgame =
      if depth.toNat < e.depth.toNat then
        canCut checked e beta depth selectivity requireEndgame
      else .ok false := by
  have hd := penalized_depth e positive
  by_cases below : depth.toNat < e.depth.toNat
  · have gates : Eligible (penalized e) depth selectivity requireEndgame ↔
        Eligible e depth selectivity requireEndgame := by
      dsimp [Eligible, penalized]
      constructor <;> intro h <;> refine ⟨h.1, ?_, h.2.2⟩ <;> omega
    rw [ite_eq_left below, canCut_contract, canCut_contract]
    simp only [gates]
    rfl
  · rw [ite_eq_right below]
    apply canCut_ineligible
    intro eligible
    have enough := eligible.2.1
    omega

theorem penalized_no_new_cut (e : EntryData) (positive : 0 < e.depth.toNat)
    (checked : Bool) (beta : Score) (depth : Depth)
    (selectivity : Selectivity) (requireEndgame : Bool)
    (cut : canCut checked (penalized e) beta depth selectivity requireEndgame = .ok true) :
    canCut checked e beta depth selectivity requireEndgame = .ok true := by
  rw [penalized_canCut e positive] at cut
  split at cut
  · exact cut
  · cases cut

theorem penalized_boundSound (e : EntryData) (truth : Int) :
    BoundSound (penalized e) truth ↔ BoundSound e truth := by rfl

theorem penalized_cut_sound (e : EntryData) (positive : 0 < e.depth.toNat)
    (checked : Bool) (beta : Score) (depth : Depth)
    (selectivity : Selectivity) (requireEndgame : Bool) (truth : Int)
    (sound : BoundSound e truth)
    (cut : canCut checked (penalized e) beta depth selectivity requireEndgame = .ok true) :
    (beta.raw.toInt ≤ e.score.toInt ↔ beta.raw.toInt ≤ truth) ∧
    (e.score.toInt < beta.raw.toInt ↔ truth < beta.raw.toInt) := by
  exact canCut_sound checked e beta depth selectivity requireEndgame truth sound
    (penalized_no_new_cut e positive checked beta depth selectivity requireEndgame cut)

inductive Trace : Nat → EntryData → EntryData → Prop where
  | refl (e : EntryData) : Trace 0 e e
  | step {n : Nat} {first last : EntryData} (prior : Trace n first last)
      (positive : 0 < last.depth.toNat) : Trace (n + 1) first (penalized last)

theorem trace_depth {n : Nat} {first last : EntryData} (trace : Trace n first last) :
    last.depth.toNat + n = first.depth.toNat := by
  induction trace with
  | refl => omega
  | @step n first last prior positive ih =>
    have hd := penalized_depth last positive
    omega

theorem trace_length_bound {n : Nat} {first last : EntryData} (trace : Trace n first last) :
    n ≤ first.depth.toNat ∧ n ≤ 255 := by
  have hd := trace_depth trace
  have bound : first.depth.toNat < 256 := first.depth.toFin.isLt
  omega

theorem trace_no_new_cut {n : Nat} {first last : EntryData}
    (trace : Trace n first last) (checked : Bool) (beta : Score) (depth : Depth)
    (selectivity : Selectivity) (requireEndgame : Bool)
    (cut : canCut checked last beta depth selectivity requireEndgame = .ok true) :
    canCut checked first beta depth selectivity requireEndgame = .ok true := by
  induction trace with
  | refl => exact cut
  | @step n first last prior positive ih =>
    exact ih (penalized_no_new_cut last positive checked beta depth selectivity
      requireEndgame cut)

theorem trace_boundSound {n : Nat} {first last : EntryData}
    (trace : Trace n first last) (truth : Int) :
    BoundSound last truth ↔ BoundSound first truth := by
  induction trace with
  | refl => rfl
  | step prior positive ih =>
    rw [penalized_boundSound]
    exact ih

private def fixture (score : Int) (depth : Nat) : EntryData :=
  ⟨BitVec.ofInt 16 score, 64, 3, BitVec.ofNat 8 depth, 5, 0, 1⟩

example : (penalized (fixture 0 1)).depth = 0 := by rfl
example : (penalized (fixture 0 255)).depth = 254 := by rfl
example : canCut true (penalized (fixture 0 12)) ⟨0⟩ 12 ⟨5, by decide⟩ true =
    .ok false := by rfl
example : canCut true (penalized (fixture 0 12)) ⟨0⟩ 11 ⟨5, by decide⟩ true =
    .ok true := by rfl

example : canCut true (penalized (fixture 0 1)) ⟨0⟩ 0 ⟨5, by decide⟩ true =
    .ok true := by rfl
example : canCut true (penalized (fixture 0 255)) ⟨0⟩ 255 ⟨5, by decide⟩ true =
    .ok false := by rfl
example : canCut true (penalized { fixture 0 12 with selectivity := 4 })
    ⟨0⟩ 10 ⟨5, by decide⟩ true = .ok false := by rfl
example : canCut true (penalized { fixture 0 12 with flags := 0 })
    ⟨0⟩ 10 ⟨5, by decide⟩ true = .ok false := by rfl

example : canCut true (fixture (-32768) 12) ⟨0⟩ 12 ⟨5, by decide⟩ true =
    .error scoreError ∧
    canCut true (penalized (fixture (-32768) 12)) ⟨0⟩ 12 ⟨5, by decide⟩ true =
    .ok false := by constructor <;> rfl

example : canCut false (fixture 0 0) ⟨0⟩ 1 ⟨5, by decide⟩ true = .ok false ∧
    canCut false (penalized (fixture 0 0)) ⟨0⟩ 1 ⟨5, by decide⟩ true = .ok true := by
  constructor <;> rfl

end NeuralReversi.TranspositionTable.Penalty

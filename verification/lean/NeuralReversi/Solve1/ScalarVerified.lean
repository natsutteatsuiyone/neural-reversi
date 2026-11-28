module
public import NeuralReversi.Solve1.ScalarCounts
public import NeuralReversi.Solve1.Bmi2Verified

@[expose] public section

namespace NeuralReversi.Solve1.Scalar

theorem params_bounds (s : Fin 64) :
    (params s.val).rowShift ≤ 56 ∧ (params s.val).rowShift % 8 = 0 ∧
    (params s.val).t0 < 8 ∧ (params s.val).t1 < 8 ∧
    (params s.val).t2 < 8 ∧ (params s.val).t3 < 8 := by
  have all : ∀ s : Fin 64,
      (params s.val).rowShift ≤ 56 ∧ (params s.val).rowShift % 8 = 0 ∧
      (params s.val).t0 < 8 ∧ (params s.val).t1 < 8 ∧
      (params s.val).t2 < 8 ∧ (params s.val).t3 < 8 := by decide
  exact all s

theorem laneCount_signed (s : Nat) (lane : Fin 4) (p : Word) :
    (laneCount s lane p).toInt = ((laneCount s lane p).toNat : Int) := by
  apply BitVec.toInt_eq_toNat_of_lt
  have := laneCount_bound s lane p
  omega

theorem count_additions_safe (s : Nat) (p : Word) :
    let a := laneCount s 0 p
    let b := laneCount s 1 p
    let c := laneCount s 2 p
    let d := if (params s).maskD9 = 0 then (0 : BitVec 32) else laneCount s 3 p
    BitVec.saddOverflow a b = false ∧
      BitVec.saddOverflow (a + b) c = false ∧
      BitVec.saddOverflow (a + b + c) d = false := by
  have h0 := laneCount_bound s 0 p
  have h1 := laneCount_bound s 1 p
  have h2 := laneCount_bound s 2 p
  have h3 := laneCount_bound s 3 p
  have e0 := laneCount_signed s 0 p
  have e1 := laneCount_signed s 1 p
  have e2 := laneCount_signed s 2 p
  have e3 := laneCount_signed s 3 p
  have ab := add_bridge (laneCount s 0 p) (laneCount s 1 p) (by omega) (by omega)
  have abc := add_bridge (laneCount s 0 p + laneCount s 1 p) (laneCount s 2 p)
    (by omega) (by omega)
  dsimp only
  refine ⟨ab.2, abc.2, ?_⟩
  split
  · exact (add_bridge _ 0 (by change -2147483648 ≤ _ + 0; omega)
      (by change _ + 0 < 2147483648; omega)).2
  · exact (add_bridge _ (laneCount s 3 p) (by omega) (by omega)).2

def solve (b : Bmi2.Board) (alpha : ScoreWord) : ScoreWord :=
  scalarWord (Bmi2.popCount b.player) (count b.square.val b.player).toNat
    (count b.square.val (~~~b.player)).toNat alpha

theorem solve_eq_bmi2 (b : Bmi2.Board) (alpha : ScoreWord) :
    solve b alpha = Bmi2.solve b alpha := by
  unfold solve Bmi2.solve
  rw [count_toNat, count_toNat, Bmi2.countWide_toNat, Bmi2.countOppWide_toNat,
    Bmi2.reference_eq_geometry, Bmi2.reference_eq_geometry]
  exact (backend_word_equivalence _ _ _ _).1

theorem solve_nws (b : Bmi2.Board) (alpha : ScoreWord) :
    NWSResult alpha.toInt (Bmi2.exactScore b) (solve b alpha).toInt := by
  rw [solve_eq_bmi2]
  exact Bmi2.solve_nws b alpha

end NeuralReversi.Solve1.Scalar

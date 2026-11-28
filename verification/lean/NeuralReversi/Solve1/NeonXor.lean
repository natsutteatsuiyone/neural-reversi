module
public import NeuralReversi.Solve1.NeonModel

@[expose] public section

namespace NeuralReversi.Solve1.Neon

private theorem laneCell_lt64 : ∀ (s : Fin 64) (lane : Fin 4) (i : Fin 8),
    (Scalar.laneCell s.val lane i.val).all (fun sq => decide (sq < 64)) = true := by
  decide +kernel

private theorem laneXor_matches_laneCell : ∀ (s : Fin 64) (lane : Fin 4) (i : Fin 8),
    (lane.val < 3 ∨ hasFourth s.val = true) →
    (laneXor s.val lane).getLsbD i.val = (Scalar.laneCell s.val lane i.val).isSome := by
  decide +kernel

theorem xor_correct (s : Fin 64) (lane : Fin 4) (p : Word)
    (active : lane.val < 3 ∨ hasFourth s.val = true) :
    Scalar.laneIndex s.val lane p ^^^ laneXor s.val lane = Scalar.laneIndex s.val lane (~~~p) := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  rw [BitVec.getLsbD_xor, Scalar.laneIndex_getLsbD s lane p i hi,
    Scalar.laneIndex_getLsbD s lane (~~~p) i hi, laneXor_matches_laneCell s lane ⟨i, hi⟩ active]
  have hlt := laneCell_lt64 s lane ⟨i, hi⟩
  cases hc : Scalar.laneCell s.val lane i with
  | none => simp
  | some sq =>
    rw [hc] at hlt
    simp only [Option.all, decide_eq_true_eq] at hlt
    simp [hlt]

end NeuralReversi.Solve1.Neon

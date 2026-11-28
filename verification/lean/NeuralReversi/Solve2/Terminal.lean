module
public import NeuralReversi.Solve2.BoardBridge
public import NeuralReversi.Board.ScoreProperties

@[expose] public section

namespace NeuralReversi.Solve2

theorem empty_masks_disjoint (b : TwoEmpty) :
    Bitboard.squareMask b.first &&& Bitboard.squareMask b.second = 0 := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  have different : b.first.val ≠ b.second.val := fun h => b.distinct (Fin.ext h)
  simp only [BitVec.getLsbD_and, Bitboard.squareMask_bit]
  by_cases h : i = b.first.val <;> simp_all

theorem exactly_two_empty (b : TwoEmpty) : Board.emptyCount b.asBoard = 2 := by
  have union := Bitboard.count_or_of_disjoint _ _ (empty_masks_disjoint b)
  rw [Board.squareMask_count_one, Board.squareMask_count_one] at union
  apply BitVec.eq_of_toNat_eq
  change (Bitboard.count (~~~(b.player ||| b.opponent))).toNat = 2
  rw [b.empty_exact]
  exact union

def terminalScore (b : TwoEmpty) : BitVec 32 :=
  Board.finishScore (Board.discScore b.asBoard) 2

theorem terminal_solve (b : TwoEmpty) (checked : Bool) :
    Board.solve checked b.asBoard 2 = .ok (terminalScore b) := by
  simp [Board.solve, exactly_two_empty b, terminalScore]

theorem terminal_awarded (b : TwoEmpty) :
    (terminalScore b).toInt = Board.AwardedScore b.asBoard := by
  obtain ⟨result, release, _, policy, _⟩ := Board.solvePolicyContract_proven b.asBoard b.disjoint
  rw [exactly_two_empty b, terminal_solve b false] at release
  cases release
  exact policy

theorem terminal_bounds (b : TwoEmpty) :
    -64 ≤ (terminalScore b).toInt ∧ (terminalScore b).toInt ≤ 64 := by
  obtain ⟨result, release, _, _, lower, upper, _⟩ :=
    Board.solvePolicyContract_proven b.asBoard b.disjoint
  rw [exactly_two_empty b, terminal_solve b false] at release
  cases release
  exact ⟨lower, upper⟩

theorem terminal_arithmetic_safe (b : TwoEmpty) :
    BitVec.smulOverflow (Board.playerCount b.asBoard) 2 = false ∧
    BitVec.ssubOverflow (Board.playerCount b.asBoard * 2) 64 = false ∧
    BitVec.saddOverflow (Board.discScore b.asBoard) 2 = false ∧
    BitVec.saddOverflow (Board.discScore b.asBoard + 2) 2 = false := by
  have base := Board.discScoreContract_proven b.asBoard
  obtain ⟨_, _, _, _, _, _, add1, add2⟩ :=
    Board.solvePolicyContract_proven b.asBoard b.disjoint
  rw [exactly_two_empty b] at add1 add2
  exact ⟨base.2.2.2.1, base.2.2.2.2, add1, add2⟩

end NeuralReversi.Solve2

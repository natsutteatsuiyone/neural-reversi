import NeuralReversi.PvsBoardTree

namespace NeuralReversi.PvsBoardTree

open Bitboard.Moves (X86Config)
open PvsMoveLoop (scoreInf scoreMax bestValue)

def exceeds (mc : X86Config) (fc : Flip.Dispatch.X86Config) (board : Position) (cut : Int) : Bool :=
  if cut < -scoreMax then true
  else if scoreMax ≤ cut then false
  else
    match moves : legalSquares mc board.val with
    | first :: rest =>
      !exceeds mc fc (play mc fc board first).val (-cut - 1) ||
        rest.any (fun sq => !exceeds mc fc (play mc fc board sq).val (-cut - 1))
    | [] =>
      if mobile : Board.hasLegalMoves mc (pass board).val = true then
        !exceeds mc fc (pass board) (-cut - 1)
      else decide (cut < (terminalScore board).val)
termination_by Board.playRank mc board.val
decreasing_by
  · exact (play mc fc board first).property
  · exact (play mc fc board sq).property
  · exact Board.legalCheckedStep_strict mc (.pass ⟨stuck_of_nil moves, mobile⟩) board.property

private theorem lt_bestValue_iff {Move : Type} (value : Move → Int) (moves : List Move)
    {seed cut : Int} (seedBound : seed ≤ cut) :
    cut < bestValue value moves seed ↔ ∃ move ∈ moves, cut < value move := by
  constructor
  · intro above
    classical
    by_cases existsMove : ∃ move ∈ moves, cut < value move
    · exact existsMove
    · have bounded := PvsMoveLoop.bestValue_le value moves seed cut seedBound fun move present => by
        have absent : ¬cut < value move := fun score => existsMove ⟨move, present, score⟩
        omega
      omega
  · rintro ⟨move, present, above⟩
    exact Int.lt_of_lt_of_le above (PvsMoveLoop.bestValue_ge_move value moves seed present)

theorem exceeds_iff (mc : X86Config) (fc : Flip.Dispatch.X86Config) (board : Position) (cut : Int) :
    exceeds mc fc board cut = true ↔ cut < minimax mc fc board := by
  have range := PvsMoveLoop.minimax_range (build mc fc board)
  change -scoreMax ≤ minimax mc fc board ∧ minimax mc fc board ≤ scoreMax at range
  rw [exceeds.eq_def]
  split
  · simp only [true_iff]
    omega
  next floor =>
    split
    · simp only [Bool.false_eq_true, false_iff]
      omega
    · split
      next first rest moves =>
        rw [minimax_moves mc fc board first rest moves,
          lt_bestValue_iff _ _ (show -scoreInf ≤ cut by unfold scoreInf scoreMax at *; omega)]
        simp only [Bool.or_eq_true, List.any_eq_true, Bool.not_eq_true']
        have child (sq : GeneratedSquare mc board.val) :
            exceeds mc fc (play mc fc board sq).val (-cut - 1) = false ↔
              cut < -minimax mc fc (play mc fc board sq).val := by
          rw [← Bool.not_eq_true, exceeds_iff mc fc (play mc fc board sq).val (-cut - 1)]
          omega
        simp only [child]
        constructor
        · rintro (head | ⟨sq, present, above⟩)
          · exact ⟨first, List.mem_cons_self, head⟩
          · exact ⟨sq, List.mem_cons_of_mem _ present, above⟩
        · rintro ⟨sq, member, above⟩
          rcases List.mem_cons.mp member with rfl | tail
          · exact Or.inl above
          · exact Or.inr ⟨sq, tail, above⟩
      next moves =>
        split
        next mobile =>
          have passStep := Board.legalCheckedStep_strict mc (.pass ⟨stuck_of_nil moves, mobile⟩) board.property
          rw [minimax_pass mc fc board moves mobile]
          simp only [Bool.not_eq_true', ← Bool.not_eq_true, exceeds_iff mc fc (pass board) (-cut - 1)]
          omega
        next stuck =>
          rw [terminal_value mc fc board moves (Bool.eq_false_iff.mpr stuck)]
          exact Iff.of_eq decide_eq_true_eq
termination_by Board.playRank mc board.val
decreasing_by
  · exact (play mc fc board sq).property
  · exact passStep

end NeuralReversi.PvsBoardTree

import NeuralReversi.AspirationMir.Wipeout
import NeuralReversi.PvsParallel.LeafReturn
import NeuralReversi.PvsParallel.Single

namespace NeuralReversi.AspirationMir

open PvsParallel ConcurrentReplay
open PvsBoardTree (Position GeneratedSquare play)
open Bitboard (Square)

theorem legalMoveCount_le_empties (mc : Bitboard.Moves.X86Config) (b : Position) :
    (PvsBoardTree.legalSquares mc b.val).length ≤ (Board.emptyCount b.val).toNat := by
    rw [← List.length_map (f := Subtype.val), PvsBoardTree.legalSquares_values,
      ← (Solve34.EmptyBoard.ofBoard b.val b.property).length_eq]
    simp only [Solve34.EmptyBoard.ofBoard]
    simp only [← List.countP_eq_length_filter]
    apply List.countP_mono_left
    intro sq _ bit
    have legal := (Board.getMoves_exact mc b.val sq.val).mp bit
    simp [Board.emptySquares, sq.isLt, legal.2.1, legal.2.2.1, -BitVec.getLsbD_eq_getElem]

private theorem one_choice (mc : Bitboard.Moves.X86Config) (b : Position)
    (roots : List Square) (covers : Root.Covers mc b roots) (one : ExactDepth.Full b 1)
    (nonempty : 0 < roots.length) : ∃ move, Root.choices mc b roots 0 = [move] := by
  have size : (Root.choices mc b roots 0).length ≤ 1 := by
    apply Nat.le_trans (List.length_filter_le _ (PvsBoardTree.legalSquares mc b.val))
    change (1 : BitVec 32) = Board.emptyCount b.val at one
    simpa [← one] using legalMoveCount_le_empties mc b
  obtain ⟨move, rest, shape⟩ := List.exists_cons_of_ne_nil ((Root.maskContract_proven mc b roots 0 covers).2.2.mpr nonempty)
  rw [shape] at size
  have empty : rest = [] := List.length_eq_zero_iff.mp (by simpa using size)
  exact ⟨move, empty ▸ shape⟩

/-- Score and visible row writes of a one-empty root. The move and its flips are
read from the list produced for this board, including its cached wipeout. -/
def oneEmptyExit (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config)
    (board : Board.Board) (roots : List Square) (paired checked : Bool)
    (mem : MoveList.Initialization.Memory Move.Move) : Option (List (Square × Int × Bool) × Int) := do
  let moves := MoveList.Generation.fill false paired fc board (Root.remainingMask mc board roots 0) mem
  match ← moves.wipeout with
  | some sq => return rootWipeoutExit sq
  | none =>
    let mv ← moves.slots 0
    let child ← (NonPvMir.pvZeroScore checked (Board.place board mv.flipped mv.sq)).toOption
    return singleRootExit mv.sq child.raw.toInt

def OneEmptyResultContract : Prop :=
  ∀ (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config) (b : Position)
    (roots : List Square) (paired checked : Bool) (mem : MoveList.Initialization.Memory Move.Move)
    (m : Machine) (pv : Square → List Square) (depth nodes : Nat),
    Root.Covers mc b roots → 0 < roots.length → ExactDepth.Full b 1 →
    m.pc = 0 → Aspiration.Window m.regs.alpha.toInt m.regs.beta.toInt →
    ∃ writes value, oneEmptyExit mc fc b.val roots paired checked mem = some (writes, value) ∧
      ∃ score last, Returns (fun _ _ _ => BitVec.ofInt 32 value) (fun _ => false) m score last ∧
        let payload := writes.foldl (fun rows (sq, s, visible) =>
          RootOrder.writeScore rows sq (if visible then s else -PvsMoveLoop.scoreInf)) (fun _ => -PvsMoveLoop.scoreInf)
        let rows := roots.map fun sq => (⟨sq, payload sq, pv sq⟩ : SearchPublication.Row)
        ∃ result, SearchFallback.fromSnapshot (SearchPublication.capture rows ⟨depth, 5⟩ true) nodes = some result ∧
          result.best.score = score.toInt ∧ score.toInt = PvsBoardTree.minimax mc fc b

theorem oneEmptyResultContract_proven : OneEmptyResultContract := by
  intro mc fc b roots paired checked mem m pv depth nodes covers nonempty one entry initial
  obtain ⟨move, only⟩ := one_choice mc b roots covers one nonempty
  obtain ⟨_, _, stored, cached⟩ := generate_single_root_any mc fc b roots 0 move covers only paired mem
  cases hit : EndgameCutoffs.wipesOut fc b.val move.val with
  | true =>
    simp only [hit, ite_true] at cached
    obtain ⟨score, last, run, saved⟩ :=
      wipeoutResultContract_proven mc fc b roots move.val paired mem m pv depth nodes covers nonempty cached entry initial
    exact ⟨_, _, by simp [oneEmptyExit, cached], score, last, run, saved⟩
  | false =>
    simp only [hit, Bool.false_eq_true, ite_false] at cached
    let child := (play mc fc b move).val
    have zero : ExactDepth.Full child 0 := (ExactDepth.move_depth mc fc b 1 move one).2.1
    obtain ⟨raw, final, _⟩ := consumed_zero_depth mc fc false checked child 0 1 zero
    let out := singleRootExit move.val raw.raw.toInt
    refine ⟨out.1, out.2, ?_, ?_⟩
    · unfold oneEmptyExit
      simp only [cached, stored]
      change (do
        let s ← (NonPvMir.pvZeroScore checked child.val).toOption
        pure (singleRootExit move.val s.raw.toInt)) = some out
      simp [final, Except.toOption, out]
    · apply completeSingle (endian := false) m entry initial b covers nonempty
        (fun _ _ _ => BitVec.ofInt 32 out.2) (fun _ _ _ => _) _ pv depth nodes
      intro _ a beta window
      obtain ⟨other, same, events, obs, used⟩ :=
        consumed_zero_depth mc fc false checked child (-beta.toInt) (-a.toInt) zero
      rw [final] at same
      cases same
      have attempt := rootAttempt_of_single_child mc fc false b roots 0 a.toInt beta.toInt raw.raw.toInt
        move only obs 2 used
      obtain ⟨actual, lower, upper⟩ := single_score_of_reply obs used window move.val
      have repr : (BitVec.ofInt 32 out.2).toInt = out.2 :=
        BitVec.toInt_ofInt_eq_self (by decide) (by dsimp [out]; omega) (by dsimp [out]; omega)
      simpa only [repr] using attempt

end NeuralReversi.AspirationMir

module
public import NeuralReversi.PvsParallel.TraceCheckModel
public import NeuralReversi.PvsBoardTree.Threshold

@[expose] public section

namespace NeuralReversi.PvsParallel.TraceCheck

open Aspiration (SoundResult Window)
open PvsMoveLoop (Bound scoreInf scoreMax)
open PvsBoardTree (GeneratedSquare)

theorem FrameReturn.sane_board {frame : FrameReturn} (sane : frame.sane = true) : Board.wellFormed frame.board := by
  simp only [FrameReturn.sane, Bool.and_eq_true, decide_eq_true_eq] at sane
  exact sane.1.1.1.1.2.2

theorem FrameReturn.sane_depth {frame : FrameReturn} (sane : frame.sane = true) :
    frame.depth = (Board.emptyCount frame.board).toNat := by
  simp only [FrameReturn.sane, Bool.and_eq_true, beq_iff_eq] at sane
  exact sane.1.1.1.2

theorem FrameReturn.sane_window {frame : FrameReturn} (sane : frame.sane = true) :
    Window frame.alpha frame.beta ∧ (frame.pv = false → frame.beta = frame.alpha + 1) := by
  simp only [FrameReturn.sane, Bool.and_eq_true, decide_eq_true_eq, Bool.or_eq_true, beq_iff_eq] at sane
  refine ⟨⟨sane.1.1.2.1, sane.1.1.2.2.1, sane.1.1.2.2.2⟩, ?_⟩
  intro nonpv
  exact sane.1.2.resolve_left (by simp [nonpv])

theorem FrameReturn.sane_score {frame : FrameReturn} (sane : frame.sane = true) :
    -scoreInf < frame.score ∧ frame.score < scoreInf := by
  simp only [FrameReturn.sane, Bool.and_eq_true, decide_eq_true_eq] at sane
  exact sane.2

theorem FrameReturn.onMove_sound {child : FrameReturn} {fc : Flip.Dispatch.X86Config}
    {board : PvsBoardTree.Position} {square : Nat}
    (checked : child.onMove fc board.val square = true) :
    ∃ move : GeneratedSquare moveConfig board.val, move.val.val = square ∧
      child.board = (PvsBoardTree.play moveConfig fc board move).val.val := by
  unfold FrameReturn.onMove at checked
  split at checked
  · rename_i inside
    obtain ⟨legal, placed⟩ := Bool.and_eq_true_iff.mp checked
    exact ⟨⟨⟨square, inside⟩, legal⟩, rfl, of_decide_eq_true placed⟩
  · contradiction

theorem FrameReturn.windowMatches_sound {frame : FrameReturn} {pv : Bool} {alpha beta score : Int}
    (checked : frame.windowMatches pv alpha beta score = true) :
    frame.alpha = alpha ∧ frame.beta = beta ∧ frame.score = score := by
  simp only [FrameReturn.windowMatches, Bool.and_eq_true, beq_iff_eq] at checked
  exact ⟨checked.1.1.2, checked.1.2, checked.2⟩

def FrameReturn.BoardSound (frame : FrameReturn) (fc : Flip.Dispatch.X86Config) : Prop :=
  ∀ valid : Board.wellFormed frame.board,
    SoundResult (PvsBoardTree.minimax moveConfig fc ⟨frame.board, valid⟩) frame.alpha frame.beta frame.score

private theorem FrameReturn.checkWith_sound {frame : FrameReturn} {value : Int} {exceeds : Int → Bool}
    (decision : ∀ cut, exceeds cut = true ↔ cut < value) (checked : frame.checkWith exceeds = true) :
    SoundResult value frame.alpha frame.beta frame.score := by
  simp only [FrameReturn.checkWith, Bool.and_eq_true, decide_eq_true_eq] at checked
  obtain ⟨⟨⟨⟨lower, upper⟩, low⟩, high⟩, exactValue⟩ := checked
  have aboveScore : exceeds (frame.score - 1) = true ↔ frame.score ≤ value := by
    rw [decision]
    omega
  have belowScore : (exceeds frame.score == false) = true ↔ value ≤ frame.score := by
    rw [beq_iff_eq, Bool.eq_false_iff]
    change (¬exceeds frame.score = true) ↔ value ≤ frame.score
    rw [decision]
    omega
  refine ⟨lower, upper, ?_, ?_, ?_⟩
  · intro h
    exact belowScore.mp (by simpa only [ite_eq_left h] using low)
  · intro h
    exact aboveScore.mp (by simpa only [ite_eq_left h] using high)
  · intro a b
    simp only [ite_eq_left (And.intro a b), Bool.and_eq_true, aboveScore, belowScore] at exactValue
    exact Int.le_antisymm exactValue.1 exactValue.2

theorem FrameReturn.checkBound_sound {frame : FrameReturn} {value : Int}
    (checked : frame.checkBound value = true) : SoundResult value frame.alpha frame.beta frame.score :=
  frame.checkWith_sound (fun _ => Iff.of_eq decide_eq_true_eq) checked

theorem FrameReturn.terminal_sound {frame : FrameReturn} {fc : Flip.Dispatch.X86Config}
    (terminal : frame.isTerminal = true) (bound : frame.checkBound frame.terminalValue = true) :
    frame.BoardSound fc := by
  simp only [FrameReturn.isTerminal, Bool.and_eq_true, beq_iff_eq] at terminal
  intro valid
  rw [PvsBoardTree.terminal_value moveConfig fc ⟨frame.board, valid⟩
    (PvsBoardTree.nil_of_stuck terminal.1) terminal.2]
  exact frame.checkBound_sound bound

def FrameReturn.checkThreshold (frame : FrameReturn) (fc : Flip.Dispatch.X86Config)
    (valid : Board.wellFormed frame.board) : Bool :=
  frame.checkWith (PvsBoardTree.exceeds moveConfig fc ⟨frame.board, valid⟩)

theorem FrameReturn.threshold_sound {frame : FrameReturn} {fc : Flip.Dispatch.X86Config}
    {valid : Board.wellFormed frame.board} (checked : frame.checkThreshold fc valid = true) : frame.BoardSound fc :=
  fun _ => frame.checkWith_sound (PvsBoardTree.exceeds_iff moveConfig fc ⟨frame.board, valid⟩) checked

private theorem play_emptyCount (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config)
    (board : PvsBoardTree.Position) (sq : GeneratedSquare mc board.val) :
    (Board.emptyCount (PvsBoardTree.play mc fc board sq).val.val).toNat + 1 =
      (Board.emptyCount board.val).toNat := by
  have accepted := (Board.precomputedMoveContract_proven mc fc true true board.val sq.val
    board.property sq.property).1
  exact (Board.checkedPlacementProgress_proven true _ _ _ _ board.property accepted).2

private theorem any_congr {α : Type} {xs : List α} {p q : α → Bool}
    (h : ∀ x ∈ xs, p x = q x) : xs.any p = xs.any q := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.any_cons, h x (by simp), ih (fun y hy => h y (by simp [hy]))]

private theorem exceedsRaw_eq_exceeds_aux (mc : Bitboard.Moves.X86Config)
    (fc : Flip.Dispatch.X86Config) (board : PvsBoardTree.Position) (cut : Int)
    (fuel : Nat) (canPass : Bool)
    (enough : (Board.emptyCount board.val).toNat ≤ fuel)
    (passValid : canPass = false → Board.hasLegalMoves mc board.val = true) :
    exceedsRaw mc fc board.val cut fuel canPass = PvsBoardTree.exceeds mc fc board cut := by
  rw [exceedsRaw.eq_def, PvsBoardTree.exceeds.eq_def]
  by_cases floor : cut < -scoreMax
  · simp [floor]
  · simp only [floor, ↓reduceIte]
    by_cases ceil : scoreMax ≤ cut
    · simp [ceil]
    · simp only [ceil, ↓reduceIte]
      have hvals := PvsBoardTree.legalSquares_values mc board.val
      cases moves : PvsBoardTree.legalSquares mc board.val with
      | nil =>
        rw [moves] at hvals
        have hfilter : (List.finRange 64).filter (fun sq => (Board.getMoves mc board.val).getLsbD sq.val) = [] :=
          hvals.symm
        cases fuel with
        | zero =>
          by_cases mobile : Board.hasLegalMoves mc (PvsBoardTree.pass board).val = true
          · obtain ⟨sq, legal⟩ := (Board.hasLegalMoves_exact mc (PvsBoardTree.pass board).val).mp mobile
            obtain ⟨move, _, _⟩ := (PvsBoardTree.legalSquares_complete mc (PvsBoardTree.pass board).val ⟨sq, legal.1⟩).mpr legal
            have dec := play_emptyCount mc fc (PvsBoardTree.pass board) move
            have same : Board.emptyCount (PvsBoardTree.pass board).val = Board.emptyCount board.val :=
              Board.switchPlayers_emptyCount board.val
            rw [same] at dec
            omega
          · simp only [mobile]
            rfl
        | succ nextFuel =>
          simp only [hfilter]
          have hcan : canPass = true := by
            cases canPass with
            | true => rfl
            | false =>
              have mobile := passValid rfl
              have stuck := PvsBoardTree.stuck_of_nil moves
              rw [stuck] at mobile
              contradiction
          simp only [hcan, ↓reduceIte]
          by_cases mobile : Board.hasLegalMoves mc (PvsBoardTree.pass board).val = true
          · have mobile' : Board.hasLegalMoves mc (Board.switchPlayers board.val) = true := mobile
            simp only [mobile, mobile', ↓reduceIte, dite_true]
            have nextEnough : (Board.emptyCount (PvsBoardTree.pass board).val).toNat ≤ nextFuel + 1 := by
              change (Board.emptyCount (Board.switchPlayers board.val)).toNat ≤ nextFuel + 1
              rw [Board.switchPlayers_emptyCount]
              exact enough
            exact congrArg Bool.not (exceedsRaw_eq_exceeds_aux mc fc (PvsBoardTree.pass board) (-cut - 1)
              (nextFuel + 1) false nextEnough (fun _ => mobile))
          · have mobile' : ¬ Board.hasLegalMoves mc (Board.switchPlayers board.val) = true := mobile
            simp only [mobile, mobile']
            rfl
      | cons first rest =>
        rw [moves] at hvals
        have hfilter : (List.finRange 64).filter (fun sq => (Board.getMoves mc board.val).getLsbD sq.val) =
            first.val :: rest.map Subtype.val := by simpa using hvals.symm
        have firstStep := play_emptyCount mc fc board first
        cases fuel with
        | zero => omega
        | succ nextFuel =>
          simp only [hfilter, List.any_map]
          have head := exceedsRaw_eq_exceeds_aux mc fc (PvsBoardTree.play mc fc board first).val
            (-cut - 1) nextFuel true (by omega) (by simp)
          change (!exceedsRaw mc fc (PvsBoardTree.play mc fc board first).val.val (-cut - 1) nextFuel true ||
            rest.any (fun sq => !exceedsRaw mc fc (PvsBoardTree.play mc fc board sq).val.val (-cut - 1) nextFuel true)) = _
          rw [head]
          congr 1
          apply any_congr
          intro sq _
          have sqStep := play_emptyCount mc fc board sq
          exact congrArg Bool.not (exceedsRaw_eq_exceeds_aux mc fc (PvsBoardTree.play mc fc board sq).val
            (-cut - 1) nextFuel true (by omega) (by simp))
termination_by Board.playRank mc board.val
decreasing_by
  · exact Board.legalCheckedStep_strict mc (.pass ⟨PvsBoardTree.stuck_of_nil moves, mobile⟩) board.property
  · exact (PvsBoardTree.play mc fc board first).property
  · exact (PvsBoardTree.play mc fc board sq).property

theorem exceedsRaw_eq_exceeds (mc : Bitboard.Moves.X86Config)
    (fc : Flip.Dispatch.X86Config) (board : PvsBoardTree.Position) (cut : Int) :
    exceedsRaw mc fc board.val cut (Board.emptyCount board.val).toNat true =
      PvsBoardTree.exceeds mc fc board cut :=
  exceedsRaw_eq_exceeds_aux mc fc board cut (Board.emptyCount board.val).toNat true (Nat.le_refl _) (by simp)

theorem FrameReturn.checkThresholdRaw_eq (frame : FrameReturn) (fc : Flip.Dispatch.X86Config)
    (sane : frame.sane = true) :
    frame.checkThresholdRaw fc = frame.checkThreshold fc (frame.sane_board sane) := by
  unfold FrameReturn.checkThresholdRaw FrameReturn.checkThreshold
  rw [frame.sane_depth sane]
  congr 1
  funext cut
  exact exceedsRaw_eq_exceeds moveConfig fc ⟨frame.board, frame.sane_board sane⟩ cut

theorem FrameReturn.thresholdRaw_sound {frame : FrameReturn} {fc : Flip.Dispatch.X86Config}
    (sane : frame.sane = true) (checked : frame.checkThresholdRaw fc = true) :
    frame.BoardSound fc :=
  frame.threshold_sound (by rwa [← frame.checkThresholdRaw_eq fc sane])

theorem FrameReturn.pass_sound {child parent : FrameReturn} {fc : Flip.Dispatch.X86Config}
    (parentSane : parent.sane = true)
    (linked : child.followsPass parent parentId = true) (result : child.BoardSound fc) :
    parent.BoardSound fc := by
  intro valid
  let here : PvsBoardTree.Position := ⟨parent.board, valid⟩
  let there := PvsBoardTree.pass here
  simp only [FrameReturn.followsPass, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at linked
  obtain ⟨⟨⟨⟨noMoves, canMove⟩, swapped⟩, _⟩, call⟩ := linked
  have empty : PvsBoardTree.legalSquares moveConfig here.val = [] := by
    unfold PvsBoardTree.legalSquares
    simp [here, noMoves]
  have goes : Board.hasLegalMoves moveConfig there.val = true := by simpa [there, PvsBoardTree.pass, here, ← swapped] using canMove
  have childValid : Board.wellFormed child.board := swapped.symm ▸ there.property
  have boardEq : (⟨child.board, childValid⟩ : PvsBoardTree.Position) = there := Subtype.ext swapped
  have ⟨a, b, s⟩ := FrameReturn.windowMatches_sound call
  have sound := result childValid
  rw [boardEq, a, b, s] at sound
  have bounds := (Bound.ofSoundResult sound).negate
  simp only [Int.neg_neg] at bounds
  have window := (parent.sane_window parentSane).1
  rw [PvsBoardTree.minimax_pass moveConfig fc here empty goes]
  exact bounds.toSoundResult window (by have := sound.upper; unfold scoreInf; omega)
    (by have := sound.lower; unfold scoreInf; omega)

end NeuralReversi.PvsParallel.TraceCheck

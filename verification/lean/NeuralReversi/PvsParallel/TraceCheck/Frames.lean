import Lean.Data.Json.FromToJson
import NeuralReversi.PvsBoardTree.Threshold

namespace NeuralReversi.PvsParallel.TraceCheck

open Lean
open Aspiration (SoundResult Window)
open PvsMoveLoop (Bound scoreInf)
open PvsBoardTree (GeneratedSquare)

def moveConfig : Bitboard.Moves.X86Config := ⟨false, false⟩
def flipConfig : Flip.Dispatch.X86Config := ⟨false, false, false⟩

structure FrameReturn where
  parent : Nat
  worker : Nat
  depth : Nat
  alpha : Int
  beta : Int
  pv : Bool
  player : Nat
  opponent : Nat
  score : Int
  deriving FromJson

def FrameReturn.board (frame : FrameReturn) : Board.Board :=
  ⟨BitVec.ofNat 64 frame.player, BitVec.ofNat 64 frame.opponent⟩

def FrameReturn.sane (frame : FrameReturn) : Bool :=
  decide (frame.player < 2 ^ 64 ∧ frame.opponent < 2 ^ 64 ∧
    frame.board.player &&& frame.board.opponent = 0) &&
    frame.depth == (Board.emptyCount frame.board).toNat &&
    decide (-scoreInf ≤ frame.alpha ∧ frame.alpha < frame.beta ∧ frame.beta ≤ scoreInf) &&
    (frame.pv || frame.beta == frame.alpha + 1) &&
    decide (-scoreInf < frame.score ∧ frame.score < scoreInf)

theorem FrameReturn.sane_board {frame : FrameReturn} (sane : frame.sane = true) : Board.wellFormed frame.board := by
  simp only [FrameReturn.sane, Bool.and_eq_true, decide_eq_true_eq] at sane
  exact sane.1.1.1.1.2.2

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

def FrameReturn.onMove (child : FrameReturn) (fc : Flip.Dispatch.X86Config) (board : Board.Board)
    (square : Nat) : Bool :=
  if inside : square < 64 then
    let sq : Bitboard.Square := ⟨square, inside⟩
    let flipped := Flip.Dispatch.resultValue fc sq board.player board.opponent
    (Board.getMoves moveConfig board).getLsbD square &&
      decide (child.board = Board.place board flipped sq)
  else false

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

def FrameReturn.windowMatches (frame : FrameReturn) (pv : Bool) (alpha beta score : Int) : Bool :=
  frame.pv == pv && frame.alpha == alpha && frame.beta == beta && frame.score == score

theorem FrameReturn.windowMatches_sound {frame : FrameReturn} {pv : Bool} {alpha beta score : Int}
    (checked : frame.windowMatches pv alpha beta score = true) :
    frame.alpha = alpha ∧ frame.beta = beta ∧ frame.score = score := by
  simp only [FrameReturn.windowMatches, Bool.and_eq_true, beq_iff_eq] at checked
  exact ⟨checked.1.1.2, checked.1.2, checked.2⟩

def FrameReturn.BoardSound (frame : FrameReturn) (fc : Flip.Dispatch.X86Config) : Prop :=
  ∀ valid : Board.wellFormed frame.board,
    SoundResult (PvsBoardTree.minimax moveConfig fc ⟨frame.board, valid⟩) frame.alpha frame.beta frame.score

private def FrameReturn.checkWith (frame : FrameReturn) (exceeds : Int → Bool) : Bool :=
  decide (-scoreInf < frame.score ∧ frame.score < scoreInf) &&
    (if frame.score ≤ frame.alpha then exceeds frame.score == false else true) &&
    (if frame.beta ≤ frame.score then exceeds (frame.score - 1) else true) &&
    (if frame.alpha < frame.score ∧ frame.score < frame.beta then
      exceeds (frame.score - 1) && (exceeds frame.score == false) else true)

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

def FrameReturn.checkBound (frame : FrameReturn) (value : Int) : Bool :=
  frame.checkWith (fun cut => decide (cut < value))

theorem FrameReturn.checkBound_sound {frame : FrameReturn} {value : Int}
    (checked : frame.checkBound value = true) : SoundResult value frame.alpha frame.beta frame.score :=
  frame.checkWith_sound (fun _ => Iff.of_eq decide_eq_true_eq) checked

def FrameReturn.isTerminal (frame : FrameReturn) : Bool :=
  (Board.hasLegalMoves moveConfig frame.board == false) &&
    (Board.hasLegalMoves moveConfig (Board.switchPlayers frame.board) == false)

def FrameReturn.terminalValue (frame : FrameReturn) : Int := 256 * Board.AwardedScore frame.board

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

def FrameReturn.followsPass (child parent : FrameReturn) (parentId : Nat) : Bool :=
  (Board.getMoves moveConfig parent.board == 0 &&
    Board.hasLegalMoves moveConfig child.board &&
    decide (child.board = Board.switchPlayers parent.board)) &&
  (child.parent == parentId && child.worker == parent.worker && child.depth == parent.depth) &&
    child.windowMatches parent.pv (-parent.beta) (-parent.alpha) (-parent.score)

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

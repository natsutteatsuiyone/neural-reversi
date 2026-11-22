import NeuralReversi.EndgameCutoffs
import NeuralReversi.ShallowSearch

namespace NeuralReversi.ShallowBoardTree

open Aspiration (SoundResult Window)
open Bitboard.Moves (X86Config)
open PvsBoardTree (GeneratedSquare Position legalSquares pass play terminalScore)
open PvsMoveLoop (Bound Tree bestValue minimax scoreInf scoreMax)

abbrev LeafEval := Position → {score : Int // -scoreMax ≤ score ∧ score ≤ scoreMax}
abbrev FlipConfig := Flip.Dispatch.X86Config

def wipeoutLoss : Tree := .leaf ⟨-scoreMax, by decide⟩

def build (mc : X86Config) (fc : FlipConfig) (evaluate : LeafEval) (remaining : Nat)
    (board : Position) : Tree :=
  match remaining with
  | 0 => .leaf (evaluate board)
  | depth + 1 =>
    match h : legalSquares mc board.val with
    | [] =>
      let next := pass board
      if other : Board.hasLegalMoves mc next.val = true then .pass (build mc fc evaluate (depth + 1) next)
      else .leaf (terminalScore board)
    | first :: rest =>
      .turn (if EndgameCutoffs.wipesOut fc board.val first.val then wipeoutLoss
        else build mc fc evaluate depth (play mc fc board first).val)
        (rest.map fun square => if EndgameCutoffs.wipesOut fc board.val square.val then wipeoutLoss
          else build mc fc evaluate depth (play mc fc board square).val)
termination_by Board.playRank mc board.val
decreasing_by
  · exact Board.legalCheckedStep_strict mc (.pass ⟨PvsBoardTree.stuck_of_nil h, other⟩) board.property
  · exact (play mc fc board first).property
  · exact (play mc fc board square).property

def value (mc : X86Config) (fc : FlipConfig) (evaluate : LeafEval) (depth : Nat) (board : Position) : Int :=
  minimax (build mc fc evaluate depth board)

def childValue (mc : X86Config) (fc : FlipConfig) (evaluate : LeafEval) (depth : Nat)
    (board : Position) (square : GeneratedSquare mc board.val) : Int :=
  if EndgameCutoffs.wipesOut fc board.val square.val then scoreMax
  else -value mc fc evaluate depth (play mc fc board square).val

theorem value_moves (mc : X86Config) (fc : FlipConfig) (evaluate : LeafEval)
    (depth : Nat) (board : Position) (first : GeneratedSquare mc board.val)
    (rest : List (GeneratedSquare mc board.val)) (moves : legalSquares mc board.val = first :: rest) :
    value mc fc evaluate (depth + 1) board =
      bestValue (childValue mc fc evaluate depth board) (first :: rest) (-scoreInf) := by
  let subtrees := fun square : GeneratedSquare mc board.val =>
    if EndgameCutoffs.wipesOut fc board.val square.val then wipeoutLoss
    else build mc fc evaluate depth (play mc fc board square).val
  have built : build mc fc evaluate (depth + 1) board = .turn (subtrees first) (rest.map subtrees) := by
    rw [build]
    split
    next empty => rw [moves] at empty; cases empty
    next head tail order => rw [moves] at order; cases order; rfl
  rw [value, built, PvsMoveLoop.minimax_turn]
  change bestValue (fun tree => -minimax tree) ((first :: rest).map subtrees) (-scoreInf) = _
  rw [PvsMoveLoop.bestValue_map]
  congr 1
  funext square
  by_cases wiped : EndgameCutoffs.wipesOut fc board.val square.val = true
  · simp [subtrees, childValue, wiped, wipeoutLoss, PvsMoveLoop.minimax]
  · simp [subtrees, childValue, wiped, value]

theorem value_pass (mc : X86Config) (fc : FlipConfig) (evaluate : LeafEval)
    (depth : Nat) (board : Position) (stuck : legalSquares mc board.val = [])
    (other : Board.hasLegalMoves mc (pass board).val = true) :
    value mc fc evaluate (depth + 1) board = -value mc fc evaluate (depth + 1) (pass board) := by
  have built : build mc fc evaluate (depth + 1) board =
      .pass (build mc fc evaluate (depth + 1) (pass board)) := by
    rw [build]
    split
    next empty => simp [other]
    next first rest moves => rw [stuck] at moves; cases moves
  unfold value
  rw [built, PvsMoveLoop.minimax.eq_def]

theorem value_terminal (mc : X86Config) (fc : FlipConfig) (evaluate : LeafEval)
    (depth : Nat) (board : Position) (stuck : legalSquares mc board.val = [])
    (other : Board.hasLegalMoves mc (pass board).val = false) :
    value mc fc evaluate (depth + 1) board = 256 * Board.AwardedScore board.val := by
  have built : build mc fc evaluate (depth + 1) board = .leaf (terminalScore board) := by
    rw [build]
    split
    next empty => simp [other]
    next first rest moves => rw [stuck] at moves; cases moves
  unfold value
  rw [built, PvsMoveLoop.minimax.eq_def]
  rfl

def OrderedTurnContract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (evaluate : LeafEval) (childDepth : Nat) (board : Position)
    (ordered : List (GeneratedSquare mc board.val)) (child : GeneratedSquare mc board.val → Int → Int → Int)
    (alpha beta : Int), Window alpha beta → ordered ≠ [] →
    (∀ square, square ∈ ordered ↔ square ∈ legalSquares mc board.val) →
    (∀ square ∈ ordered, EndgameCutoffs.wipesOut fc board.val square.val = false →
      ∀ a b, Window a b →
        SoundResult (value mc fc evaluate childDepth (play mc fc board square).val) a b (child square a b)) →
    let scored := fun square a => -(child square (-beta) (-a))
    let wipeout := fun square : GeneratedSquare mc board.val => EndgameCutoffs.wipesOut fc board.val square.val
    SoundResult (value mc fc evaluate (childDepth + 1) board) alpha beta
      (ShallowSearch.scan scored wipeout beta ordered alpha (-scoreInf))

theorem orderedTurnContract_proven : OrderedTurnContract := by
  intro mc fc evaluate childDepth board ordered child alpha beta window nonempty order sound scored wipeout
  let v := childValue mc fc evaluate childDepth board
  have same : value mc fc evaluate (childDepth + 1) board = bestValue v ordered (-scoreInf) := by
    cases seen : legalSquares mc board.val with
    | nil =>
      cases list : ordered with
      | nil => exact False.elim (nonempty list)
      | cons head tail =>
        have member := (order head).1 (by simp [list])
        rw [seen] at member
        cases member
    | cons first rest =>
      rw [value_moves mc fc evaluate childDepth board first rest seen]
      apply PvsMoveLoop.bestValue_eq_of_same_moves
      intro square
      simpa only [seen] using (order square).symm
  have range (square : GeneratedSquare mc board.val) : v square ≤ scoreMax := by
    unfold v childValue
    split
    · exact Int.le_refl _
    · have lo := (PvsMoveLoop.minimax_range (build mc fc evaluate childDepth (play mc fc board square).val)).1
      change -minimax (build mc fc evaluate childDepth (play mc fc board square).val) ≤ scoreMax
      omega
  have wins (square : GeneratedSquare mc board.val) (wiped : wipeout square = true) :
      v square = scoreMax := by
    change EndgameCutoffs.wipesOut fc board.val square.val = true at wiped
    simp [v, childValue, wiped]
  rw [same]
  apply ShallowSearch.scanContract_proven _ scored wipeout v ordered alpha beta nonempty window
  · exact fun square _ => range square
  · exact fun square _ => wins square
  · intro square member a active ordinary
    have score := sound square member ordinary (-beta) (-a) (PvsMoveLoop.reflectedWindow active)
    have correct : v square = -value mc fc evaluate childDepth (play mc fc board square).val := by
      simp [v, childValue, wipeout, ordinary]
    rw [correct]
    exact (Bound.ofSoundResult score).negate.toSoundResult active
      (by have := score.upper; unfold scored scoreInf; omega)
      (by have := score.lower; unfold scored scoreInf; omega)

inductive Completed (mc : X86Config) (fc : FlipConfig) (evaluate : LeafEval) :
    Nat → Position → Int → Int → Int → Prop where
  | leaf {board : Position} {alpha beta : Int} : Completed mc fc evaluate 0 board alpha beta (evaluate board).val
  | pass {depth : Nat} {board : Position} {alpha beta child : Int} :
      legalSquares mc board.val = [] → Board.hasLegalMoves mc (PvsBoardTree.pass board).val = true →
      Completed mc fc evaluate (depth + 1) (PvsBoardTree.pass board) (-beta) (-alpha) child →
      Completed mc fc evaluate (depth + 1) board alpha beta (-child)
  | terminal {depth : Nat} {board : Position} {alpha beta : Int} :
      legalSquares mc board.val = [] → Board.hasLegalMoves mc (PvsBoardTree.pass board).val = false →
      Completed mc fc evaluate (depth + 1) board alpha beta (terminalScore board).val
  | wipeout {board : Position} {alpha beta : Int} (square : GeneratedSquare mc board.val) :
      square ∈ legalSquares mc board.val → EndgameCutoffs.wipesOut fc board.val square.val = true →
      Completed mc fc evaluate 2 board alpha beta scoreMax
  | turn {board : Position} {childDepth : Nat} {alpha beta : Int} (ordered : List (GeneratedSquare mc board.val))
      (child : GeneratedSquare mc board.val → Int → Int → Int) :
      ordered ≠ [] → (∀ square, square ∈ ordered ↔ square ∈ legalSquares mc board.val) →
      (childDepth = 0 ∨ ∀ square ∈ ordered, EndgameCutoffs.wipesOut fc board.val square.val = false) →
      (∀ square ∈ ordered, EndgameCutoffs.wipesOut fc board.val square.val = false →
        ∀ a b, Window a b → Completed mc fc evaluate childDepth (play mc fc board square).val a b (child square a b)) →
      Completed mc fc evaluate (childDepth + 1) board alpha beta
        (ShallowSearch.scan (fun square a => -(child square (-beta) (-a)))
          (fun square => EndgameCutoffs.wipesOut fc board.val square.val) beta ordered alpha (-scoreInf))

def ReplayContract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (evaluate : LeafEval) (depth : Nat) (board : Position)
    (alpha beta result : Int), Completed mc fc evaluate depth board alpha beta result → Window alpha beta →
    SoundResult (value mc fc evaluate depth board) alpha beta result

private theorem exact_sound (window : Window alpha beta) (bounds : -scoreMax ≤ score ∧ score ≤ scoreMax) :
    SoundResult score alpha beta score :=
  Bound.toSoundResult window ⟨fun _ => Int.le_refl _, fun _ => Int.le_refl _⟩
    (by unfold scoreMax at bounds; unfold scoreInf; omega) (by unfold scoreMax at bounds; unfold scoreInf; omega)

theorem replayContract_proven : ReplayContract := by
  intro mc fc evaluate depth board alpha beta result completed
  induction completed with
  | leaf =>
    intro window
    rw [value, build, PvsMoveLoop.minimax.eq_def]
    exact exact_sound window (evaluate _).property
  | pass stuck mobile searched ih =>
    intro window
    rw [value_pass _ _ _ _ _ stuck mobile]
    have sound := ih (PvsMoveLoop.reflectedWindow window)
    exact (Bound.ofSoundResult sound).negate.toSoundResult window
      (by have := sound.upper; unfold scoreInf; omega) (by have := sound.lower; unfold scoreInf; omega)
  | terminal stuck mobile =>
    intro window
    rw [value_terminal _ _ _ _ _ stuck mobile]
    exact exact_sound window (terminalScore _).property
  | @wipeout board alpha beta square member wipe =>
    intro window
    have result : value mc fc evaluate 2 board = scoreMax := by
      cases legal : legalSquares mc board.val with
      | nil =>
        rw [legal] at member
        cases member
      | cons first rest =>
        have upper := (PvsMoveLoop.minimax_range (build mc fc evaluate 2 board)).2
        change value mc fc evaluate 2 board ≤ scoreMax at upper
        rw [value_moves mc fc evaluate 1 board first rest legal]
        rw [value_moves mc fc evaluate 1 board first rest legal] at upper
        rw [legal] at member
        have floor := PvsMoveLoop.bestValue_ge_move (childValue mc fc evaluate 1 board)
          (first :: rest) (-scoreInf) member
        simp [childValue, wipe] at floor
        exact Int.le_antisymm upper floor
    rw [result]
    exact exact_sound window (by decide)
  | turn ordered child nonempty same _ searched ih =>
    intro window
    exact orderedTurnContract_proven mc fc evaluate _ _ ordered child _ _ window nonempty same
      (fun square member ordinary a b good => ih square member ordinary a b good good)

def TwoPlyContract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (evaluate : LeafEval) (board : Position)
    (pcBeta result : Int), Window (pcBeta - 1) pcBeta →
    Completed mc fc evaluate 2 board (pcBeta - 1) pcBeta result →
    (pcBeta ≤ result ↔ pcBeta ≤ value mc fc evaluate 2 board) ∧
    (pcBeta ≤ result ↔ pcBeta ≤ ShallowSearch.run (build mc fc evaluate 2 board) (pcBeta - 1) pcBeta)

theorem twoPlyContract_proven : TwoPlyContract := by
  intro mc fc evaluate board pcBeta result window trace
  have verified := replayContract_proven mc fc evaluate 2 board _ _ result trace window
  have modeled := ShallowSearch.contract_proven (build mc fc evaluate 2 board) _ _ window
  have accepted := ShallowSearch.reachesThreshold _ _ _ verified
  exact ⟨accepted, accepted.trans (ShallowSearch.reachesThreshold _ _ _ modeled).symm⟩

example : value ⟨false, false⟩ ⟨false, false, false⟩ (fun _ => ⟨1, by decide⟩) 1
    ⟨⟨0xfffe#64, 0xffffffffffff0000#64⟩, by unfold Board.wellFormed; decide⟩ = 1 := by decide +kernel

end NeuralReversi.ShallowBoardTree

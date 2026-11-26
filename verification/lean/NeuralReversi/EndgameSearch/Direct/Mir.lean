import NeuralReversi.EndgameSearch.Direct.EmptySquares
import NeuralReversi.EndgameSearch.Direct.Mir.Program

namespace NeuralReversi.EndgameSearch.Direct.Mir

open PvsBoardTree (Position minimax)
open ScoreRefresh (Score DiscScore toDiscDiff fromDiscDiff)
open Bitboard (Square)
open EmptyListLinks (Links head)
open Flip.Dispatch (TargetConfig)

private def args? (board : Board.Board) (n : Nat) (squares : List Square) : Option (Solve34.EmptyBoard n) :=
  if valid : squares.length = n ∧ squares.Nodup ∧ board.player &&& board.opponent = 0 ∧
      Board.emptySquares board = Solve34.emptyMask squares then
    some ⟨board.player, board.opponent, squares, valid.1, valid.2.1, valid.2.2.1, valid.2.2.2⟩
  else none

private theorem args_recover (arg : Solve34.EmptyBoard n) :
    args? arg.asBoard n arg.squares = some arg := by
  cases arg with
  | mk p o slots length distinct disjoint covered =>
    have same : Board.emptySquares ⟨p, o⟩ = _ := covered
    simp [args?, Solve34.EmptyBoard.asBoard, length, distinct, disjoint, same]

private def one? (target : TargetConfig) (bmi2 : Bool) (player : BitVec 64)
    (square : Square) (alpha : BitVec 32) : Option (BitVec 32) :=
  if empty : player.getLsbD square.val = false then
    some (Solve2.solve1On target bmi2 ⟨player, square, empty⟩ alpha)
  else none

private theorem one_recover (target : TargetConfig) (bmi2 : Bool) (arg : Solve1.Bmi2.Board)
    (alpha : BitVec 32) : one? target bmi2 arg.player arg.square alpha =
      some (Solve2.solve1On target bmi2 arg alpha) := by
  cases arg with
  | mk player sq empty => simp only [one?, empty, dite_true]

def discResult (arm : Arm) (target : TargetConfig) (bmi2 : Bool) (board : Board.Board)
    (nodes : Links) (parity : BitVec 8) (checked : Bool) (alpha : BitVec 32) : Option DiscScore := do
  match arm with
  | .terminal => (ScoreRefresh.BoardBridge.finalDiscDiff checked board).toOption
  | .solve1 =>
    let sq ← (EmptyListTraversal.squareAfter nodes head).toOption
    return ⟨← one? target bmi2 board.player sq alpha⟩
  | .solve2 =>
    let slots ← (readSquares nodes head 2).toOption
    let arg ← args? board 2 slots
    return ⟨(Solve2.targetRun target bmi2 arg.toTwoEmpty alpha).score⟩
  | .solve3 =>
    let slots ← (readSquares nodes head 3).toOption
    let arg ← args? board 3 slots
    return ⟨(Solve34.target3Run target bmi2 arg alpha).score⟩
  | .solve4 =>
    let slots ← SortLastFour.sortLastFour nodes parity
    let arg ← args? board 4 slots
    return ⟨(Solve34.target4Run target bmi2 arg alpha).score⟩

def run (target : TargetConfig) (bmi2 : Bool) (board : Board.Board) (nodes : Links)
    (parity : BitVec 8) (count : BitVec 32) (alpha : Score) (checked : Bool) : Option Score := do
  let arm ← dispatch count.toNat
  let result ← discResult arm target bmi2 board nodes parity checked (toDiscDiff alpha).raw
  (fromDiscDiff checked result).toOption

private theorem scaled_fixed (mc : Bitboard.Moves.X86Config) (fc : FlipConfig) (alpha : Score)
    (checked : Bool) (fixed : FixedDisc target bmi2 b (toDiscDiff alpha).raw word) :
    ∃ score, fromDiscDiff checked ⟨word⟩ = .ok score ∧ Returned target bmi2 mc fc b alpha checked score := by
  have floor := ScoreRefresh.disc_floor alpha
  have low := BitVec.le_toInt alpha.raw
  have high := BitVec.toInt_lt (x := alpha.raw)
  have safe : Solve34.WindowSafe (toDiscDiff alpha).raw := by
    unfold Solve34.WindowSafe
    rw [floor]
    omega
  have ⟨range, _⟩ := fixed.bounds mc fc safe
  have within : ¬(word.toInt < -127 ∨ word.toInt > 127) := by obtain ⟨_, _⟩ := range; omega
  have converted : fromDiscDiff checked ⟨word⟩ = .ok ⟨word <<< 8⟩ := by simp [fromDiscDiff, within]
  exact ⟨_, converted, .fixed ⟨word⟩ fixed converted⟩

theorem run_returns (target : TargetConfig) (bmi2 : Bool) (mc : Bitboard.Moves.X86Config)
    (fc : FlipConfig) (b : Position) (nodes : Links) (parity : BitVec 8) (count : BitVec 32)
    (alpha : Score) (checked : Bool) (aligned : EmptyStateMatches b nodes parity count) (size : count.toNat ≤ 4) :
    ∃ score, run target bmi2 b.val nodes parity count alpha checked = some score ∧
      Returned target bmi2 mc fc b alpha checked score := by
  obtain ⟨one, two, three, four⟩ := fixedOutputs target bmi2 b (toDiscDiff alpha).raw aligned
  have cases : count.toNat = 0 ∨ count.toNat = 1 ∨ count.toNat = 2 ∨ count.toNat = 3 ∨ count.toNat = 4 := by omega
  rcases cases with n | n | n | n | n
  · have full : PvsParallel.ExactDepth.Full b 0 := by
      change 0 = Board.emptyCount b.val
      apply BitVec.eq_of_toNat_eq
      rw [← aligned.1, n]
      rfl
    have valid := (PvsParallel.ExactDepth.zeroDepthContract_proven mc fc b checked full).2.2
    cases h : ScoreRefresh.BoardBridge.finalScore checked b.val with
    | error e => rw [h] at valid; cases valid
    | ok s =>
      refine ⟨s, ?_, .zero full h⟩
      unfold ScoreRefresh.BoardBridge.finalScore at h
      cases d : ScoreRefresh.BoardBridge.finalDiscDiff checked b.val with
      | error e => rw [d] at h; cases h
      | ok discs =>
        rw [d] at h
        change fromDiscDiff checked discs = .ok s at h
        simp [run, n, dispatch, discResult, d, h, Except.toOption]
  · obtain ⟨arg, read, player, fixed⟩ := one n
    obtain ⟨s, converted, sound⟩ := scaled_fixed mc fc alpha checked fixed
    refine ⟨s, ?_, sound⟩
    simp [run, n, dispatch, discResult, read, ← player, one_recover, converted, Except.toOption]
  · obtain ⟨arg, read, board, fixed⟩ := two n
    obtain ⟨s, converted, sound⟩ := scaled_fixed mc fc alpha checked fixed
    refine ⟨s, ?_, sound⟩
    simp [run, n, dispatch, discResult, read, ← board, args_recover, converted, Except.toOption]
  · obtain ⟨arg, read, board, fixed⟩ := three n
    obtain ⟨s, converted, sound⟩ := scaled_fixed mc fc alpha checked fixed
    refine ⟨s, ?_, sound⟩
    simp [run, n, dispatch, discResult, read, ← board, args_recover, converted, Except.toOption]
  · obtain ⟨arg, read, board, fixed⟩ := four n
    obtain ⟨s, converted, sound⟩ := scaled_fixed mc fc alpha checked fixed
    refine ⟨s, ?_, sound⟩
    simp [run, n, dispatch, discResult, read, ← board, args_recover, converted, Except.toOption]

def ScoreContract : Prop :=
  ∀ target bmi2 mc fc (b : Position) nodes parity count (alpha beta : Score) checked,
    EmptyStateMatches b nodes parity count → count.toNat ≤ 4 → beta.raw.toInt = alpha.raw.toInt + 1 →
    ∃ score, run target bmi2 b.val nodes parity count alpha checked = some score ∧
      PvsMoveLoop.Bound (minimax mc fc b) alpha.raw.toInt beta.raw.toInt score.raw.toInt

theorem scoreContract_proven : ScoreContract := by
  intro target bmi2 mc fc b nodes parity count alpha beta checked aligned size null
  obtain ⟨score, result, returned⟩ := run_returns target bmi2 mc fc b nodes parity count alpha checked aligned size
  exact ⟨score, result, returned.bound beta null⟩

end NeuralReversi.EndgameSearch.Direct.Mir

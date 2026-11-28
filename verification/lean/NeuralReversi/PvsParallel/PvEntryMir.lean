module
public import NeuralReversi.PvsParallel.NonPvMir.Program
public import NeuralReversi.PvsParallel.PvEntryMir.Program
public import NeuralReversi.PvsParallel
public import NeuralReversi.MoveList.Generation

@[expose] public section

namespace NeuralReversi.PvsParallel.PvEntryMir

open PvsBoardTree (Position GeneratedSquare legalSquares build pass terminalScore)
open Bitboard.Moves (X86Config)
open Cutoffs (FlipConfig)
open MoveList
open ScoreRefresh.BoardBridge (solve)

def generated (mc : X86Config) (fc : FlipConfig) (b : Position) (paired : Bool)
    (mem : MoveList.Initialization.Memory Move.Move) :=
  Generation.fill false paired fc b.val (Board.getMoves mc b.val) mem

theorem generated_count (mc : X86Config) (fc : FlipConfig) (b : Position) (paired : Bool) (mem) :
    (generated mc fc b paired mem).len = some (legalSquares mc b.val).length := by
  have count := (Generation.generationContract_proven mc fc b.val (Board.getMoves mc b.val) false paired mem
    b.property (fun _ h => h) (by simp)).1
  have squares := congrArg List.length (PvsBoardTree.legalSquares_values mc b.val)
  apply count.trans
  apply congrArg some
  simpa [Generation.records, Generation.squares, Function.comp_def] using squares.symm

theorem cached_wipe (mc : X86Config) (fc : FlipConfig) (b : Position) (paired : Bool) (mem)
    (found : (generated mc fc b paired mem).wipeout = some (some square)) :
    ∃ m : GeneratedSquare mc b.val, m ∈ legalSquares mc b.val ∧
      EndgameCutoffs.wipesOut fc b.val m.val = true := by
  have cache := (Generation.generationContract_proven mc fc b.val (Board.getMoves mc b.val) false paired mem
    b.property (fun _ h => h) (by simp)).2.2.2.2
  rw [show (Generation.fill false paired fc b.val (Board.getMoves mc b.val) mem).wipeout = _ from found] at cache
  obtain ⟨candidate, result, _⟩ := Option.map_eq_some_iff.mp (Option.some.inj cache).symm
  have comparison := List.find?_some result
  have entryMem := List.mem_reverse.mp (List.mem_of_find?_eq_some result)
  obtain ⟨entry, included, rfl⟩ := List.mem_map.mp entryMem
  obtain ⟨sq, member, rfl⟩ := List.mem_map.mp included
  have bit : (Board.getMoves mc b.val).getLsbD sq.val = true := (List.mem_filter.mp member).2
  obtain ⟨m, legal, same⟩ := (PvsBoardTree.legalSquares_complete mc b.val sq).mpr
    ((Board.getMoves_exact mc b.val sq.val).mp bit)
  exact ⟨m, legal, same ▸ comparison⟩

def run (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : ExactDepth.Depth)
    (paired checked : Bool) (mem : MoveList.Initialization.Memory Move.Move) (alpha beta : Int)
    (child : Position → ExactDepth.Depth → Int → Int → Option Int) : Option Int := do
  let moves := generated mc fc b paired mem
  let act := dispatch depth.toNat (moves.len.getD 0 != 0) (Board.hasLegalMoves mc (pass b).val)
    (moves.wipeout.join.isSome)
  match act with
  | .zero => return (← (NonPvMir.pvZeroScore checked b.val).toOption).raw.toInt
  | .terminal => return (← (solve checked b.val depth).toOption).raw.toInt
  | .pass => return -(← child (pass b) depth (-beta) (-alpha))
  | .wipeout => some PvsMoveLoop.scoreMax
  | .continueSearch => none

theorem terminal_solve (b : Position) (checked : Bool) :
    (solve checked b.val (Board.emptyCount b.val)).map (fun s => s.raw.toInt) =
      .ok (terminalScore b).val := by
  obtain ⟨discs, release, debug, eq, lo, hi⟩ := ScoreRefresh.BoardBridge.solveDiscDiff_awarded b.val b.property
  have within : ¬ (discs.raw.toInt < -127 ∨ discs.raw.toInt > 127) := by omega
  obtain ⟨score, cast, _, scaled, _⟩ := ScoreRefresh.bounded_roundtrip discs (by omega) (by omega)
  have raw : ScoreRefresh.fromDiscDiff checked discs = .ok score := by
    simpa [ScoreRefresh.fromDiscDiff, within] using cast
  have disc : ScoreRefresh.BoardBridge.solveDiscDiff checked b.val (Board.emptyCount b.val) = .ok discs := by
    cases checked <;> assumption
  simp [solve, disc, raw, Except.map, terminalScore, scaled, eq, Int.mul_comm]

theorem exitExecutionWith (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : ExactDepth.Depth)
    (paired checked : Bool) (mem : MoveList.Initialization.Memory Move.Move) (alpha beta : Int)
    (child : Position → ExactDepth.Depth → Int → Int → Option Int) (full : ExactDepth.Full b depth)
    (calls : legalSquares mc b.val = [] → Board.hasLegalMoves mc (pass b).val = true →
      ∀ answer, child (pass b) depth (-beta) (-alpha) = some answer →
      children .serial (build mc fc (pass b)) true (-beta) (-alpha) answer)
    (answer : run mc fc b depth paired checked mem alpha beta child = some score)
    (wipeout : ∀ m ∈ legalSquares mc b.val, EndgameCutoffs.wipesOut fc b.val m.val = true →
      exit mc fc b true alpha beta PvsMoveLoop.scoreMax) :
    ExecutionWithUses exit children (build mc fc b) true alpha beta score := by
  simp only [run, generated_count, Option.getD_some] at answer
  by_cases zero : depth.toNat = 0
  · have word : depth = 0 := BitVec.eq_of_toNat_eq zero
    have ⟨stuck, other, value⟩ := ExactDepth.zeroDepthContract_proven mc fc b checked (word ▸ full)
    have tree := PvsBoardTree.terminal_tree mc fc b (PvsBoardTree.nil_of_stuck stuck) other
    cases eval : NonPvMir.pvZeroScore checked b.val with
    | error reason => simp [dispatch, zero, eval, Except.toOption] at answer
    | ok s =>
      simp [dispatch, zero, eval, Except.toOption] at answer
      change ScoreRefresh.BoardBridge.finalScore checked b.val = .ok s at eval
      simp [eval, Except.map] at value
      rw [← answer, value, tree]
      unfold PvsBoardTree.minimax
      rw [tree, PvsMoveLoop.minimax.eq_def]
      exact .leaf
  · cases moves : legalSquares mc b.val with
    | nil =>
      cases other : Board.hasLegalMoves mc (pass b).val with
      | false =>
        cases eval : solve checked b.val depth with
        | error reason => simp [moves, other, dispatch, zero, eval, Except.toOption] at answer
        | ok s =>
          have value := terminal_solve b checked
          rw [← full, eval] at value
          simp [Except.map] at value
          simp [moves, other, dispatch, zero, eval, Except.toOption] at answer
          rw [← answer, value, PvsBoardTree.terminal_tree mc fc b moves other]
          exact .leaf
      | true =>
        cases response : child (pass b) depth (-beta) (-alpha) with
        | none => simp [moves, other, dispatch, zero, response] at answer
        | some got =>
          simp [moves, other, dispatch, zero, response] at answer
          have tree : build mc fc b = .pass (build mc fc (pass b)) := by
            rw [build.eq_def]
            split
            next eq => rw [moves] at eq; cases eq
            next => simp [other]
          rw [tree, ← answer]
          exact .pass (calls moves other got response)
    | cons first rest =>
      cases cache : (generated mc fc b paired mem).wipeout with
      | none => simp [moves, dispatch, zero, cache] at answer
      | some item =>
        cases item with
        | none => simp [moves, dispatch, zero, cache] at answer
        | some square =>
          simp [moves, dispatch, zero, cache] at answer
          obtain ⟨m, member, wipe⟩ := cached_wipe mc fc b paired mem cache
          rw [← answer]
          exact .cutoff mc fc b (wipeout m member wipe)

theorem exitExecution (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : ExactDepth.Depth)
    (paired checked : Bool) (mem : MoveList.Initialization.Memory Move.Move) (alpha beta : Int)
    (child : Position → ExactDepth.Depth → Int → Int → Option Int) (full : ExactDepth.Full b depth)
    (calls : legalSquares mc b.val = [] → Board.hasLegalMoves mc (pass b).val = true →
      ∀ answer, child (pass b) depth (-beta) (-alpha) = some answer →
      children .serial (build mc fc (pass b)) true (-beta) (-alpha) answer)
    (answer : run mc fc b depth paired checked mem alpha beta child = some score) :
    ExecutionWithUses Cutoffs.Execution children (build mc fc b) true alpha beta score :=
  exitExecutionWith mc fc b depth paired checked mem alpha beta child full calls answer
    (fun m legal wipes => .wipeout m legal wipes)

theorem stuck_returns (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : ExactDepth.Depth)
    (paired checked : Bool) (mem : MoveList.Initialization.Memory Move.Move) (alpha beta : Int)
    (child : Position → ExactDepth.Depth → Int → Int → Option Int) (full : ExactDepth.Full b depth)
    (stuck : legalSquares mc b.val = []) (other : Board.hasLegalMoves mc (pass b).val = true →
      ∃ score, child (pass b) depth (-beta) (-alpha) = some score) :
    ∃ score, run mc fc b depth paired checked mem alpha beta child = some score := by
  simp only [run, generated_count, stuck, List.length_nil, Option.getD_some]
  by_cases zero : depth.toNat = 0
  · have word : depth = 0 := BitVec.eq_of_toNat_eq zero
    have value := (ExactDepth.zeroDepthContract_proven mc fc b checked (word ▸ full)).2.2
    cases eval : NonPvMir.pvZeroScore checked b.val with
    | error reason =>
      change ScoreRefresh.BoardBridge.finalScore checked b.val = _ at eval
      simp [eval, Except.map] at value
    | ok s => exact ⟨s.raw.toInt, by simp [dispatch, zero, Except.toOption]⟩
  · cases mobile : Board.hasLegalMoves mc (pass b).val with
    | true =>
      obtain ⟨v, response⟩ := other mobile
      exact ⟨-v, by simp [dispatch, zero, response]⟩
    | false =>
      have result := terminal_solve b checked
      rw [← full] at result
      cases eval : solve checked b.val depth with
      | error e => simp [eval, Except.map] at result
      | ok s => exact ⟨s.raw.toInt, by simp [dispatch, zero, Except.toOption]⟩

end NeuralReversi.PvsParallel.PvEntryMir

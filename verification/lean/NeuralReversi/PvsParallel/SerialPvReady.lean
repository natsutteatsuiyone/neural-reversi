import NeuralReversi.PvsParallel.SerialPvCalls

namespace NeuralReversi.PvsParallel.SerialPvReady

open SerialCalls (Program Request)
open PvsBoardTree (Position legalSquares pass)
open SerialPvCalls (frame action moveRequest)
open Aspiration (Window)

theorem frame_ready {mc : Bitboard.Moves.X86Config} {fc : Cutoffs.FlipConfig} {b : Position}
    (full : ExactDepth.Full b d) (window : Window a z) (paired checked cut : Bool)
    (mem : MoveList.Initialization.Memory Move.Move) (moves : List (PvsBoardTree.GeneratedSquare mc b.val)) :
    Program.Ready (fun r => ExactDepth.RecursiveCall mc fc b d r.board r.depth ∧ Window r.alpha r.beta ∧
      (r.pv = false → r.beta = r.alpha+1)) (frame mc fc b d paired checked cut mem a z moves) := by
  unfold frame
  cases which : action mc fc b d paired mem
  case continueSearch =>
    apply Program.Ready.map
    apply Program.Ready.mapRequests
    exact (SerialOwnerCalls.run_ready window moves).weaken fun q h =>
      ⟨.ply b q.move d, h⟩
  case pass =>
    have dispatch := which
    simp only [action, PvEntryMir.dispatch, PvEntryMir.generated_count, Option.getD_some] at dispatch
    have forced : Board.ForcedPass mc b.val := by
      split at dispatch
      · contradiction
      · split at dispatch
        · split at dispatch <;> contradiction
        · rename_i absent
          split at dispatch
          · rename_i mobile
            exact ⟨PvsBoardTree.stuck_of_nil (by simpa using absent), mobile⟩
          · contradiction
    exact .call ⟨.forcedPass b d forced, PvsMoveLoop.reflectedWindow window, by simp⟩ fun _ => .pure
  case zero =>
    have dz : d.toNat = 0 := by
      by_cases nz : d.toNat = 0
      · exact nz
      · simp only [action, PvEntryMir.dispatch, nz, ite_false] at which
        repeat' split at which
        all_goals contradiction
    have bits : d = 0 := BitVec.eq_of_toNat_eq dz
    have value := (ExactDepth.zeroDepthContract_proven mc fc b checked (bits ▸ full)).2.2
    cases eval : NonPvMir.pvZeroScore checked b.val with
    | error reason =>
      change ScoreRefresh.BoardBridge.finalScore checked b.val = _ at eval
      simp [eval, Except.map] at value
    | ok score =>
      have got : PvEntryMir.run mc fc b d paired checked mem a z (fun _ _ _ _ => none) = some score.raw.toInt := by
        simp [PvEntryMir.run, PvEntryMir.dispatch, dz, eval, Except.toOption]
      rw [got]
      exact .pure
  case terminal =>
    have result := PvEntryMir.terminal_solve b checked
    rw [← full] at result
    cases eval : ScoreRefresh.BoardBridge.solve checked b.val d with
    | error reason => simp [eval, Except.map] at result
    | ok score =>
      have got : PvEntryMir.run mc fc b d paired checked mem a z (fun _ _ _ _ => none) = some score.raw.toInt := by
        unfold PvEntryMir.run
        change (match action mc fc b d paired mem with
          | .zero => _ | .terminal => _ | .pass => _ | .wipeout => _ | .continueSearch => _) = _
        rw [which, eval]
        rfl
      rw [got]
      exact .pure
  case wipeout =>
    have got : PvEntryMir.run mc fc b d paired checked mem a z (fun _ _ _ _ => none) = some PvsMoveLoop.scoreMax := by
      unfold PvEntryMir.run
      change (match action mc fc b d paired mem with
        | .zero => _ | .terminal => _ | .pass => _ | .wipeout => _ | .continueSearch => _) = _
      rw [which]
    rw [got]
    exact .pure

end NeuralReversi.PvsParallel.SerialPvReady

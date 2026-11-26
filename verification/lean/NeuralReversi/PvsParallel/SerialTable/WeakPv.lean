import NeuralReversi.PvsParallel.SerialTable.Stack

namespace NeuralReversi.PvsParallel.SerialTable

open SerialCalls (Program Request)
open SerialCalls.ScoreWords (Finite)
open PvsBoardTree (Position GeneratedSquare)
open SerialPvCalls (action)

theorem pv_finite {mc : Bitboard.Moves.X86Config} {fc : Cutoffs.FlipConfig} {b : Position}
    {d : ExactDepth.Depth} {mem : MoveList.Initialization.Memory Move.Move} {moves : List (GeneratedSquare mc b.val)}
    (full : ExactDepth.Full b d) (order : moves.Perm (PvsBoardTree.legalSquares mc b.val)) (window : Aspiration.Window a z)
    (ran : Program.Runs (Writes.pv mc fc b d paired checked cut mem a z moves) cs out)
    (children : ∀ t ∈ cs, Finite t.2) : Finite out.score := by
  have run := Stack.runs_map ran Writes.Result.score
  rw [Writes.pv_score] at run
  generalize out.score = score at run ⊢
  unfold SerialPvCalls.frame at run
  cases which : action mc fc b d paired mem <;> rw [which] at run
  case zero | terminal | wipeout =>
    have replay : boundedReplay 1 (PvsBoardTree.build mc fc b) true a z score := PvEntryMir.exitExecution mc fc b d paired checked mem a z (fun _ _ _ _ => none)
      full (by simp) (Program.runs_ofOption run).1
    exact SerialCalls.ScoreWords.finite_of_replay window (by simp) replay
  case pass =>
    cases run with
    | call more =>
      cases more
      have word := children _ List.mem_cons_self
      dsimp only [Prod.snd] at word
      unfold Finite at *
      omega
  case continueSearch =>
    obtain ⟨owner,requests,eq⟩ := Program.runs_map run
    obtain ⟨words,issued,rfl⟩ := Program.runs_mapRequests requests
    have nonempty : moves ≠ [] := by
      intro empty
      have missing : PvsBoardTree.legalSquares mc b.val = [] := List.Perm.eq_nil (empty ▸ order.symm)
      cases other : Board.hasLegalMoves mc (PvsBoardTree.pass b).val <;> by_cases zero : d.toNat = 0 <;>
        simp [action,PvEntryMir.dispatch,PvEntryMir.generated_count,missing,other,zero] at which
    rw [← eq]
    exact SerialOwnerCalls.run_finite nonempty issued fun t ht =>
      children _ (List.mem_map_of_mem (f := fun r : SerialCalls.MoveRequest (GeneratedSquare mc b.val) × Int =>
        (SerialPvCalls.moveRequest b mc fc d cut r.1,r.2)) ht)

end NeuralReversi.PvsParallel.SerialTable

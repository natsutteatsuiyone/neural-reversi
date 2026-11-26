import NeuralReversi.PvsParallel.PvSerialMir
import NeuralReversi.PvsParallel.Replay
import NeuralReversi.PvsParallel.RootOwnerMir.Serial

namespace NeuralReversi.PvsParallel.SerialPv

open Aspiration (Window)
open Bitboard (Square)
open PvsBoardTree (Position GeneratedSquare legalSquares build play pass)
open RootOwnerMir (Replies)
open RootOwnerMir.Serial (schedule owner_returns)

def runFrame (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (depth : ExactDepth.Depth) (paired checked : Bool) (mem : MoveList.Initialization.Memory Move.Move)
    (alpha beta : Int) (passCall : Position → ExactDepth.Depth → Int → Int → Option Int)
    (moves : List (GeneratedSquare mc b.val)) (replies : Replies (GeneratedSquare mc b.val)) : Option Int :=
  PvSerialMir.scheduled mc fc b depth paired checked mem alpha beta passCall moves replies schedule (fun _ => none)

/-- Given the smaller calls, the serial PV frame returns a score and constructs
its replay, including the stop-free owner loop at arbitrary depth. -/
theorem frame_returns (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (depth : ExactDepth.Depth) (paired checked : Bool) (mem : MoveList.Initialization.Memory Move.Move)
    (alpha beta : Int) (passCall : Position → ExactDepth.Depth → Int → Int → Option Int)
    (moves : List (GeneratedSquare mc b.val)) (replies : Replies (GeneratedSquare mc b.val))
    (frames : Nat) (full : ExactDepth.Full b depth) (order : moves.Perm (legalSquares mc b.val))
    (window : Window alpha beta)
    (passes : legalSquares mc b.val = [] → Board.hasLegalMoves mc (pass b).val = true →
      ∃ v, passCall (pass b) depth (-beta) (-alpha) = some v ∧ boundedReplay frames (build mc fc (pass b)) true (-beta) (-alpha) v)
    (calls : ∀ m pv a z, Window a z → (pv = false → z = a + 1) →
      ∃ v, replies m pv a z = some v ∧ boundedReplay frames (build mc fc (play mc fc b m).val) pv a z v) :
    ∃ score, runFrame mc fc b depth paired checked mem alpha beta passCall moves replies = some score ∧
      boundedReplay (frames + 1) (build mc fc b) true alpha beta score := by
  have passTrace (stuck : legalSquares mc b.val = []) (mobile : Board.hasLegalMoves mc (pass b).val = true)
      (v : Int) (returned : passCall (pass b) depth (-beta) (-alpha) = some v) :
      boundedReplay frames (build mc fc (pass b)) true (-beta) (-alpha) v := by
    obtain ⟨w, got, trace⟩ := passes stuck mobile
    exact Option.some.inj (got.symm.trans returned) ▸ trace
  have leaf (score : Int) (returned : PvEntryMir.run mc fc b depth paired checked mem alpha beta passCall = some score) :
      boundedReplay (frames + 1) (build mc fc b) true alpha beta score :=
    PvEntryMir.exitExecution (children := fun _ => boundedReplay frames) mc fc b depth paired checked mem
      alpha beta passCall full passTrace returned
  cases legal : legalSquares mc b.val with
  | nil =>
    obtain ⟨score, returned⟩ := PvEntryMir.stuck_returns mc fc b depth paired checked mem alpha beta passCall full legal
      (fun mobile => (passes legal mobile).imp (fun _ h => h.1))
    refine ⟨score, ?_, leaf score returned⟩
    by_cases zero : depth.toNat = 0
    · simpa [runFrame, PvSerialMir.scheduled, PvEntryMir.dispatch, PvEntryMir.generated_count, legal, zero] using returned
    · cases mobile : Board.hasLegalMoves mc (pass b).val <;>
        simpa [runFrame, PvSerialMir.scheduled, PvEntryMir.dispatch, PvEntryMir.generated_count, legal, zero, mobile] using returned
  | cons first rest =>
    have nonzero : depth.toNat ≠ 0 := by
      intro zero
      have depthZero : depth = 0 := BitVec.eq_of_toNat_eq zero
      have stuck := (ExactDepth.zeroDepthContract_proven mc fc b checked (depthZero ▸ full)).1
      have impossible := legal.symm.trans (PvsBoardTree.nil_of_stuck stuck)
      cases impossible
    cases wipes : (PvEntryMir.generated mc fc b paired mem).wipeout.join.isSome with
    | true =>
      have returned : PvEntryMir.run mc fc b depth paired checked mem alpha beta passCall =
          some PvsMoveLoop.scoreMax := by
        simp [PvEntryMir.run, PvEntryMir.dispatch, PvEntryMir.generated_count, legal, nonzero, wipes]
      exact ⟨_, by simpa [runFrame, PvSerialMir.scheduled, PvEntryMir.dispatch, PvEntryMir.generated_count,
        legal, nonzero, wipes] using returned, leaf _ returned⟩
    | false =>
      obtain ⟨_, out, _, _, ran, joined, snapshot, consumed⟩ := owner_returns Subtype.val replies
        (fun m p a z v => boundedReplay frames (build mc fc (play mc fc b m).val) p a z v)
        depth.toNat rootMinimumSplitDepth alpha beta moves window calls
      refine ⟨out.snapshot.state.best, ?_, ?_⟩
      · simp [runFrame, PvSerialMir.scheduled, PvEntryMir.dispatch, PvEntryMir.generated_count, legal,
          nonzero, wipes, ran, joined]
      · rw [snapshot, RootReplay.serial_best]
        change Execution (boundedReplay frames) (build mc fc b) true alpha beta _
        rw [build.eq_def]
        split
        next head tail members =>
          obtain ⟨rfl,rfl⟩ := List.cons.inj (legal.symm.trans members)
          rw [legal] at order
          exact .sequential moves (fun m => build mc fc (play mc fc b m).val)
            (order.map _) (RootOwnerMir.observed replies) consumed
        next absent => rw [legal] at absent; cases absent

end NeuralReversi.PvsParallel.SerialPv

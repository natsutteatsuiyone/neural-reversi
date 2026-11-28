module
public import NeuralReversi.AspirationMir.SerialTable.PvInterior
public import NeuralReversi.PvsParallel.SerialTable.WeakTrial
public import NeuralReversi.PvsParallel.SerialTable.ExactScores

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialTable.WeakTrial

open AspirationMir.SerialTable.PvInterior (InsideRange)
open AspirationMir.SerialTable
open SerialCalls (Program Request)
open PvsBoardTree (Position GeneratedSquare)
open RootMoveStorage (Row)
open PvsMoveLoop (scoreMax)

def Range (q : Request) (word : Int) : Prop := q.pv = true → InsideRange q.alpha q.beta word

theorem pv_range {mc : Bitboard.Moves.X86Config} {fc : Cutoffs.FlipConfig} {b : Position}
    {d : ExactDepth.Depth} {mem : MoveList.Initialization.Memory Move.Move} {moves : List (GeneratedSquare mc b.val)}
    (full : ExactDepth.Full b d) (order : moves.Perm (PvsBoardTree.legalSquares mc b.val)) (window : Aspiration.Window a z)
    (ran : Program.Runs (Writes.pv mc fc b d paired checked cut mem a z moves) cs out)
    (children : ∀ t ∈ cs, Range t.1 t.2) : InsideRange a z out.score := by
  have run := Stack.runs_map ran Writes.Result.score
  rw [Writes.pv_score] at run
  generalize out.score = score at run ⊢
  unfold SerialPvCalls.frame at run
  cases which : SerialPvCalls.action mc fc b d paired mem <;> rw [which] at run
  case zero | terminal | wipeout =>
    have replay : boundedReplay 1 (PvsBoardTree.build mc fc b) true a z score :=
      PvEntryMir.exitExecution mc fc b d paired checked mem a z (fun _ _ _ _ => none) full (by simp) (Program.runs_ofOption run).1
    have sound := replayContract_proven _ _ _ _ _ _ window (by simp) replay
    exact fun lo hi => (sound.exact lo hi) ▸ PvsMoveLoop.minimax_range _
  case pass =>
    cases run with
    | call more =>
      cases more
      have closed := children _ List.mem_cons_self rfl
      intro lo hi
      obtain ⟨small,big⟩ := closed (by change -z < _; omega) (by change _ < -a; omega)
      dsimp only [Prod.snd] at small big
      omega
  case continueSearch =>
    obtain ⟨owner,requests,eq⟩ := Program.runs_map run
    obtain ⟨words,issued,rfl⟩ := Program.runs_mapRequests requests
    rw [← eq]
    exact PvInterior.owner (order.nodup_iff.mpr (EndgameOrderMir.generated_layout mc b).1) issued window
      fun t ht => children _ (List.mem_map_of_mem (f := fun x : SerialCalls.MoveRequest (GeneratedSquare mc b.val) × Int =>
        (SerialPvCalls.moveRequest b mc fc d cut x.1,x.2)) ht)

theorem root_range {mc fc b d paired mem rows root a z} (ran : Program.Runs (Root.frame mc fc b d paired mem rows idx root a z) cs out)
    (window : Aspiration.Window a.toInt z.toInt) (children : ∀ t ∈ cs, Range t.1 t.2) :
    InsideRange a.toInt z.toInt out.answer.value := by
  dsimp only [Root.frame] at ran
  split at ran
  · cases ran
  next sq _ =>
    cases ran
    intro _ _
    change (-16384 : Int) ≤ 16384 ∧ (16384 : Int) ≤ 16384
    decide
  next =>
    obtain ⟨owner,requests,rfl⟩ := Program.runs_map ran
    obtain ⟨words,issued,rfl⟩ := Program.runs_mapRequests requests
    have order := (RootFrameMir.ordered_candidates mc fc b (rows.map Row.square) idx root.key root.sort).1
    have unique := order.nodup_iff.mpr ((EndgameOrderMir.generated_layout mc b).1.filter _)
    exact PvInterior.owner unique issued window fun t ht => children _ (List.mem_map_of_mem
      (f := fun x : SerialCalls.MoveRequest (GeneratedSquare mc b.val) × Int => (SerialRootCalls.request mc fc b d x.1,x.2)) ht)

end NeuralReversi.PvsParallel.SerialTable.WeakTrial

import NeuralReversi.PvsParallel.SerialNonPvCalls
import NeuralReversi.PvsParallel.SerialCalls.Ready

namespace NeuralReversi.PvsParallel.SerialNonPvReady

open SerialCalls (Program Request)
open SerialNonPvCalls (owner mobile prelude frame scout passRequest child)
open GeneralNonPvMir (entryAction checkedHint probeHint runPrefix)
open Bitboard.Moves (X86Config)
open Cutoffs (FlipConfig)
open PvsBoardTree (Position GeneratedSquare legalSquares pass)
open TranspositionTable (Depth)
open TranspositionTable.ProbeRetry (RawAttempt)
open TableReads (ChildLookupFrom)
open Aspiration (Window)

private theorem owner_ready (req : Move → Req) (ready : ∀ m, allowed (req m)) (alpha : Int)
    (moves : List Move) (best : Int) : Program.Ready allowed (owner req alpha moves best) := by
  induction moves generalizing best with
  | nil => exact .pure
  | cons head tail ih =>
    simp only [owner, SerialCalls.Sites.ownerWith, Bool.false_and, Bool.false_eq_true, ite_false]
    apply Program.Ready.call (ready head)
    intro word
    split
    · exact .pure
    · exact ih _

theorem mobile_ready {Move : Type} [DecidableEq Move] (req : Move → Req) (ready : ∀ m, allowed (req m))
    (priority : Move → Int) (hint : Option Move) (raw : Bool) (valid : hint.isSome = raw)
    (first : Move) (rest : List Move) (alpha beta : Int) :
    Program.Ready allowed (mobile req priority hint raw first rest alpha beta) := by
  simp only [mobile, valid, ite_true]
  split
  · apply Program.Ready.call (ready _)
    intro word
    dsimp only
    split
    · exact .pure
    · exact owner_ready req ready _ _ _
  · exact owner_ready req ready _ _ _

theorem mobile_of_rest {d : Nat} {stable moves other wipes parent enhanced : Bool}
    (selected : GeneralNonPvMir.dispatch d stable moves other wipes parent enhanced = .rest) :
    moves = true := by
  unfold GeneralNonPvMir.dispatch at selected
  repeat' split at selected
  all_goals first | contradiction | simp_all

/-- A full deep NonPV frame can accept every word returned by an issued child.
    All recursive edges decrease board rank and preserve a valid scout window. -/
theorem frame_ready (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : Depth)
    (alpha beta : ScoreRefresh.Score) (paired checked cut : Bool)
    (mem : MoveList.Initialization.Memory Move.Move) (attempts : List RawAttempt)
    (reads : List (ChildLookupFrom mc fc b (·.publication history))) (priority : GeneratedSquare mc b.val → Int)
    (full : ExactDepth.Full b depth) (deep : 12 < depth.toNat)
    (window : Window alpha.raw.toInt beta.raw.toInt) (null : beta.raw.toInt = alpha.raw.toInt+1)
    (observed : TableReads.ProbeReads history attempts) (hints : TableReads.LegalHints history) :
    Program.Ready (fun r => ExactDepth.RecursiveCall mc fc b depth r.board r.depth ∧ Window r.alpha r.beta ∧
      (r.pv = false → r.beta = r.alpha+1))
      (frame mc fc b depth alpha beta paired checked cut mem attempts reads priority) := by
  unfold frame
  split
  next entered =>
    cases moves : legalSquares mc b.val with
    | nil =>
      have available := mobile_of_rest entered
      simp [PvEntryMir.generated_count, moves] at available
    | cons first rest =>
      apply mobile_ready _ _ _ _ _ _ first rest
      · intro m
        exact ⟨.ply b m depth, PvsMoveLoop.scoutWindow window,
          fun _ => by dsimp [scout, SerialCalls.Sites.scoutRequest]; omega⟩
      · have hint := congrArg Option.isSome (GeneralNonPvMir.observedHint mc b observed hints)
        simpa only [Option.isSome_map] using hint
  next other =>
    unfold prelude
    split
    next passed =>
      have moves := SerialNonPvCalls.forced_of_action passed
      rw [PvEntryMir.generated_count] at moves
      have stuck : legalSquares mc b.val = [] := by simpa using moves.1
      apply Program.Ready.call _ (fun _ => .pure)
      exact ⟨.forcedPass b depth ⟨PvsBoardTree.stuck_of_nil stuck, moves.2⟩, PvsMoveLoop.reflectedWindow window,
        fun _ => by dsimp; omega⟩
    next noPass =>
      let reply := fun (_ : Request) => some (0 : Int)
      obtain ⟨value, result⟩ := SerialNonPv.prefix_returns mc fc b depth alpha beta paired checked cut mem
        attempts reads (child reply) full deep other (fun _ _ => ⟨0, rfl⟩)
      have known := (SerialNonPvCalls.eval_prelude mc fc b depth alpha beta paired checked cut mem attempts
        reads reply).trans result
      simp only [prelude, ite_eq_right noPass] at known
      cases actual : runPrefix mc fc b depth alpha beta paired checked cut mem attempts reads (fun _ _ _ _ _ => none) with
      | some _ => exact .pure
      | none => rw [actual] at known; cases known

end NeuralReversi.PvsParallel.SerialNonPvReady

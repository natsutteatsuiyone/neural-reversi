module
public import NeuralReversi.PvsParallel.SerialOwnerCalls
public import NeuralReversi.PvsParallel.SerialPv
public import NeuralReversi.PvsParallel.EndgameOrderMir

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialPvCalls

open SerialCalls (Request MoveRequest Program)
open PvsBoardTree (Position GeneratedSquare build play pass legalSquares)
open Bitboard.Moves (X86Config)
open Aspiration (Window)

def moveRequest (b : Position) (mc : X86Config) (fc : Cutoffs.FlipConfig) (d : ExactDepth.Depth) (cut : Bool)
    (q : MoveRequest (GeneratedSquare mc b.val)) : Request :=
  ⟨(play mc fc b q.move).val, d - 1, q.pv, !q.pv && !cut, q.alpha, q.beta⟩

theorem request_scout (b : Position) (mc : X86Config) (fc : Cutoffs.FlipConfig) (d : ExactDepth.Depth)
    (cut : Bool) (m : GeneratedSquare mc b.val) (alpha : Int) :
    moveRequest b mc fc d cut ⟨m, false, -(alpha+1), -alpha⟩ =
      SerialCalls.Sites.scoutRequest (play mc fc b m).val d cut alpha := rfl

theorem request_wide (b : Position) (mc : X86Config) (fc : Cutoffs.FlipConfig) (d : ExactDepth.Depth)
    (cut : Bool) (m : GeneratedSquare mc b.val) (alpha beta : Int) :
    moveRequest b mc fc d cut ⟨m, true, -beta, -alpha⟩ =
      SerialCalls.Sites.wideRequest (play mc fc b m).val d alpha beta := rfl

def action (mc fc) (b : Position) (d : ExactDepth.Depth) (paired : Bool) (mem) :=
  let g := PvEntryMir.generated mc fc b paired mem
  PvEntryMir.dispatch d.toNat (g.len.getD 0 != 0) (Board.hasLegalMoves mc (pass b).val) g.wipeout.join.isSome

def frame (mc : X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (d : ExactDepth.Depth) (paired checked cut : Bool) (mem : MoveList.Initialization.Memory Move.Move) (a z : Int)
    (moves : List (GeneratedSquare mc b.val)) : Program Request Int :=
  match action mc fc b d paired mem with
  | .continueSearch =>
    ((SerialOwnerCalls.run Subtype.val z moves (RootReplay.start a)).mapRequests
      (moveRequest b mc fc d cut)).map (fun result => result.2.state.best)
  | .pass => SerialCalls.Sites.pass (pass b) d true cut a z
  | _ => .ofOption (PvEntryMir.run mc fc b d paired checked mem a z (fun _ _ _ _ => none))

/-- The one-worker frame agrees with the scheduled MIR on both returns and missing replies.
    The handoff continuation may be arbitrary because this schedule never issues it. -/
theorem eval_frame (mc : X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (d : ExactDepth.Depth)
    (paired checked cut : Bool) (mem : MoveList.Initialization.Memory Move.Move) (a z : Int)
    (moves : List (GeneratedSquare mc b.val)) (respond : Request → Option Int)
    (split : RootOwnerMir.Slice.Exit (GeneratedSquare mc b.val) → Option Int) :
    (frame mc fc b d paired checked cut mem a z moves).eval respond =
      PvSerialMir.scheduled mc fc b d paired checked mem a z
        (fun next cd lo hi => respond ⟨next, cd, true, false, lo, hi⟩) moves
        (fun m p lo hi => respond (moveRequest b mc fc d cut ⟨m,p,lo,hi⟩)) RootOwnerMir.Serial.schedule split := by
  let replies := fun m p lo hi => respond (moveRequest b mc fc d cut ⟨m,p,lo,hi⟩)
  have leaf (v : Option Int) : (Program.ofOption v).eval respond = v := by cases v <;> rfl
  unfold frame
  cases which : action mc fc b d paired mem with
  | zero | terminal | wipeout =>
    dsimp only [action] at which
    simp only [PvSerialMir.scheduled, PvEntryMir.run, which, leaf]
  | pass =>
    dsimp only [action] at which
    simp only [PvSerialMir.scheduled, PvEntryMir.run, which]
    rfl
  | continueSearch =>
    dsimp only [action] at which
    simp only [PvSerialMir.scheduled, which, Program.eval_map, Program.eval_mapRequests]
    change ((SerialOwnerCalls.run Subtype.val z moves (RootReplay.start a)).eval
      (SerialOwnerCalls.reply replies)).map _ = _
    have agree := SerialOwnerCalls.eval_handoff Subtype.val replies d.toNat rootMinimumSplitDepth a z moves
    cases issued : (SerialOwnerCalls.run Subtype.val z moves (RootReplay.start a)).eval
      (SerialOwnerCalls.reply replies) <;>
      cases source : RootOwnerMir.Slice.run Subtype.val replies RootOwnerMir.Serial.schedule d.toNat
        rootMinimumSplitDepth z 0 0 moves.length moves (RootReplay.start a) <;>
      simp only [issued, source, Option.map_none, Option.map_some, Option.some.injEq, Prod.mk.injEq,
        reduceCtorEq] at agree <;> try contradiction
    · change none = (none : Option (RootOwnerMir.Slice.Exit _)).bind _
      rfl
    · obtain ⟨snap, joined⟩ := agree
      simp [← joined, snap]

private theorem skipOwner (mc fc) (b : Position) (d : ExactDepth.Depth) (paired checked : Bool) (mem) (a z : Int)
    (depth : ExactDepth.Full b d) (v : Int)
    (ran : PvEntryMir.run mc fc b d paired checked mem a z (fun _ _ _ _ => none) = some v) (n : Nat) :
    boundedReplay (n+1) (build mc fc b) true a z v :=
  PvEntryMir.exitExecution mc fc b d paired checked mem a z _ depth (by simp) ran

private theorem pass_replay (mc fc) (b : Position) (d : ExactDepth.Depth) (paired checked : Bool) (mem)
    (a z word : Int) (full : ExactDepth.Full b d)
    (which : action mc fc b d paired mem = .pass)
    (child : boundedReplay n (build mc fc (pass b)) true (-z) (-a) word) :
    boundedReplay (n+1) (build mc fc b) true a z (-word) := by
  apply PvEntryMir.exitExecution mc fc b d paired checked mem a z (fun _ _ _ _ => some word) full
  · intro _ _ v found
    exact Option.some.inj found ▸ child
  · change (match action mc fc b d paired mem with
      | .zero => _ | .terminal => _ | .pass => _ | .wipeout => _ | .continueSearch => _) = _
    rw [which]
    rfl

/-- A normal PV return certifies its own score from the children it actually
    resumed. The owner carries each child's word into the following instruction. -/
theorem frame_replay {mc : X86Config} {fc : Cutoffs.FlipConfig} {b : Position} {d : ExactDepth.Depth}
    {mem : MoveList.Initialization.Memory Move.Move} {moves : List (GeneratedSquare mc b.val)}
    (full : ExactDepth.Full b d) (order : moves.Perm (legalSquares mc b.val)) (w : Window a z)
    (ran : Program.Runs (frame mc fc b d paired checked cut mem a z moves) children v)
    (child : ∀ item ∈ children, Window item.1.alpha item.1.beta →
      (item.1.pv = false → item.1.beta = item.1.alpha+1) →
      boundedReplay n (build mc fc item.1.board) item.1.pv item.1.alpha item.1.beta item.2) :
    boundedReplay (n+1) (build mc fc b) true a z v := by
  unfold frame at ran
  cases which : action mc fc b d paired mem <;> rw [which] at ran
  case zero | terminal | wipeout =>
    exact skipOwner mc fc b d paired checked mem a z full v (Program.runs_ofOption ran).1 n
  case pass =>
    cases ran with
    | call more =>
      cases more
      apply pass_replay mc fc b d paired checked mem a z _ full which
      exact child _ List.mem_cons_self (PvsMoveLoop.reflectedWindow w) (by simp)
  case continueSearch =>
    obtain ⟨out, mapped, rfl⟩ := Program.runs_map ran
    obtain ⟨calls, owner, rfl⟩ := Program.runs_mapRequests mapped
    have unique : moves.Nodup := order.nodup_iff.mpr (EndgameOrderMir.generated_layout mc b).1
    let key := fun (q : MoveRequest (GeneratedSquare mc b.val)) => (q.move, q.pv)
    let search : PvsMoveLoop.ChildSearch (GeneratedSquare mc b.val) :=
      fun m role _ _ => Program.returnAt key calls (m, role)
    let done : PvsMoveLoop.ChildRelation (GeneratedSquare mc b.val) :=
      fun m p lo hi x => boundedReplay n (build mc fc (play mc fc b m).val) p lo hi x
    have completed (t : MoveRequest (GeneratedSquare mc b.val) × Int) (seen : t ∈ calls) :
        done t.1.move t.1.pv t.1.alpha t.1.beta t.2 := by
      obtain ⟨window, narrow⟩ := SerialOwnerCalls.runs_windows owner w t seen
      let q := moveRequest b mc fc d cut t.1
      have present := List.mem_map_of_mem (f := fun item : MoveRequest (GeneratedSquare mc b.val) × Int =>
        (moveRequest b mc fc d cut item.1, item.2)) seen
      exact child (q,t.2) present window narrow
    have supplied (t) (seen : t ∈ calls) : search t.1.move t.1.pv t.1.alpha t.1.beta = t.2 :=
      Program.returnAt_mem key calls (SerialOwnerCalls.runs_distinct owner unique) t seen
    obtain ⟨same, scan⟩ := SerialOwnerCalls.runs_serial owner search done (fun t h => ⟨supplied t h, completed t h⟩)
    rw [same, RootReplay.serial_best]
    change Execution (boundedReplay n) (build mc fc b) true a z _
    rw [build.eq_def]
    split
    next first tail h =>
      exact .sequential moves (fun m => build mc fc (play mc fc b m).val) ((h ▸ order).map _) search scan
    next absent =>
      cases mobile : Board.hasLegalMoves mc (pass b).val <;>
        by_cases zero : d.toNat = 0 <;>
        simp [action, PvEntryMir.dispatch, PvEntryMir.generated_count, absent, mobile, zero] at which

end NeuralReversi.PvsParallel.SerialPvCalls

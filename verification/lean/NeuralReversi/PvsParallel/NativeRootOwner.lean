module
public import NeuralReversi.PvsParallel.NativeSplitFrame
public import NeuralReversi.PvsParallel.NativeFrames.OwnerSites
public import NeuralReversi.PvsParallel.RootStorageMir.Rows
public import NeuralReversi.PvsParallel.FrameTableReads.Execution

@[expose] public section

namespace NeuralReversi.PvsParallel.NativeRootOwner

open TranspositionTable
open NativeFrames (Row Capture logOf)
open FrameReturns (RootOwner rootOrder)
open NativeWorkerFeed (labels)

structure Input (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (r : Row) (cap : Capture pool) (identify before) (history : Threading.CutoffVisibility.History) where
  paired : Bool
  memory : MoveList.Initialization.Memory Move.Move
  mode : Bitboard.Square → EndgameOrderMir.Heuristic
  tie : Move.Move → Fin 64
  probe : Option (FrameTableReads.Operations.Execution.ProbeCapture (Cutoffs.key r.frame.position))
  schedule : Nat → Bool × Bool
  split : Option (NativeSplit.Packet mc fc r.frame true r.frame.beta cap identify before history)

def Input.owner {cap : Capture pool} (p : Input mc fc r cap identify before history)
    (probe : FrameTableReads.Operations.Execution.ProbeCapture (Cutoffs.key r.frame.position)) (seq)
    (squares : List Bitboard.Square) (idx : Nat) :
    Option (RootOwner (logOf mc fc cap.rows cap.table identify before seq) r.frame r.depth squares idx
      r.frame.alpha r.frame.beta) := do
  let log := logOf mc fc cap.rows cap.table identify before seq
  let f := r.frame
  let order := rootOrder mc fc f.position squares idx p.mode p.tie (ReaderMir.bestMove probe.result)
  let receipt ← (RootOwnerMir.fromMoves Subtype.val (log.pvReplies f r.depth) p.schedule r.depth.toNat
    rootMinimumSplitDepth f.alpha f.beta order).attach
  return ⟨p.mode,p.tie,ReaderMir.bestMove probe.result,p.schedule,receipt.val,receipt.property⟩

/-- A physical return supplies the root receipt, including the exact order and
    split cursor that the workers receive after any sibling postponements. -/
theorem Input.owner_of_slice {cap : Capture pool} (p : Input mc fc r cap identify before history)
    (probe : FrameTableReads.Operations.Execution.ProbeCapture (Cutoffs.key r.frame.position)) (seq)
    (squares : List Bitboard.Square) (idx : Nat)
    (ran : let f := r.frame
      let order := rootOrder mc fc f.position squares idx p.mode p.tie (ReaderMir.bestMove probe.result)
      RootOwnerMir.Slice.run Subtype.val ((logOf mc fc cap.rows cap.table identify before seq).pvReplies f r.depth)
        p.schedule r.depth.toNat rootMinimumSplitDepth f.beta 0 0 order.length order (RootReplay.start f.alpha) =
          some physical) :
    ∃ owner, p.owner probe seq squares idx = some owner ∧ RootOwnerMir.Slice.layout [] owner.receipt = physical := by
  dsimp only at ran
  rw [RootOwnerMir.Slice.fromMoves_eq] at ran
  obtain ⟨receipt,completed,rfl⟩ := Option.map_eq_some_iff.mp ran
  unfold Input.owner
  dsimp only
  rw [Option.attach_eq_some _ ⟨receipt,completed⟩]
  exact ⟨_,rfl,rfl⟩

/-- Recover the root result from individual child-frame returns. The source may
    return different words at different invocations, regardless of their windows. -/
theorem Input.owner_of_calls {cap : Capture pool} (p : Input mc fc r cap identify before history)
    (probe : FrameTableReads.Operations.Execution.ProbeCapture (Cutoffs.key r.frame.position)) (seq)
    (squares : List Bitboard.Square) (idx : Nat)
    (path : let order := rootOrder mc fc r.frame.position squares idx p.mode p.tie (ReaderMir.bestMove probe.result)
      RootOwnerMir.Slice.Path Subtype.val source p.schedule r.depth.toNat rootMinimumSplitDepth r.frame.beta
        0 [] order order.length (RootReplay.start r.frame.alpha) visits physical)
    (captured : NativeFrames.decode pool inputs stream = some cap)
    (issued : NativeFrames.OwnerSites.Execution pool inputs stream r.frame (NativeFrames.OwnerSites.moveSite false)
      (FrameReturns.OwnerCalls.emitted fc r.frame r.depth visits)) :
    ∃ owner, p.owner probe seq squares idx = some owner ∧ RootOwnerMir.Slice.layout [] owner.receipt = physical := by
  apply p.owner_of_slice
  have order := (EndgameOrderMir.root_perm mc fc r.frame.position squares idx p.mode p.tie (ReaderMir.bestMove probe.result)).1
  have unique := order.nodup_iff.mpr ((EndgameOrderMir.generated_layout mc r.frame.position).1.filter _)
  have returns := issued.observed mc fc captured identify before seq
  exact path.replays _ (FrameReturns.OwnerCalls.supplies _ _ _ _ (path.distinct unique).1 returns)

/-- Construct the split using the owner's two returned fields and the workers' own
    receipts. The branch used to replay root-vector writes retains their mutex timestamps. -/
noncomputable def joined {f : FrameReturns.Frame} {cap : Capture pool}
    (p : NativeSplit.Packet mc fc f true f.beta cap identify before history) (seq)
    (owner : RootOwner (logOf mc fc cap.rows cap.table identify before seq)
      f depth (initial.map RootMoveStorage.Row.square) idx f.alpha f.beta) (gave : owner.receipt.handoff = true) :
    Option (RootStorageMir.Handoff (logOf mc fc cap.rows cap.table identify before seq) f depth initial idx) := do
  match frozen : p.order with
  | [] => none
  | first :: rest =>
    if agrees : first :: rest = owner.receipt.visited ++ owner.receipt.remaining.pending ∧
        p.tickets.start.toNat = owner.receipt.remaining.idx ∧
        p.tickets.initial.state.alpha = owner.receipt.snapshot.state.alpha ∧
        p.tickets.initial.state.best = owner.receipt.snapshot.state.best then
      let h := NativeSplit.handoff p.owner p.workers p.loads f.alpha (NativeSplit.moveCount _ _ p.frozen)
      have attended := h.calls p.loads cap.inputs cap p.captured p.unique p.frozen
        (fun part => p.frames (h.reports.symm.trans part)) identify before seq _ p.causal p.before_parent
      have entry : p.tickets.initial.state = owner.receipt.snapshot.state := by
        cases x : p.tickets.initial.state
        cases y : owner.receipt.snapshot.state
        simpa [x,y] using agrees.2.2
      return ⟨owner,first,rest,p.owner.score,labels mc fc f.position p.order,by rw [frozen],
        h.split,gave,agrees.1,h.start.trans agrees.2.1,h.entry.trans entry,attended⟩
    else none

noncomputable def Input.branch {cap : Capture pool} (p : Input mc fc r cap identify before history) (seq)
    (rows : List RootMoveStorage.Row) (idx : Nat) :
    Option (RootStorageMir.Branch (logOf mc fc cap.rows cap.table identify before seq) r.frame r.depth rows idx) := do
  let f := r.frame
  let squares := rows.map RootMoveStorage.Row.square
  let filled := MoveList.Generation.fill false p.paired fc f.position.val
    (Root.remainingMask mc f.position.val squares idx) p.memory
  match cached : filled.wipeout with
  | none => none
  | some (some sq) => return .wipe ⟨p.paired,p.memory,sq,cached⟩
  | some none =>
    let probe ← p.probe
    let owner ← p.owner probe seq squares idx
    match gave : owner.receipt.handoff with
    | false => return .serial ⟨owner,gave⟩
    | true =>
      let packet ← p.split
      return .split (← joined packet seq owner gave)

end NeuralReversi.PvsParallel.NativeRootOwner

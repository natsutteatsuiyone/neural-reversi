import NeuralReversi.PvsParallel.SlotFrameMir.RawHandoff
import NeuralReversi.PvsParallel.WorkerPollMir.Source
import NeuralReversi.PvsParallel.IteratorFrames.EntryRegisters
import NeuralReversi.PvsParallel.NativeFrames.SplitReturn

namespace NeuralReversi.PvsParallel.NativeSplit

open Threading.SplitWindow (State)
open PvsBoardTree (Position GeneratedSquare build play)
open NativeWorkerFeed (labels)
open FrameReturns (Frame Children ScoreStep Samples)

/-- The split and parent see the same decoded native frames and table writes. A packet
    records one invocation of Thread::split, which returns directly to its calling move loop. -/
structure Packet (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (f : Frame)
    (pv : Bool) (beta : Int) {pool : EndgameSearch.CacheHistory.Pool} (cap : NativeFrames.Capture pool)
    (identify : Threading.AtomicWord.Event → Nat) (before : Nat → Nat → Prop) (history : Threading.CutoffVisibility.History) where
  rc : List Move.Move
  tickets : Tickets rc pv beta
  mask : List Threading.AtomicWord.Write
  generation : SlotFrameMir.RawMask mask
  time : Threading.AtomicWord.Event → Nat
  slot : SlotFrameMir.Slot
  physical : SlotFrameMir.RawProjection.PhysicalOwner generation history time slot tickets
  roster : tickets.workers.Perm (SlotFrameMir.RawProjection.roster generation)
  receipts : List (WorkerPollMir.Receipt tickets.width)
  iterator : WorkerPollMir.Trace rc.length tickets.start tickets.state receipts
  workerPath : ∀ i ∈ tickets.workers, (tickets.result i).completed.departure = .exhausted →
    ∃ marks, WorkerPollMir.Native (IteratorFrames.source tickets.journal tickets.stream) 0 .poll marks ∧
      WorkerPollMir.polls marks = receipts.filter (fun r => r.worker == i.val)
  stopped : let calls := IteratorFields.chronological tickets.state.writes tickets.journal tickets.stream tickets.workers
    let stamp := physical.exit.clock.atStep (3 + physical.exit.log.length)
    ∀ c ∈ calls, c.ticket.locked.stopped = true → Nonempty (SplitWorkerMir.Stop.Poll physical.check stamp calls) ∨
      (∀ obs, physical.check.shared = some obs → Threading.SplitReturn.Observation.AbortCause history obs stamp)
  reports : SlotFrameMir.RawProjection.Reports physical
  order : List (GeneratedSquare mc f.position.val)
  frozen : NativeWorkerFeed.frozen mc fc f.position rc order = true
  samples : List NativeFrames.EntryStream.Sample
  source : samples.map NativeFrames.EntryStream.Sample.record = tickets.stream
  capture : NativeFrames.EntryStream.capture pool samples = some cap
  unique : IteratorFrames.unique tickets.state.writes tickets.journal = true
  entries : ∀ e ∈ IteratorFrames.entries f.position.val rc tickets.journal tickets.stream tickets.state.writes,
    e ∈ samples.filterMap NativeFrames.EntryStream.Sample.entry
  causal : ∀ a z, Threading.CutoffVisibility.HappensBefore history a z → before (identify a) (identify z)
  ownerWorker : slot.1.val = f.worker
  resumed : NativeFrames.resumed (SplitSchedulerMir.OwnerFields.returned physical.exit.timeline physical.exit.log.length).step
    f.scoreStep f.score (NativeFrames.Returns.localEvents f.worker tickets.stream)

namespace Packet

variable (p : Packet mc fc f pv beta cap identify before history)

noncomputable abbrev owner := p.physical.owner

theorem captured : NativeFrames.decode _ cap.inputs p.tickets.stream = some cap := by
  rw [←p.source,(NativeFrames.source_of_decode p.capture).1]
  exact p.capture

theorem same_stream (source : NativeFrames.decode _ inputs stream = some cap) :
    p.tickets.stream = stream :=
  (NativeFrames.source_of_decode p.captured).2.symm.trans (NativeFrames.source_of_decode source).2

theorem exhausted (i) (member : i ∈ p.tickets.workers) (ended : (p.tickets.result i).completed.departure = .exhausted) :
    Nonempty (SplitWorkerMir.Iterator.Drained p.rc.length p.tickets.start p.tickets.state i.val) := by
  obtain ⟨marks, ran, reads⟩ := p.workerPath i member ended
  exact WorkerPollMir.exhausted_ticket p.iterator i member ran reads ended

abbrev workers := (SlotFrameMir.RawProjection.team_of_roster p.physical p.roster p.exhausted p.stopped).departures
noncomputable abbrev loads := p.reports.alpha

theorem frames {earlier r later} (part : p.tickets.reports = earlier ++ r :: later) :
    let x := p.tickets.post part
    ∀ q, SplitWorkerMir.Step.returned q ∈ x.body.steps ++ SplitWorkerMir.lockedSteps →
      NativeWorkerFeed.stepOk mc fc f.position p.order p.rc x.op.old cap.rows x.returned x.lock.prefix (.returned q) = true :=
  (p.tickets.post part).child_step p.captured p.frozen ((p.tickets.post part).operands _
    (fun e member => NativeFrames.EntryStream.at_entry p.samples p.capture e (p.entries e member)))

theorem before_parent : before (identify p.owner.load.event) (identify f.use) := by
  have scored := NativeFrames.resume_order p.captured f.worker p.resumed
  have loaded := SplitSchedulerMir.OwnerFields.read_before_return p.physical.exit.timeline p.physical.exit.log p.physical.source
  apply p.causal _ _ (.sequenced ⟨p.physical.thread.trans p.ownerWorker, ?_⟩)
  simpa only [owner,Owner.load,SlotFrameMir.RawProjection.PhysicalOwner.owner,SplitSchedulerMir.scoreLoad,List.length_map,Frame.use]
    using Nat.lt_trans loaded.2 scored

def answer (moves : List (GeneratedSquare mc f.position.val)) (idx : Nat) (entry : State) : Option Int :=
  if p.order = moves ∧ p.tickets.start.toNat = idx ∧ p.tickets.initial.state.alpha = entry.alpha ∧
      p.tickets.initial.state.best = entry.best then
    some (Threading.AtomicWord.signed32At _ p.physical.source) else none

theorem division (alpha : Int) (moves : List (GeneratedSquare mc f.position.val)) (idx : Nat) (entry : State)
    (ok : p.answer moves idx entry = some v) (sequenced) :
    let log := NativeFrames.logOf mc fc cap.rows cap.table identify before sequenced
    ∃ split : SplitExecution (labels mc fc f.position moves).length pv alpha beta v,
      split.start.toNat = idx ∧ split.entry = entry ∧ ReportCalls (Children log f) (labels mc fc f.position moves) split := by
  unfold answer at ok
  split at ok <;> try contradiction
  rename_i accepted
  obtain ⟨rfl,rfl,lo,best⟩ := accepted
  cases ok
  let h := handoff p.owner p.workers p.loads alpha (moveCount _ _ p.frozen)
  have state : p.tickets.initial.state = entry := by
    cases entry
    dsimp only at lo best
    rw [← lo,← best]
  exact ⟨h.split,h.start,h.entry.trans state,h.calls p.loads cap.inputs cap p.captured p.unique p.frozen
    (fun part => p.frames (h.reports.symm.trans part)) identify before sequenced _ p.causal p.before_parent⟩

end Packet

def pvReply {f : Frame} (p : Option (Packet mc fc f true beta cap identify before history))
    (r : RootOwnerMir.Slice.Exit (GeneratedSquare mc f.position.val)) : Option Int :=
  p.bind fun packet => packet.answer r.moves r.idx r.snapshot.state

def scoutReply {f : Frame} (p : Option (Packet mc fc f false beta cap identify before history)) :
    GeneralOwnerMir.SplitReturn (GeneratedSquare mc f.position.val) := fun order idx r => do
  let packet ← p
  match r.outcome with
  | .handoff s => packet.answer (order.take idx ++ r.visited ++ r.remaining.pending) r.remaining.idx s
  | .returned _ => none

theorem pv {f : Frame} {a z : ScoreRefresh.Score} {cap : NativeFrames.Capture pool}
    (packet : Option (Packet mc fc f true z.raw.toInt cap identify before history)) (sequenced)
    (paired checked : Bool) (mem : MoveList.Initialization.Memory Move.Move)
    (mode : Bitboard.Square → EndgameOrderMir.Heuristic) (tie : GeneratedSquare mc f.position.val → Fin 64)
    (hint : Option Bitboard.Square) (schedule : Nat → Bool × Bool)
    (tag : f.pv = true)
    (normal : let log := NativeFrames.logOf mc fc cap.rows cap.table identify before sequenced
      PvSerialMir.scheduled mc fc f.position depth paired checked mem a.raw.toInt z.raw.toInt
        (fun b d => log.reply f.id f.use b d true false)
        (EndgameOrderMir.pvOrder mc fc f.position mode tie hint)
        (log.pvReplies f depth) schedule (pvReply packet) = some f.score) :
    ScoreStep (NativeFrames.logOf mc fc cap.rows cap.table identify before sequenced) endian f depth a z := by
  apply ScoreStep.pv paired checked mem mode tie hint schedule (pvReply packet) tag _ normal
  intro r v _ got
  obtain ⟨p,_,answer⟩ := Option.bind_eq_some_iff.mp got
  exact p.division a.raw.toInt _ _ _ answer sequenced

theorem nonPv {f : Frame} {a z : ScoreRefresh.Score} {cap : NativeFrames.Capture pool}
    (packet : Option (Packet mc fc f false z.raw.toInt cap identify before history)) (sequenced)
    (paired checked : Bool) (mem : MoveList.Initialization.Memory Move.Move)
    (attempts : List TranspositionTable.ProbeRetry.RawAttempt)
    (reads : List (TableReads.ChildLookupFrom mc fc f.position
      (Samples (NativeFrames.logOf mc fc cap.rows cap.table identify before sequenced) endian f)))
    (mode : Bitboard.Square → EndgameOrderMir.Heuristic) (tie : GeneratedSquare mc f.position.val → Fin 64)
    (schedule : GeneralOwnerMir.Schedule (GeneratedSquare mc f.position.val)) (tag : f.pv = false)
    (sampled : TableReads.ProbeReadsFrom (Samples (NativeFrames.logOf mc fc cap.rows cap.table identify before sequenced)
      endian f) attempts)
    (normal : let log := NativeFrames.logOf mc fc cap.rows cap.table identify before sequenced
      GeneralOwnerMir.scheduled mc fc f.position depth a z paired checked f.cut mem attempts reads
        (EndgameOrderMir.framePriority mc fc f.position mode tie) (log.nonPv f.id f.use) schedule (scoutReply packet) =
          some f.score) :
    ScoreStep (NativeFrames.logOf mc fc cap.rows cap.table identify before sequenced) endian f depth a z := by
  apply ScoreStep.nonPv paired checked mem attempts reads mode tie schedule (scoutReply packet) tag sampled _ normal
  intro order idx r state v split got
  obtain ⟨p,_,answer⟩ := Option.bind_eq_some_iff.mp got
  simp only [split] at answer
  unfold GeneralOwnerMir.Completed
  rw [split]
  exact p.division a.raw.toInt _ _ _ answer sequenced

end NeuralReversi.PvsParallel.NativeSplit

import NeuralReversi.PvsParallel.IteratorCutoff
import NeuralReversi.PvsParallel.IteratorReports
import NeuralReversi.PvsParallel.SplitWorkerMir.OwnerFields
import NeuralReversi.PvsParallel.SplitWorkerMir.Stop
import NeuralReversi.PvsParallel.SplitSchedulerMir.WorkerFields

namespace NeuralReversi.PvsParallel.NativeSplit

open Threading
open AtomicWord (Write Read SequencedBefore signed32At)
open CutoffVisibility (History)
open SplitCompletion (Activation Retirement Worker)
open SplitResult (Report)
open SplitWorkerMir
open SplitSchedulerMir (Clock Guard Returned OwnerMemory scoreLoad)
open TranspositionTable.Packing (encodeBestMove)

structure Tickets (rc : List Move.Move) (pv : Bool) (beta : Int) where
  width : Nat
  start : BitVec width
  state : MoveList.ConcurrentTrace.State width
  run : MoveList.ConcurrentTrace.Execution rc.length start state
  fits : rc.length < 2 ^ width
  offset : start.toNat ≤ rc.length
  journal : List IteratorFrames.Observation
  stream : List NativeFrames.Record
  workers : List Worker
  distinct : (workers.map Fin.val).Nodup
  covers : ∀ op ∈ state.writes, op.worker ∈ workers.map Fin.val
  dispatches : List SplitSchedulerMir.WorkerFields.Dispatch
  traced : state.writes.all (SplitSchedulerMir.WorkerFields.matchesTicket dispatches journal stream)
  initial : FieldStores.Fields
  a : Write
  b : Write
  m : Write
  seedAlpha : FieldStores.low32 a = initial.alpha
  seedBest : FieldStores.low32 b = initial.best
  seedMove : FieldStores.low8 m = encodeBestMove initial.picked
  final : FieldStores.Log
  replay : IteratorFields.replay rc state.writes journal stream workers
    (SplitSchedulerMir.WorkerFields.traffic dispatches journal) pv beta encodeBestMove
    (FieldStores.initialLog initial a b m) = some final

namespace Tickets

variable (t : Tickets rc pv beta)

abbrev traffic := SplitSchedulerMir.WorkerFields.traffic t.dispatches t.journal

def result := IteratorFields.results t.state.writes t.journal t.stream
def reports := SplitWorkerMir.Iterator.lockReports t.result t.workers
def calls := (IteratorFields.chronological t.state.writes t.journal t.stream t.workers).map (IteratorFields.raw t.traffic)

theorem ran (i : Worker) (here : i ∈ t.workers) :
    SplitWorkerMir.Iterator.worker (IteratorFrames.source t.journal t.stream) t.state.writes i.val = some (t.result i) :=
  IteratorFields.worker_returns t.replay i here

def post (part : t.reports = earlier ++ r :: later) :
    IteratorFrames.Posted rc t.state t.journal t.stream earlier r (IteratorFrames.ranked t.workers t.result) :=
  IteratorFrames.recover (MoveList.ConcurrentTrace.consecutive _ t.run.replayed t.fits).2
    (IteratorFrames.posted rc t.run t.fits t.workers t.result t.ran part)

def interval (part : t.reports = earlier ++ r :: later) :=
  IteratorReports.interval rc t.workers t.result t.ran t.seedAlpha t.seedBest t.seedMove t.replay (t.post part) part

theorem valid (r : Report) (member : r ∈ t.reports) : r.index ∈ SplitResult.slots t.start.toNat rc.length := by
  obtain ⟨pre,post,part⟩ := List.append_of_mem member
  obtain ⟨x⟩ := IteratorFrames.posted rc t.run t.fits t.workers t.result t.ran part
  rw [x.index,SplitResult.mem_slots]
  have range := (MoveList.ConcurrentTrace.consecutive _ t.run.replayed t.fits).1
  have inRange := List.mem_map_of_mem (f := fun (op : MoveList.ConcurrentTrace.Write t.width) => op.old.toNat) x.issued
  rw [range] at inRange
  obtain ⟨_,_,same⟩ := List.mem_range'.mp inRange
  exact ⟨by omega,(MoveList.Concurrent.advance_some _ (MoveList.ConcurrentTrace.issued _ t.run.replayed x.issued)).1⟩

end Tickets

/-- Physical word histories include earlier and later uses of this slot. The recorded
    sources of the owner's loads may point anywhere in those histories. -/
structure Owner (t : Tickets rc pv beta) where
  history : History
  mask : List Write
  activation : Activation mask
  clock : Clock
  guards : List (Guard mask)
  returned : Returned SplitSchedulerMir.idle mask 0 guards
  live : Guard.shutdown true ∉ guards
  bPre : List Write
  bPost : List Write
  mPre : List Write
  mPost : List Write
  lastB : Fin (bPre ++ t.final.best ++ bPost).length
  source : Fin (bPre ++ t.final.best ++ bPost).length
  lastM : Fin (mPre ++ t.final.moves ++ mPost).length
  move : Read (mPre ++ t.final.moves ++ mPost)
  endB : lastB.val + 1 = bPre.length + t.final.best.length
  endM : lastM.val + 1 = mPre.length + t.final.moves.length
  memory : OwnerMemory history mask _ activation clock guards source
  seedScore : t.b.event = clock.atStep 2
  seedMove : SequencedBefore t.m.event (clock.atStep 3)
  afterScore : SequencedBefore (scoreLoad _ source clock guards).event move.event
  release : ∀ call ∈ t.calls, ∀ op ∈ call.ops, ∀ w, FieldStores.BestOrMove op w →
    Nonempty (FieldStores.WorkerRelease activation call)
  nextScore : ∀ i : Fin (bPre ++ t.final.best ++ bPost).length, i.val = lastB.val + 1 →
    SequencedBefore (scoreLoad _ source clock guards).event ((bPre ++ t.final.best ++ bPost).get i).event
  moveMemory : FieldStores.CoherentRead history _ move lastM
  check : SplitReturn.Observation.OwnerCheck history
  running : check.stopped = false

namespace Owner

variable {t : Tickets rc pv beta} (o : Owner t)

def scores := o.bPre ++ t.final.best ++ o.bPost
def load := scoreLoad o.scores o.source o.clock o.guards
def score := signed32At o.scores o.source

theorem reads : ∃ wait : Retirement o.history o.mask o.activation,
    SplitReturn.Observation.ScoreRead o.history wait o.load o.lastB ∧
    SplitReturn.Observation.ScoreRead o.history wait o.move o.lastM ∧
    wait.read.event = o.clock.atStep (3 + o.guards.length) :=
  FieldStores.owner_reads o.activation o.history o.clock o.guards o.returned o.live t.a t.b t.m t.calls
    (IteratorFields.field_replay t.replay) o.bPre o.bPost o.mPre o.mPost o.lastB o.source o.lastM o.move
    o.endB o.endM o.memory o.seedScore o.seedMove o.afterScore o.release o.nextScore o.moveMemory

noncomputable def wait := Classical.choose o.reads

theorem wait_event : o.wait.read.event = o.clock.atStep (3 + o.guards.length) :=
  (Classical.choose_spec o.reads).2.2

end Owner

structure Departures {t : Tickets rc pv beta} (o : Owner t) : Prop where
  participant : ∀ i ∈ t.workers, i = o.activation.owner ∨ ∃ e, .book i e ∈ o.activation.actions
  released : ∀ i, SplitReturn.Released o.activation.actions i → i ∈ t.workers
  exhausted : ∀ i, SplitReturn.Released o.activation.actions i → (t.result i).completed.departure = .exhausted →
    Nonempty (SplitWorkerMir.Iterator.Drained rc.length t.start t.state i.val)
  stopped : let calls := IteratorFields.chronological t.state.writes t.journal t.stream t.workers
    ∀ c ∈ calls, c.ticket.locked.stopped = true →
      Nonempty (SplitWorkerMir.Stop.Poll o.check o.wait.read.event calls) ∨
        (∀ obs, o.check.shared = some obs → SplitReturn.Observation.AbortCause o.history obs o.wait.read.event)

structure AlphaReads {t : Tickets rc pv beta} (o : Owner t) where
  pre : List Write
  post : List Write
  finish : ∀ {earlier r later} (part : t.reports = earlier ++ r :: later), IteratorReports.Release (t.post part) o.activation
  loads : ∀ {earlier r later} (part : t.reports = earlier ++ r :: later),
    IteratorReports.Modifications (t.interval part) (t.post part) o.history pre post

noncomputable def AlphaReads.occurrence {t : Tickets rc pv beta} {o : Owner t}
    (data : AlphaReads o) (part : t.reports = earlier ++ r :: later) :=
  IteratorReports.occurrence rc t.run t.fits t.workers t.result t.covers t.ran t.seedAlpha t.seedBest t.seedMove
    t.replay (t.post part) part (data.finish part) data.pre data.post (data.loads part)

noncomputable def execution {t : Tickets rc pv beta} (original : Int) (o : Owner t)
    (workers : Departures o) (posts : AlphaReads o) : SplitExecution rc.length pv original beta o.score where
  history := o.history
  mask := o.mask
  activation := o.activation
  wait := o.wait
  check := o.check
  width := t.width
  start := t.start
  last := t.state.current
  issued := t.state.writes.map MoveList.ConcurrentTrace.Write.old
  entry := t.initial.state
  reports := t.reports
  writes := o.scores
  result := o.load
  lastScore := o.lastB
  capacity := t.fits
  startValid := t.offset
  claims := t.run.claims
  valid := t.valid
  dispatches := SplitWorkerMir.Iterator.dispatches t.run (IteratorFrames.source t.journal t.stream)
    t.workers t.result o.history o.wait.read.event o.activation.actions o.activation.owner o.check _
    t.distinct t.covers workers.participant t.ran workers.released workers.exhausted
    (fun i rel => IteratorFields.worker_cutoff t.seedAlpha t.seedBest t.seedMove t.replay i (workers.released i rel))
    (by
      intro i rel stopped
      obtain ⟨c,here,read⟩ := SplitWorkerMir.Stop.worker_stop t.replay i (workers.released i rel) stopped
      rcases workers.stopped c here read with localStop | aborted
      · obtain ⟨poll⟩ := localStop
        obtain ⟨source⟩ := poll.origin
        exact source.explains t.seedAlpha t.seedBest t.seedMove t.replay
      · exact .inr (.abort aborted))
  read := (Classical.choose_spec o.reads).1
  bestStores := ⟨_,_,_,rfl,o.endB,(IteratorFields.stores rc _ t.workers t.result t.ran t.seedAlpha t.seedBest t.seedMove t.replay).2.1⟩
  accepted := by simp [SplitReturn.checked,o.running,Owner.score,Owner.load,SplitSchedulerMir.scoreLoad]
  reportReads := fun _ _ _ part => (posts.occurrence part).receipt.reads
  reportPublications := fun _ _ _ part => (posts.occurrence part).publication

noncomputable def moveRead {t : Tickets rc pv beta} (original : Int) (o : Owner t)
    (workers : Departures o) (posts : AlphaReads o) :
    (execution original o workers posts).MoveRead encodeBestMove (IteratorFields.atIndex rc)
      t.initial.picked t.final.fields.picked where
  writes := o.mPre ++ t.final.moves ++ o.mPost
  load := o.move
  last := o.lastM
  order := (Classical.choose_spec o.reads).2.1
  afterScore := o.afterScore
  stores := ⟨_,_,_,rfl,o.endM,(IteratorFields.stores rc _ t.workers t.result t.ran t.seedAlpha t.seedBest t.seedMove t.replay).2.2⟩

end NeuralReversi.PvsParallel.NativeSplit

import NeuralReversi.PvsParallel.SlotFrameMir.RawGuards
import NeuralReversi.PvsParallel.NativeSplitCalls
import NeuralReversi.PvsParallel.SplitSchedulerMir.WorkerReturn

namespace NeuralReversi.PvsParallel.SlotFrameMir.RawProjection

open Threading
open AtomicWord (Event Write Read SequencedBefore InReleaseSequence)
open CutoffVisibility (History HappensBefore)
open SplitWorkerMir
open NativeSplit (Tickets Owner Departures AlphaReads)
open SplitSchedulerMir
open SplitSchedulerMir.OwnerFields (Initialization laterWrites phases moveLoad)

theorem normal_credit {g : RawMask ws} (closed : ClosedAfterFirst g)
    (causal : ∀ x y, HappensBefore hist x y → time x<time y)
    (p : WorkerReset hist time g slot) :
    ∃ j : Fin (activation g closed).actions.length,
      (activation g closed).actions.get j=.finish p.worker p.release := by
  have bit:=p.scheduled.normal_release_is_set causal p.epilogue
  rw [p.maskWord] at bit
  obtain ⟨j,eq⟩:=credit g closed p.index (by simpa only [keeps,p.observed] using bit)
  exact ⟨j,eq.trans p.observed⟩

private def lastWrite (pre current post : List Write) (present : current ≠ []) :
    Fin (pre ++ current ++ post).length :=
  ⟨pre.length + current.length - 1, by
    have := List.length_pos_iff.mpr present
    simp only [List.length_append]; omega⟩

private theorem lastWrite_end (pre current post : List Write) (present : current ≠ []) :
    (lastWrite pre current post present).val + 1 = pre.length + current.length := by
  have := List.length_pos_iff.mpr present
  simp only [lastWrite]; omega

abbrev subsequent {g : RawMask ws} (exit : OwnerExit hist g) (slot : Slot) (field : SplitSchedulerMir.OwnerFields.Field) :=
  laterWrites exit.timeline slot.2 field (phases exit.log.length).length

structure OwnerSetup {ws : List Write} (g : RawMask ws) (hist : History) (time : Event → Nat)
    (slot : Slot) (t : Tickets rc pv beta) where
  exit : OwnerExit hist g
  gates : GateHistory hist g
  causal : ∀ x y, HappensBefore hist x y → time x<time y
  thread : exit.timeline.worker = slot.1.val
  mark : (ws.get g.mark).event = exit.timeline.atStep (SplitSchedulerMir.initPhases.idxOf .mark)
  bestSeed : t.b.event = exit.timeline.atStep (SplitSchedulerMir.initPhases.idxOf .seed)
  alphaSeed : t.a.event = exit.timeline.atStep (SplitSchedulerMir.initPhases.idxOf .seedAlpha)
  moveSeed : t.m.event = exit.timeline.atStep (SplitSchedulerMir.initPhases.idxOf .seedMove)
  sync : ∀ pre s post, exit.log=pre++ .mask s::post → ∀ i,
    InReleaseSequence ws i s → hist.otherSync (ws.get i).event (exit.clock.atStep (4+pre.length))
  bPre : List Write
  mPre : List Write
  later : List Initialization
  resets : List (WorkerReset hist time g slot)
  completed : t.dispatches.all fun d => !(d.sections.any WorkerFields.Section.needsReset) ||
    (d.retired t.stream && resets.any (fun e => e.worker.val == d.clock.worker && e.release == d.reset))
  check : SplitReturn.Observation.OwnerCheck hist
  running : check.stopped=false

structure PhysicalOwner {ws : List Write} (g : RawMask ws) (hist : History) (time : Event → Nat)
    (slot : Slot) (t : Tickets rc pv beta) extends OwnerSetup g hist time slot t where
  source : Fin (bPre++t.final.best++subsequent exit slot .score later).length
  moveSource : Fin (mPre++t.final.moves++subsequent exit slot .move later).length
  coherence : SplitRebookingMir.Coherence hist _ (scoreLoad _ source exit.clock exit.log)
  moveCoherence : SplitRebookingMir.Coherence hist _ (moveLoad exit.timeline exit.log.length moveSource)

theorem PhysicalOwner.reset {g : RawMask ws} {t : Tickets rc pv beta}
    (p : PhysicalOwner g hist time slot t) {a : WorkerFields.Address}
    (found : WorkerFields.locate t.dispatches obs = some a) (required : a.row.needsReset) :
    WorkerFields.resetBetween a.dispatch.opening a.dispatch.stamps a.dispatch.reset.step
      (NativeFrames.Returns.localEvents a.dispatch.worker t.stream) = true ∧
      ∃ e ∈ p.resets, e.worker.val = a.dispatch.clock.worker ∧ e.release = a.dispatch.reset := by
  have checked := List.all_eq_true.mp p.completed _ (WorkerFields.located found).1
  have needs := List.any_eq_true.mpr ⟨_, a.in_run, required⟩
  simp only [needs, Bool.not_true, Bool.false_or] at checked
  obtain ⟨e, member, same⟩ := List.any_eq_true.mp (Bool.and_eq_true_iff.mp checked).2
  exact ⟨a.dispatch.traced_reset (WorkerFields.located found).2.2 (Bool.and_eq_true_iff.mp checked).1,
    e, member, by simpa using same⟩

noncomputable def PhysicalOwner.owner {g : RawMask ws} {t : Tickets rc pv beta}
    (p : PhysicalOwner g hist time slot t) : Owner t := by
  have orders := FieldStores.modification_orders t.calls (IteratorFields.field_replay t.replay)
  have best : t.final.best ≠ [] := by rw [orders.2.1]; simp [FieldStores.initialLog]
  have moves : t.final.moves ≠ [] := by rw [orders.2.2]; simp [FieldStores.initialLog]
  exact {
    history:=hist
    mask:=words g
    activation:=activation g p.gates.closed
    clock:=p.exit.clock
    guards:=p.exit.log.map (guard g)
    returned:=exit_returned p.exit
    live:=open_pool g p.exit.poolOpen
    bPre:=p.bPre
    bPost:=subsequent p.exit slot .score p.later
    mPre:=p.mPre
    mPost:=subsequent p.exit slot .move p.later
    lastB:=lastWrite _ _ _ best
    source:=p.source
    lastM:=lastWrite _ _ _ moves
    move:=moveLoad p.exit.timeline p.exit.log.length p.moveSource
    endB:=lastWrite_end _ _ _ best
    endM:=lastWrite_end _ _ _ moves
    memory:=memory p.exit p.gates.closed p.mark p.sync _ p.source p.coherence
    seedScore:=p.bestSeed
    seedMove:=p.moveSeed ▸ SplitSchedulerMir.OwnerFields.move_before_assign p.exit.timeline
    afterScore:=by simpa only [scoreLoad,List.length_map] using
      SplitSchedulerMir.OwnerFields.read_order p.exit.timeline p.exit.log p.source p.moveSource
    release:=by
      intro call mem op here w wrote
      obtain ⟨c, _, rfl⟩ := List.mem_map.mp mem
      change op ∈ t.traffic c.order at here
      obtain ⟨obs, addr, found, fields⟩ := WorkerFields.traffic_source here
      obtain ⟨_, epilogue, _, _, reset⟩ := p.reset found (addr.needs_reset_of_access (op := op) (by rw [← fields]; exact here))
      obtain ⟨index, effective⟩ := normal_credit p.gates.closed p.causal epilogue
      refine ⟨⟨addr.unlock, index, by rw [effective]; rfl, ?_, ?_⟩⟩
      · rw [effective]; change SequencedBefore addr.unlock epilogue.release
        rw [reset]; exact addr.unlock_before_reset
      · intro op mem w store
        exact addr.store_before_unlock (op := op) (by rw [← fields]; exact mem) store
    nextScore:=by
      intro i pos
      rw [lastWrite_end] at pos
      rw [show (scoreLoad _ p.source p.exit.clock (p.exit.log.map (guard g))).event =
        (scoreLoad _ p.source p.exit.clock p.exit.log).event by simp [scoreLoad], SplitSchedulerMir.OwnerFields.score_event]
      exact SplitSchedulerMir.OwnerFields.before_reuse _ _ .score _ _ _ _ _ (by simp [SplitSchedulerMir.OwnerFields.phases_eq]) i pos
    moveMemory:=⟨by
      intro i pos
      rw [lastWrite_end] at pos
      exact SplitSchedulerMir.OwnerFields.before_reuse _ _ .move _ _ _ _ _
        (by rw [(SplitSchedulerMir.OwnerFields.return_indices _).2]; simp [SplitSchedulerMir.OwnerFields.phases_eq]) i pos,
      p.moveCoherence.writeRead, p.moveCoherence.readWrite⟩
    check:=p.check
    running:=p.running }

theorem PhysicalOwner.read_sources {g : RawMask ws} {t : Tickets rc pv beta}
    (p : PhysicalOwner g hist time slot t) :
    p.source.val + 1 = p.bPre.length + t.final.best.length ∧
      p.moveSource.val + 1 = p.mPre.length + t.final.moves.length := by
  obtain ⟨wait, score, move, _⟩ := p.owner.reads
  have lastB := (SplitReturn.Observation.scoreReadContract_proven _ _ _ _ wait score).1
  have lastM := (SplitReturn.Observation.scoreReadContract_proven _ _ _ _ wait move).1
  exact ⟨(congrArg (fun i => i.val + 1) lastB).trans p.owner.endB,
    (congrArg (fun i => i.val + 1) lastM).trans p.owner.endM⟩

theorem PhysicalOwner.wait_event {g : RawMask ws} {t : Tickets rc pv beta}
    (p : PhysicalOwner g hist time slot t) :
    p.owner.wait.read.event=p.exit.clock.atStep (3+p.exit.log.length) := by
  rw [NativeSplit.Owner.wait_event]
  simp only [owner,List.length_map]

def Effective (g : RawMask ws) (w : Worker) : Prop :=
  ∃ i : Fin g.actions.length, ∃ e, g.actions.get i=.finish w e ∧ (ws.get (g.before i)).value.getLsbD w.val=true

theorem released_iff (g : RawMask ws) (closed : ClosedAfterFirst g) (w : Worker) :
    SplitReturn.Released (activation g closed).actions w ↔ Effective g w := by
  constructor
  · rintro ⟨e,mem⟩
    obtain ⟨i,here,eq⟩:=List.mem_map.mp mem
    have kept:=(List.mem_filter.mp here).2
    exact ⟨i,e,eq,by simpa only [keeps,eq] using kept⟩
  · rintro ⟨i,e,op,bit⟩
    obtain ⟨j,eq⟩:=credit g closed i (by simpa only [keeps,op] using bit)
    exact ⟨e,(eq.trans op) ▸ List.get_mem (activation g closed).actions j⟩

theorem booked (g : RawMask ws) (closed : ClosedAfterFirst g) (w : Worker) (e : Event)
    (here : .book w e ∈ g.actions) : .book w e ∈ (activation g closed).actions := by
  obtain ⟨i,op⟩:=List.get_of_mem here
  obtain ⟨j,eq⟩:=credit g closed i (by simp only [keeps,op])
  exact (eq.trans op) ▸ List.get_mem (activation g closed).actions j

structure Team {g : RawMask ws} {t : Tickets rc pv beta} (p : PhysicalOwner g hist time slot t) : Prop where
  participant : ∀ i ∈ t.workers, i=g.owner ∨ ∃ e, .book i e ∈ g.actions
  released : ∀ i, Effective g i → i ∈ t.workers
  exhausted : ∀ i, Effective g i → (t.result i).completed.departure=.exhausted →
    Nonempty (SplitWorkerMir.Iterator.Drained rc.length t.start t.state i.val)
  stopped : let calls:=IteratorFields.chronological t.state.writes t.journal t.stream t.workers
    let event:=p.exit.clock.atStep (3+p.exit.log.length)
    ∀ c ∈ calls, c.ticket.locked.stopped=true →
      Nonempty (SplitWorkerMir.Stop.Poll p.check event calls) ∨
        (∀ obs, p.check.shared=some obs → SplitReturn.Observation.AbortCause hist obs event)

theorem Team.departures {g : RawMask ws} {t : Tickets rc pv beta} {p : PhysicalOwner g hist time slot t}
    (team : Team p) : Departures p.owner where
  participant w h := (team.participant w h).imp_right fun ⟨e,here⟩ => ⟨e,booked g p.gates.closed w e here⟩
  released w h := team.released w ((released_iff g p.gates.closed w).1 h)
  exhausted w h := team.exhausted w ((released_iff g p.gates.closed w).1 h)
  stopped := by
    dsimp only
    rw [p.wait_event]
    exact team.stopped

end NeuralReversi.PvsParallel.SlotFrameMir.RawProjection

module
public import NeuralReversi.PvsParallel.SlotFrameMir.RawTeam
public import NeuralReversi.PvsParallel.NativeSplit.AlphaOrder
public import NeuralReversi.PvsParallel.SplitSchedulerMir.Startup
public import NeuralReversi.PvsParallel.NativeSplit.AlphaLoads

@[expose] public section

namespace NeuralReversi.PvsParallel.SlotFrameMir.RawProjection

open Threading
open AtomicWord (Event Write)
open SplitWorkerMir (Step)
open NativeSplit (Tickets Handoff)
open SplitSchedulerMir

theorem address {t : Tickets rc pv beta} (x : IteratorFrames.Posted rc t.state t.journal t.stream earlier r seq) :
    ∃ addr, WorkerFields.locate t.dispatches x.lock = some addr ∧
      addr.aligned x.lock x.body.steps = true ∧ addr.row.needsReset = true := by
  obtain ⟨addr,found,matched⟩ := WorkerFields.for_posted x (List.all_eq_true.mp t.traced _ x.issued)
  refine ⟨addr,found,matched,?_⟩
  unfold WorkerFields.Section.needsReset
  rw [(addr.observed matched).1, x.running]
  rfl

/-- The replayed field section supplies both the report's clock and the mask reset
    used by the owner's reads. All tickets in this dispatch share the same reset. -/
noncomputable def PhysicalOwner.release {g : RawMask ws} {t : Tickets rc pv beta} (p : PhysicalOwner g hist time slot t)
    (x : IteratorFrames.Posted rc t.state t.journal t.stream earlier r seq) :
    IteratorReports.Release x p.owner.activation := by
  let addr := Classical.choose (address (t := t) x)
  obtain ⟨found,matched,needs⟩ := Classical.choose_spec (address (t := t) x)
  have aligned := addr.observed matched
  let closed := (p.reset found needs).2
  let e := Classical.choose closed
  have event := (Classical.choose_spec closed).2
  let credit := normal_credit p.gates.closed p.causal e
  refine {
    prior := addr.prior
    later := addr.later
    clock := addr.dispatch.projected
    worker := e.worker
    actor := event.1.trans (WorkerFields.located found).2.1
    clockWorker := event.1.symm
    observed := aligned.2.2
    finish := Classical.choose credit
    rmw := ?_ }
  apply (Classical.choose_spec credit).trans
  apply congrArg (SplitCompletion.Action.finish e.worker)
  apply event.2.trans
  exact congrArg (fun steps => addr.dispatch.projected.atStep (releaseIndex steps))
    (addr.steps_eq.trans (by rw [aligned.2.1]))

structure Reports {g : RawMask ws} {t : Tickets rc pv beta} (p : PhysicalOwner g hist time slot t) where
  pre : List Write
  mutex : Mutex.Capture hist
  checked : Mutex.covers mutex.ops (t.journal.map fun row => (row.acquired,row.published))
  stateLocks : Startup.StateLocks hist
  started : t.dispatches.all fun d => !(d.sections.any WorkerFields.Section.needsReset) ||
    Startup.ready mutex.ops (Startup.stateOps stateLocks) (p.exit.timeline.atStep (initPhases.idxOf .assign)) d.entry
  memory : NativeSplit.AlphaLoads.Memory hist (pre ++ t.final.alpha ++ subsequent p.exit slot .alpha p.later)
  operands : t.state.writes.all (NativeSplit.AlphaLoads.ticket memory.reads t.journal t.stream)

def Reports.loads {g : RawMask ws} {t : Tickets rc pv beta} {p : PhysicalOwner g hist time slot t}
    (data : Reports p) (part : t.reports = earlier ++ r :: later) :
    IteratorReports.Loads (t.post part) hist (data.pre ++ t.final.alpha ++ subsequent p.exit slot .alpha p.later) t.a.event := by
  let x := t.post part
  obtain ⟨safe,scout,wide⟩ := NativeSplit.AlphaLoads.checked x (List.all_eq_true.mp data.operands _ x.issued)
  have publish (again : Bool) (used : again = true → SplitWorkerMir.retry x.input = true) :
      SplitPublication.PublishedThrough hist t.a.event (x.lock.stamp (.load again)) := by
    obtain ⟨addr,found,matched,required⟩ := address (t := t) x
    have accepted := List.all_eq_true.mp data.started _ (WorkerFields.located found).1
    have needed := List.any_eq_true.mpr ⟨_,addr.in_run,required⟩
    simp only [needed,Bool.not_true,Bool.false_or] at accepted
    apply Startup.published data.mutex data.stateLocks accepted _ (addr.entry_before_load x.searched matched again used)
    rw [p.alphaSeed]
    exact ⟨rfl,p.exit.timeline.advances _ _ (by decide)⟩
  exact ⟨safe,data.memory.load scout (publish false (by simp)),fun yes => data.memory.load (wide yes) (publish true (fun _ => yes))⟩

theorem Reports.after_acquire {g : RawMask ws} {t : Tickets rc pv beta} {p : PhysicalOwner g hist time slot t}
    (data : Reports p) (part : t.reports = earlier ++ r :: later)
    (i : Fin (data.pre ++ t.final.alpha ++ subsequent p.exit slot .alpha p.later).length)
    (pos : i.val = ((t.interval part).last data.pre (subsequent p.exit slot .alpha p.later)).val + 1) :
    CutoffVisibility.HappensBefore hist (t.post part).lock.acquired
      ((data.pre ++ t.final.alpha ++ subsequent p.exit slot .alpha p.later).get i).event := by
  let x := t.post part
  let pub := (p.release x).publication
  rcases t.next_alpha part _ _ i pos with ⟨k, row, le, atRow, holds⟩ | reuse
  · rcases Nat.eq_or_lt_of_le le with eq | lt
    · rw [← eq, x.atLock] at atRow
      cases atRow
      exact .sequenced holds
    · obtain ⟨_, seen, sections⟩ := data.mutex.sections data.checked
      rw [List.flatMap_map] at sections
      have locked := IteratorReports.unlock_before_later x (data.mutex.trace seen) sections lt atRow
      exact .trans (.sequenced pub.afterAcquire) (.trans (.sequenced pub.afterCheck) (.trans locked (.sequenced holds)))
  · apply pub.beforeOwnerRead p.owner.wait
    rw [p.wait_event]
    have shifted : p.exit.clock.atStep (3 + p.exit.log.length) =
        p.exit.timeline.atStep (3 + p.exit.log.length + 2) := by
      simp [OwnerExit.clock, OwnerFields.maskClock, Clock.atStep, show ¬ 3 + p.exit.log.length < 3 by omega]
    rw [shifted]
    apply OwnerFields.before_reuse p.exit.timeline slot.2 .alpha p.exit.log.length p.later data.pre t.final.alpha
      (3 + p.exit.log.length + 2) _ i reuse
    simp [OwnerFields.phases_eq]; omega

noncomputable def Reports.alpha {g : RawMask ws} {t : Tickets rc pv beta} {p : PhysicalOwner g hist time slot t}
    (reported : Reports p) : NativeSplit.AlphaReads p.owner where
  pre:=reported.pre
  post:=subsequent p.exit slot .alpha p.later
  finish part:=p.release (t.post part)
  loads part:=⟨reported.loads part, reported.after_acquire part⟩

/-- Both field loads and all worker report publications used by the ordinary
    numeric split come from the same projected mask and the recorded owner clock. -/
noncomputable def handoff {g : RawMask ws} {t : Tickets rc pv beta} (p : PhysicalOwner g hist time slot t)
    (team : Team p) (reported : Reports p) (alpha : Int) (size : rc.length=len) :
    Handoff p.owner alpha len :=
  NativeSplit.handoff p.owner team.departures reported.alpha alpha size

end NeuralReversi.PvsParallel.SlotFrameMir.RawProjection

module
public import NeuralReversi.PvsParallel.NativeSplit

@[expose] public section

namespace NeuralReversi.PvsParallel.NativeSplit

open Threading.AtomicWord (Event)
open PvsBoardTree (Position GeneratedSquare)
open NativeWorkerFeed (labels)
open Threading.SplitResult (Report)

structure Stamps (x : IteratorFrames.Posted rc s journal stream earlier r seq)
    (split : SplitExecution len pv alpha beta score) (part : split.reports = earlier ++ r :: later) : Prop where
  worker : (split.reportPublications earlier r later part).val.worker.val = x.lock.worker
  acquired : (split.reportPublications earlier r later part).val.acquired = x.lock.acquired
  checked : (split.reportPublications earlier r later part).val.checked = x.lock.polled
  published : (split.reportPublications earlier r later part).val.published = x.lock.published
  scout : (split.reportReads earlier r later part).scoutLoad.event = x.lock.scoutLoad
  alphaWord : (split.reportReads earlier r later part).alpha = x.input.alpha.toInt
  scoutWord : (split.reportReads earlier r later part).scout = x.input.scout.toInt
  wide : SplitWorkerMir.retry x.input = true → (split.reportReads earlier r later part).wideLoad.event = x.lock.wideLoad
  freshWord : SplitWorkerMir.retry x.input = true → (split.reportReads earlier r later part).fresh = x.input.fresh.toInt
  wideWord : (split.reportReads earlier r later part).wide = x.input.wide.toInt

structure Handoff {t : Tickets rc pv beta} (o : Owner t) (alpha : Int) (len : Nat) where
  split : SplitExecution len pv alpha beta o.score
  start : split.start.toNat = t.start.toNat
  entry : split.entry = t.initial.state
  reports : split.reports = t.reports
  history : split.history = o.history
  loaded : split.result.event = o.load.event
  chosen : split.MoveRead TranspositionTable.Packing.encodeBestMove (IteratorFields.atIndex rc)
    t.initial.picked t.final.fields.picked
  stamps : ∀ {earlier r later} (part : split.reports = earlier ++ r :: later),
    Stamps (t.post (reports.symm.trans part)) split part

noncomputable def handoff {t : Tickets rc pv beta} (o : Owner t) (workers : Departures o) (loads : AlphaReads o)
    (alpha : Int) (size : rc.length = len) : Handoff o alpha len := by
  subst len
  refine ⟨execution alpha o workers loads,rfl,rfl,rfl,rfl,rfl,moveRead alpha o workers loads,?_⟩
  intro earlier r later part
  let row := loads.occurrence part
  exact ⟨row.worker,row.acquired,row.checked,row.published,row.receipt.scoutStamp,row.receipt.alphaWord,
    row.receipt.scoutWord,row.receipt.wideStamp,row.receipt.freshWord,row.receipt.wideWord⟩

theorem moveCount (b : Position) (order : List (GeneratedSquare mc b.val))
    (same : NativeWorkerFeed.frozen mc fc b rc order = true) : rc.length = (labels mc fc b order).length := by
  have eq := of_decide_eq_true same
  simpa only [labels,List.length_map] using congrArg List.length eq

noncomputable def Handoff.clock {t : Tickets rc pv beta} {o : Owner t}
    (h : Handoff o alpha (labels mc fc b order).length) (part : h.split.reports = earlier ++ r :: later) :=
  let x := t.post (h.reports.symm.trans part)
  NativeWorkerFeed.forReport (moves := labels mc fc b order) h.split earlier r later part x.op.worker
    x.lock.dispatch x.op.opening x.lock.nextClaim x.lock.scoutOpening x.lock.wideOpening x.lock.ends

private theorem Handoff.at_step {t : Tickets rc pv beta} {o : Owner t}
    (h : Handoff o alpha (labels mc fc b order).length) (part : h.split.reports = earlier ++ r :: later)
    (step : SplitWorkerMir.Step)
    (member : step ∈ (t.post (h.reports.symm.trans part)).body.steps ++ SplitWorkerMir.lockedSteps) :
    (h.clock part).stops step = (t.post (h.reports.symm.trans part)).lock.stamp step := by
  let x := t.post (h.reports.symm.trans part)
  have z := h.stamps part
  cases step with
  | lock => exact z.acquired
  | poll => exact z.checked
  | unlock => exact z.published
  | returned => rfl
  | load wide =>
    cases wide with
    | false => exact z.scout
    | true =>
      apply z.wide
      change .load true ∈ x.body.steps ++ SplitWorkerMir.lockedSteps at member
      rw [SplitWorkerMir.issued_of_search x.searched] at member
      cases again : SplitWorkerMir.retry x.input <;> simp_all [SplitWorkerMir.issued,SplitWorkerMir.lockedSteps]

private theorem Handoff.ordered {t : Tickets rc pv beta} {o : Owner t}
    (h : Handoff o alpha (labels mc fc b order).length) (loads : AlphaReads o)
    (part : h.split.reports = earlier ++ r :: later) :
    let x := t.post (h.reports.symm.trans part)
    NativeWorkerFeed.ordered (h.clock part).stops (x.body.steps ++ SplitWorkerMir.lockedSteps) = true := by
  intro x
  apply decide_eq_true
  apply List.pairwise_iff_getElem.mpr
  intro i j hi hj lt
  let xs := x.body.steps ++ SplitWorkerMir.lockedSteps
  let release := loads.finish (h.reports.symm.trans part)
  have stamp (k : Nat) (bound : k < xs.length) :
      (h.clock part).stops xs[k] = release.clock.atStep (release.prior.length + k) :=
    (h.at_step part _ (List.getElem_mem bound)).trans (release.observed k _ (List.getElem?_eq_getElem bound))
  rw [stamp i hi,stamp j hj]
  exact ⟨rfl,release.clock.advances _ _ (by omega)⟩

private theorem Handoff.return_ok {t : Tickets rc pv beta} {o : Owner t}
    (h : Handoff o alpha (labels mc fc b order).length) (part : h.split.reports = earlier ++ r :: later)
    (q : SplitWorkerMir.Query) (rows : List NativeFrames.Row) :
    let x := t.post (h.reports.symm.trans part)
    NativeWorkerFeed.stepOk mc fc b order rc x.op.old rows x.returned (h.clock part) (.returned q) =
      NativeWorkerFeed.stepOk mc fc b order rc x.op.old rows x.returned x.lock.prefix (.returned q) := by
  intro x
  cases cell : NativeWorkerFeed.ticket rc x.op.old <;> cases sq : order[q.index]? <;>
    simp only [NativeWorkerFeed.stepOk,cell,sq]
  rename_i entry chosen
  apply Bool.eq_iff_iff.mpr
  simp only [NativeWorkerFeed.recorded,NativeWorkerFeed.Prefix.opening,clock,NativeWorkerFeed.forReport,
    IteratorFrames.Observation.prefix,SplitWorkerMir.stamp,IteratorFrames.Observation.stamp,
    Bool.and_eq_true,List.any_eq_true,decide_eq_true_eq]
  simp only [x.worker,x.opening]
  rfl

theorem Handoff.calls {t : Tickets rc pv beta} {o : Owner t} {pool : EndgameSearch.CacheHistory.Pool}
    (h : Handoff o alpha (labels mc fc b order).length) (loads : AlphaReads o)
    (inputs : Nat → Nat → Option NativeFrames.Args) (cap : NativeFrames.Capture pool)
    (decoded : NativeFrames.decode pool inputs t.stream = some cap)
    (unique : IteratorFrames.unique t.state.writes t.journal = true)
    (board : NativeWorkerFeed.frozen mc fc b rc order = true)
    (frames : ∀ {earlier r later} (part : h.split.reports = earlier ++ r :: later),
      let x := t.post (h.reports.symm.trans part)
      ∀ q, SplitWorkerMir.Step.returned q ∈ x.body.steps ++ SplitWorkerMir.lockedSteps →
        NativeWorkerFeed.stepOk mc fc b order rc x.op.old cap.rows x.returned x.lock.prefix (.returned q) = true)
    (identify before sequenced) (consumer : Nat)
    (hb : ∀ a z, Threading.CutoffVisibility.HappensBefore o.history a z → before (identify a) (identify z))
    (after : before (identify o.load.event) consumer) :
    ReportCalls (ConcurrentReplay.UsedChild identify before
      (NativeFrames.logOf mc fc cap.rows cap.table identify before sequenced).events consumer) (labels mc fc b order) h.split := by
  apply SplitWorkerMir.reportCalls_of_worker_frames h.split identify before _ consumer (h.history ▸ hb) (h.loaded ▸ after)
  intro earlier r later part
  let x := t.post (h.reports.symm.trans part)
  have stamps := h.stamps part
  have observed := loads.loads (h.reports.symm.trans part)
  have window := IteratorFields.posted_window t.covers t.ran t.replay x
  have steps : x.frameWords mc fc b order cap.rows (h.clock part) = true := by
    unfold IteratorFrames.Posted.frameWords
    have ordered : NativeWorkerFeed.ordered (h.clock part).stops (x.body.steps ++ SplitWorkerMir.lockedSteps) = true :=
      h.ordered loads part
    simp only [board,ordered,Bool.true_and]
    apply List.all_eq_true.mpr
    intro step member
    cases step with
    | returned q => exact (h.return_ok part q cap.rows).trans (frames part q member)
    | _ => rfl
  have ok : x.checked h.split part cap = true := by
    apply decide_eq_true
    exact ⟨steps,stamps.worker.symm,stamps.acquired.symm,stamps.checked.symm,stamps.published.symm,
      stamps.scout.symm,window.1,window.2,stamps.alphaWord.symm,stamps.scoutWord.symm,
      observed.safe.alphaLow,observed.safe.alphaHigh,observed.safe.scout,fun retry =>
        ⟨(stamps.wide retry).symm,(stamps.freshWord retry).symm,stamps.wideWord.symm,observed.safe.beta retry,
          observed.safe.fresh retry,observed.safe.wide retry⟩⟩
  have source : ∃ j, x.fromNative pool inputs h.split part identify before sequenced = some ⟨cap,j⟩ := by
    unfold IteratorFrames.Posted.fromNative
    simp only [unique,Bool.not_true,Bool.false_eq_true,↓reduceIte,decoded]
    cases found : x.segment h.split part cap identify before sequenced with
    | none => unfold IteratorFrames.Posted.segment at found; simp [ok] at found
    | some j => exact ⟨j,by simp [found]⟩
  exact Classical.choose source

end NeuralReversi.PvsParallel.NativeSplit

import NeuralReversi.PvsParallel.SplitWorkerMir.Correct

namespace NeuralReversi.PvsParallel.SplitWorkerMir

open Threading
open AtomicWord (Event Read Write SequencedBefore signed32At)
open SplitWindow (State StoreWindow AlphaRead reSearch)
open SplitResult (Report)
open PvsMoveLoop (Tree)
open ConcurrentReplay (UsedChild)

def recordLocked (i : Input) (body : Searched) (locked : Locked) : Option Report :=
  if (commit i locked body.chosen).exit = .stopped then none else some (selected i body)

theorem recordLocked_eq (i : Input) (body : Searched) (locked : Locked) :
    recordLocked i body locked = if locked.stopped then none else some (selected i body) := by
  unfold recordLocked
  rw [commit_eq]
  cases locked.stopped <;> simp <;> split <;> decide

def readFromWorker (i : Input) (body : Searched) (hit : search i = some body) (safe : Safe i)
    (writes : List Write) (first last : Fin writes.length) (scoutLoad wideLoad : Read writes)
    (source : i.alpha.toInt = signed32At writes scoutLoad.source)
    (fresh : i.fresh.toInt = signed32At writes wideLoad.source)
    (probe : AlphaRead history scoutLoad first last)
    (stores : StoreWindow writes first last i.pv i.beta.toInt initial (earlier.map Report.score))
    (reread : retry i = true → AlphaRead history wideLoad first last) :
    ReportReads history i.pv i.beta.toInt initial earlier (selected i body) := by
  refine ⟨writes, first, last, scoutLoad, wideLoad, i.scout.toInt, i.wide.toInt, probe, stores, ?_, ?_⟩
  · rw [← source, ← retry_eq safe]
    exact reread
  · rw [← source, ← fresh]
    exact selected_eq hit safe

structure ProgramOrder (time : Step → Event) (steps : List Step) : Prop where
  adjacent : ∀ a b pre post, steps = pre ++ a :: b :: post → SequencedBefore (time a) (time b)

structure FrameFeed (moves : List Tree) (identify : Event → Nat)
    (events : List ConcurrentReplay.Event) (time : Step → Event) (steps : List Step) : Prop
    extends ProgramOrder time steps where
  returned : ∀ q, .returned q ∈ steps →
    ∃ event ∈ events, ∃ frame, event.id = identify (time (.returned q)) ∧
      event.action = .returned frame ⟨childAt moves q.index, q.pv, q.alpha.toInt, q.beta.toInt, q.result.toInt⟩

private theorem FrameFeed.prefix {moves : List Tree} {identify : Event → Nat} {events time xs tail}
    (feed : FrameFeed moves identify events time (xs ++ tail)) : FrameFeed moves identify events time xs where
  adjacent a b pre post same := feed.adjacent a b pre (post ++ tail) (by simp [same])
  returned q member := feed.returned q (List.mem_append_left _ member)

/-- The unlock is a publication point even for a score that leaves the split fields unchanged. -/
def publicationFromWorker (activation : SplitCompletion.Activation mask) (i : Input)
    (body : Searched) (locked : Locked) (hit : search i = some body)
    (credited : recordLocked i body locked = some report)
    (time : Step → Event) (order : ProgramOrder time (body.steps ++ lockedSteps))
    (worker : SplitCompletion.Worker) (finish : Fin activation.actions.length)
    (finished : activation.actions.get finish = .finish worker (activation.actions.get finish).event)
    (sameWorker : (time .unlock).worker = worker.val)
    (beforeFinish : SequencedBefore (time .unlock) (activation.actions.get finish).event) :
    SplitReturn.Observation.ReportPublication activation := by
  rw [recordLocked_eq] at credited
  have running : locked.stopped = false := by
    cases h : locked.stopped
    · rfl
    · rw [h] at credited; contradiction
  let pre : List Step := if retry i then
    [.load false, .returned (scoutQuery i), .load true, .returned (wideQuery i)]
    else [.load false, .returned (scoutQuery i)]
  have last : body.steps = pre ++ [.lock] := by
    rw [issued_of_search hit]
    cases h : retry i <;> simp [issued, pre, h]
  exact ⟨worker, time .lock, time .poll, time .unlock, locked.stopped, running,
    order.adjacent .lock .poll pre [.unlock] (by simp [last, lockedSteps]),
    order.adjacent .poll .unlock (pre ++ [.lock]) [] (by simp [last, lockedSteps]),
    finish, finished, sameWorker, beforeFinish⟩

def stamp (reads : ReportReads history pv beta initial earlier report)
    (publication : SplitReturn.Observation.ReportPublication activation)
    (returns : Bool → Event) : Step → Event
  | .load again => if again then reads.wideLoad.event else reads.scoutLoad.event
  | .returned q => returns q.pv
  | .lock => publication.acquired
  | .poll => publication.checked
  | .unlock => publication.published

structure Segment {moves : List Tree} (split : SplitExecution moves.length pv alpha beta score)
    (earlier : List Report) (report : Report) (later : List Report)
    (same : split.reports = earlier ++ report :: later)
    (identify : Event → Nat) (events : List ConcurrentReplay.Event) where
  input : Input
  body : Searched
  locked : Locked
  searched : search input = some body
  credited : recordLocked input body locked = some report
  safe : Safe input
  returns : Bool → Event
  feed : FrameFeed moves identify events
    (stamp (split.reportReads earlier report later same)
      (split.reportPublications earlier report later same).val returns) (body.steps ++ lockedSteps)
  pvTag : input.pv = pv
  betaWord : input.beta.toInt = beta
  scoutLoad : input.alpha.toInt = (split.reportReads earlier report later same).alpha
  scoutWord : input.scout.toInt = (split.reportReads earlier report later same).scout
  wideLoad : retry input = true → input.fresh.toInt = (split.reportReads earlier report later same).fresh
  wideWord : retry input = true → input.wide.toInt = (split.reportReads earlier report later same).wide

private theorem reportUse_shape {moves : List Tree} (split : SplitExecution moves.length pv alpha beta score)
    (earlier : List Report) (report : Report) (later : List Report) (same : split.reports = earlier ++ report :: later)
    (wide : Bool) :
    let r := split.reportReads earlier report later same
    let pub := (split.reportPublications earlier report later same).val
    ∃ u, split.reportUse earlier report later same wide = .report u ∧ u.history = split.history ∧
      u.load = (if wide then r.wideLoad.event else r.scoutLoad.event) ∧
      u.next = (if wide = false ∧ reSearch pv r.alpha r.scout = true then r.wideLoad.event else pub.acquired) ∧
      u.ownerRead = split.result.event := by
  intro r pub
  dsimp only [SplitExecution.reportUse, r, pub]
  by_cases h : wide = false ∧ reSearch pv r.alpha r.scout = true
  · rw [dite_eq_left h, ite_eq_left h]
    exact ⟨_, rfl, rfl, rfl, rfl, rfl⟩
  · rw [dite_eq_right h, ite_eq_right h]
    exact ⟨_, rfl, rfl, rfl, rfl, rfl⟩

theorem reportCalls_of_worker_frames {moves : List Tree} (split : SplitExecution moves.length pv alpha beta score)
    (identify : Event → Nat) (before : Nat → Nat → Prop) (events : List ConcurrentReplay.Event)
    (consumer : Nat)
    (clock : ∀ a b, CutoffVisibility.HappensBefore split.history a b → before (identify a) (identify b))
    (owner : before (identify split.result.event) consumer)
    (segments : ∀ earlier report later (same : split.reports = earlier ++ report :: later),
      Segment split earlier report later same identify events) :
    ReportCalls (UsedChild identify before events consumer) moves split := by
  intro earlier report later same
  let r := split.reportReads earlier report later same
  let pub := (split.reportPublications earlier report later same).val
  let seg := segments earlier report later same
  let i := seg.input
  let time := stamp r pub seg.returns
  have feed : FrameFeed moves identify events time (issued i).steps := issued_of_search seg.searched ▸ seg.feed.prefix
  have record := seg.credited
  rw [recordLocked_eq] at record
  split at record
  · contradiction
  next =>
    have index : i.index = report.index := congrArg Report.index (Option.some.inj record)
    have retried : retry i = reSearch pv r.alpha r.scout := by
      rw [retry_eq seg.safe, seg.pvTag, seg.scoutLoad, seg.scoutWord]
    obtain ⟨us, scoutUse, sh, sl, sn, so⟩ := reportUse_shape split earlier report later same false
    obtain ⟨uw, wideUse, wh, wl, wn, wo⟩ := reportUse_shape split earlier report later same true
    simp only [Bool.false_eq_true, ite_false, true_and] at sl sn
    simp only [Bool.true_eq_false, false_and, ite_false, ite_true] at wl wn
    have qScout : (scoutQuery i).alpha.toInt = -(r.alpha + 1) := by
      change (Solve2.childAlpha i.alpha).toInt = _
      rw [Solve2.childAlpha_safe i.alpha seg.safe.alphaHigh, seg.scoutLoad]
    have bScout : (scoutQuery i).beta.toInt = -r.alpha := by
      change (-i.alpha).toInt = _; rw [Solve2.neg_safe _ seg.safe.alphaLow, seg.scoutLoad]
    have scoutMem : .returned (scoutQuery i) ∈ (issued i).steps := by unfold issued; split <;> simp
    obtain ⟨e, mem, frame, id, action⟩ := feed.returned _ scoutMem
    have eScout : e.action = .returned frame ⟨childAt moves report.index, false, -(r.alpha + 1), -r.alpha, r.scout⟩ := by
      rw [qScout, bScout] at action
      have raw : i.scout.toInt = r.scout := seg.scoutWord
      simpa only [scoutQuery, index, raw] using action
    constructor
    · rw [scoutUse]
      refine ⟨e, mem, time (.returned (scoutQuery i)), frame, id, eScout, ?_, ?_, sh ▸ clock, so ▸ owner⟩
      · rw [sl]
        cases h : retry i
        · exact feed.adjacent (.load false) (.returned (scoutQuery i)) [] [.lock] (by simp [issued,h])
        · exact feed.adjacent (.load false) (.returned (scoutQuery i)) [] [.load true, .returned (wideQuery i), .lock] (by simp [issued,h])
      · rw [sn, ← retried]
        cases h : retry i
        · exact feed.adjacent (.returned (scoutQuery i)) .lock [.load false] [] (by simp [issued,h])
        · exact feed.adjacent (.returned (scoutQuery i)) (.load true) [.load false] [.returned (wideQuery i), .lock] (by simp [issued,h])
    · intro needed
      have fired := retried.trans needed
      have wideMem : .returned (wideQuery i) ∈ (issued i).steps := by simp [issued, fired]
      obtain ⟨ew, member, framew, idw, actionw⟩ := feed.returned _ wideMem
      have qWide : (wideQuery i).alpha.toInt = -beta := by
        change (-i.beta).toInt = _; rw [Solve2.neg_safe _ (seg.safe.beta fired), seg.betaWord]
      have bWide : (wideQuery i).beta.toInt = -r.fresh := by
        change (-i.fresh).toInt = _; rw [Solve2.neg_safe _ (seg.safe.fresh fired), seg.wideLoad fired]
      rw [qWide, bWide] at actionw
      rw [wideUse]
      refine ⟨ew, member, time (.returned (wideQuery i)), framew, idw, ?_, ?_, ?_, wh ▸ clock, wo ▸ owner⟩
      · have raw : i.wide.toInt = r.wide := seg.wideWord fired
        simpa only [wideQuery, index, raw] using actionw
      · rw [wl]
        exact feed.adjacent (.load true) (.returned (wideQuery i)) [.load false, .returned (scoutQuery i)] [.lock] (by simp [issued,fired])
      · rw [wn]
        exact feed.adjacent (.returned (wideQuery i)) .lock [.load false, .returned (scoutQuery i), .load true] [] (by simp [issued,fired])

end NeuralReversi.PvsParallel.SplitWorkerMir

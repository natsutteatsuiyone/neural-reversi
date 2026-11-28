module
public import NeuralReversi.PvsParallel.IteratorFields
public import NeuralReversi.PvsParallel.SplitSchedulerMir.Worker

@[expose] public section

namespace NeuralReversi.PvsParallel.IteratorReports

open Threading
open AtomicWord (Event Read Write SequencedBefore signed32At)
open CutoffVisibility (History HappensBefore)
open SplitCompletion (Worker Activation)
open SplitResult (Report)
open SplitWindow (AlphaRead StoreWindow AlphaStores)
open SplitWorkerMir (Step Input Searched retry Safe)
open SplitWorkerMir.Iterator (WorkerResult LockedCall)
open IteratorFrames (Posted source ranked)
open MoveList.ConcurrentTrace (State)
open NativeFrames (Record)

private theorem retired_index (calls : List LockedCall) (done : SplitWorkerMir.Iterator.retire calls = some result)
    (picked : r ∈ result.completed.reports) : r.index ∈ result.completed.tickets := by
  induction calls generalizing result with
  | nil => cases done; contradiction
  | cons c rest ih =>
    unfold SplitWorkerMir.Iterator.retire at done
    obtain ⟨b,_,tail⟩ := Option.bind_eq_some_iff.mp done
    dsimp only at tail
    cases ex : (SplitWorkerMir.commit c.ticket.input c.ticket.locked b.chosen).exit with
    | stopped | cutoff =>
      cases rest <;> simp [ex] at tail
      subst result
      first | contradiction | simpa [SplitWorkerMir.selected] using congrArg Report.index (List.mem_singleton.mp picked)
    | again =>
      rw [ex] at tail
      obtain ⟨next,more,same⟩ := Option.bind_eq_some_iff.mp tail
      cases same
      rcases List.mem_cons.mp picked with head | later
      · exact List.mem_cons.mpr (.inl (congrArg Report.index head))
      · exact List.mem_cons_of_mem _ (ih more later)

private theorem key_unique (f : α → β) (xs : List α) (keys : (xs.map f).Nodup)
    (aIn : a ∈ xs) (bIn : b ∈ xs) (same : f a = f b) : a = b := by
  induction xs with
  | nil => contradiction
  | cons h t ih =>
    obtain ⟨fresh,more⟩ := List.nodup_cons.mp keys
    rcases List.mem_cons.mp aIn with e | m
    · subst a
      rcases List.mem_cons.mp bIn with e | n
      · exact e.symm
      · exact False.elim (fresh (List.mem_map.mpr ⟨b,n,same.symm⟩))
    · rcases List.mem_cons.mp bIn with e | n
      · subst b
        exact False.elim (fresh (List.mem_map.mpr ⟨a,m,same⟩))
      · exact ih more m n

theorem in_work (rc : List Move.Move) {s : State w} (run : MoveList.ConcurrentTrace.Execution rc.length start s)
    (fits : rc.length < 2 ^ w) (workers : List Worker) (out : Worker → WorkerResult)
    (ran : ∀ i ∈ workers, SplitWorkerMir.Iterator.worker (source journal stream) s.writes i.val = some (out i))
    (x : Posted rc s journal stream earlier r (ranked workers out))
    (worker : Worker) (actor : worker.val = x.lock.worker) : r ∈ (out worker).completed.reports := by
  have pub := (List.mergeSort_perm _ _).mem_iff.mp (List.mem_of_getElem? x.sorted)
  obtain ⟨j,registered,member⟩ := List.mem_flatMap.mp pub
  have observed := SplitWorkerMir.Iterator.worker_observed (ran j registered)
  have reported : r ∈ (out j).completed.reports := observed.2 ▸ List.mem_map.mpr ⟨(x.rank,r),member,rfl⟩
  obtain ⟨cs,_,done⟩ := Option.bind_eq_some_iff.mp (ran j registered)
  have index := retired_index cs done reported
  rw [observed.1] at index
  obtain ⟨op,owns,matched⟩ := List.mem_map.mp index
  obtain ⟨issued,team⟩ := List.mem_filter.mp owns
  have unique := (MoveList.ConcurrentTrace.consecutive _ run.replayed fits).2
  have sameOp := key_unique (fun a : MoveList.ConcurrentTrace.Write w => a.old.toNat) _ unique issued x.issued
    (matched.trans x.index)
  have id : j = worker := by
    apply Fin.ext
    simpa [sameOp,actor,x.worker] using (beq_iff_eq.mp team).symm
  exact id ▸ reported

structure Release {s : State w} (x : Posted rc s journal stream earlier r seq) (act : Activation mask) where
  prior : List Step
  later : List Step
  clock : SplitSchedulerMir.Clock
  worker : Worker
  actor : worker.val = x.lock.worker
  clockWorker : clock.worker = worker.val
  observed : ∀ n step, (x.body.steps ++ SplitWorkerMir.lockedSteps)[n]? = some step →
    x.lock.stamp step = clock.atStep (prior.length + n)
  finish : Fin act.actions.length
  rmw : act.actions.get finish = .finish worker
    (clock.atStep (SplitSchedulerMir.releaseIndex (prior ++ x.body.steps ++ SplitWorkerMir.lockedSteps ++ later)))

def Release.publication (t : Release x act) : SplitReturn.Observation.ReportPublication act :=
  SplitSchedulerMir.workerPublication act x.input x.body x.lock.locked x.searched x.credited t.prior t.later
    t.clock t.worker t.clockWorker x.lock.stamp t.observed t.finish t.rmw

theorem Release.stamps (t : Release x act) :
    t.publication.worker = t.worker ∧ t.publication.acquired = x.lock.acquired ∧
      t.publication.checked = x.lock.polled ∧ t.publication.published = x.lock.published := ⟨rfl,rfl,rfl,rfl⟩

theorem Release.before (x : Posted rc s journal stream earlier r seq) (t : Release x act) (wide : Bool)
    (needed : wide = true → retry x.input = true) :
    SequencedBefore (x.lock.stamp (.load wide)) x.lock.acquired := by
  let n := if wide then 2 else 0
  let k := if retry x.input then 4 else 2
  have body := SplitWorkerMir.issued_of_search x.searched
  have atLoad : (x.body.steps ++ SplitWorkerMir.lockedSteps)[n]? = some (.load wide) := by
    rw [body]
    cases wide <;> cases choice : retry x.input <;> simp_all [SplitWorkerMir.issued,n]
  have atLock : (x.body.steps ++ SplitWorkerMir.lockedSteps)[k]? = some .lock := by
    rw [body]
    cases choice : retry x.input <;> simp [SplitWorkerMir.issued,k,choice]
  change SequencedBefore (x.lock.stamp (.load wide)) (x.lock.stamp .lock)
  rw [t.observed _ _ atLoad,t.observed _ _ atLock]
  refine ⟨rfl,t.clock.advances _ _ (Nat.add_lt_add_left ?_ _)⟩
  cases wide <;> cases choice : retry x.input <;> simp_all [n,k]

structure Interval (pv : Bool) (beta : Int) (initial : SplitWindow.State) (earlier : List Report)
    (seed : Write) (final : List Write) where
  current : List Write
  seeded : [seed] <+: current
  prior : current <+: final
  stores : AlphaStores pv beta initial (earlier.map Report.score) current

theorem replay_prefix (xs : List SplitWorkerMir.FieldStores.Raw)
    (ok : SplitWorkerMir.FieldStores.replay pv beta encode cells a xs = some b) : a.alpha <+: b.alpha := by
  induction xs generalizing a with
  | nil => cases ok; exact ⟨[],by simp⟩
  | cons x xs ih =>
    obtain ⟨row,_,next⟩ := Option.bind_eq_some_iff.mp ok
    exact (List.prefix_append _ row.writes.alpha.toList).trans (ih next)

/-- Extract the result of a successful prefix replay with its proved properties. -/
def prefixLog {out : Option SplitWorkerMir.FieldStores.Log} {p : SplitWorkerMir.FieldStores.Log → Prop}
    (found : ∃ mid, out = some mid ∧ p mid) : {mid // out = some mid ∧ p mid} :=
  match out with
  | none => False.elim (by obtain ⟨_,no,_⟩ := found; cases no)
  | some mid => ⟨mid,rfl,by obtain ⟨x,eq,h⟩ := found; exact (Option.some.inj eq).symm ▸ h⟩

def interval (rc : List Move.Move) {s : State w} (workers : List Worker) (out : Worker → WorkerResult)
    (ran : ∀ i ∈ workers, SplitWorkerMir.Iterator.worker (source journal stream) s.writes i.val = some (out i))
    (a : SplitWorkerMir.FieldStores.low32 aStore = seed.alpha) (b : SplitWorkerMir.FieldStores.low32 bStore = seed.best)
    (m : SplitWorkerMir.FieldStores.low8 mStore = encode seed.picked)
    (ok : IteratorFields.replay rc s.writes journal stream workers traffic pv beta encode
      (SplitWorkerMir.FieldStores.initialLog seed aStore bStore mStore) = some final)
    (x : Posted rc s journal stream earlier r (ranked workers out))
    (part : SplitWorkerMir.Iterator.lockReports out workers = earlier ++ r :: later) :
    Interval pv beta seed.state earlier aStore final.alpha := by
  let h := IteratorFields.scout_stores rc workers out ran a b m ok x part
  let mid := prefixLog h
  have facts := mid.property
  exact ⟨mid.val.alpha,replay_prefix _ facts.1,⟨_,facts.2.1.symm⟩,facts.2.2⟩

def Interval.first (p : Interval pv beta initial earlier seed final) (pre post : List Write) :
    Fin (pre ++ final ++ post).length := by
  have a := p.seeded.length_le
  have b := p.prior.length_le
  exact ⟨pre.length,by simp at a ⊢; omega⟩

def Interval.last (p : Interval pv beta initial earlier seed final) (pre post : List Write) :
    Fin (pre ++ final ++ post).length := by
  have a := p.seeded.length_le
  have b := p.prior.length_le
  exact ⟨pre.length + p.current.length - 1,by simp at a ⊢; omega⟩

theorem Interval.seed_at (p : Interval pv beta initial earlier seed final) (pre post : List Write) :
    (pre ++ final ++ post).get (p.first pre post) = seed := by
  obtain ⟨tail,rest⟩ := p.prior
  obtain ⟨extra,start⟩ := p.seeded
  have found : (pre ++ final ++ post)[pre.length]? = some seed := by simp [← rest,← start,List.append_assoc]
  exact (List.getElem?_eq_some_iff.mp found).2

theorem Interval.window (p : Interval pv beta initial earlier seed final) (pre post : List Write) :
    StoreWindow (pre ++ final ++ post) (p.first pre post) (p.last pre post) pv beta initial (earlier.map Report.score) := by
  obtain ⟨tail,rest⟩ := p.prior
  have a := p.seeded.length_le
  exact ⟨pre,p.current,tail ++ post,by simpa only [List.append_assoc] using congrArg (fun l => pre ++ l ++ post) rest.symm,rfl,
    by change pre.length + p.current.length - 1 + 1 = _; simp at a; omega,p.stores⟩

structure Load (history : History) (writes : List Write) (seed event : Event) (word : SplitWorkerMir.Word) where
  source : Fin writes.length
  wordRead : word.toInt = signed32At writes source
  published : SplitPublication.PublishedThrough history seed event
  writeRead : ∀ i : Fin writes.length, HappensBefore history (writes.get i).event event → i.val ≤ source.val
  readWrite : ∀ i : Fin writes.length, HappensBefore history event (writes.get i).event → source.val < i.val

def Load.read (got : Load history writes seed event word) : Read writes := ⟨event,got.source,false⟩

def Load.probe {acquired : Event} (p : Interval pv beta initial earlier seed final) (pre post : List Write)
    (got : Load history (pre ++ final ++ post) seed.event event word)
    (before : SequencedBefore event acquired)
    (next : ∀ i : Fin (pre ++ final ++ post).length, i.val = (p.last pre post).val + 1 →
      HappensBefore history acquired ((pre ++ final ++ post).get i).event) :
    AlphaRead history got.read (p.first pre post) (p.last pre post) :=
  ⟨by rw [p.seed_at]; exact got.published,acquired,before,next,got.writeRead,got.readWrite⟩

structure Receipt (history : History) (pv : Bool) (beta : Int) (initial : SplitWindow.State)
    (earlier : List Report) (report : Report) (i : Input) (scoutEvent wideEvent acquired : Event) where
  reads : ReportReads history pv beta initial earlier report
  scoutStamp : reads.scoutLoad.event = scoutEvent
  alphaWord : reads.alpha = i.alpha.toInt
  scoutWord : reads.scout = i.scout.toInt
  wideStamp : retry i = true → reads.wideLoad.event = wideEvent
  freshWord : retry i = true → reads.fresh = i.fresh.toInt
  wideWord : reads.wide = i.wide.toInt
  probeLock : reads.probe.acquire = acquired
  wideLock : ∀ needed, (reads.reread needed).acquire = acquired

def both (i : Input) (b : Searched) (hit : SplitWorkerMir.search i = some b) (safe : Safe i)
    (first last : Fin writes.length) (scout wide : Read writes)
    (wa : i.alpha.toInt = signed32At writes scout.source) (wf : i.fresh.toInt = signed32At writes wide.source)
    (probe : AlphaRead history scout first last)
    (stores : StoreWindow writes first last i.pv i.beta.toInt initial (earlier.map Report.score))
    (again : retry i = true → AlphaRead history wide first last)
    (lock : probe.acquire = acquired) (same : ∀ h, (again h).acquire = acquired) :
    Receipt history i.pv i.beta.toInt initial earlier (SplitWorkerMir.selected i b) i scout.event wide.event acquired := by
  let reads := SplitWorkerMir.readFromWorker i b hit safe writes first last scout wide wa wf probe stores again
  have needed (h : SplitWindow.reSearch i.pv (signed32At writes scout.source) i.scout.toInt = true) : retry i = true := by
    rw [SplitWorkerMir.retry_eq safe,wa]; exact h
  let result : ReportReads history i.pv i.beta.toInt initial earlier (SplitWorkerMir.selected i b) :=
    {reads with reread := fun h => again (needed h)}
  exact ⟨result,rfl,wa.symm,rfl,fun _ => rfl,fun _ => wf.symm,rfl,lock,fun h => same (needed h)⟩

def shortInput (i : Input) : Input := {i with fresh := i.alpha}

theorem short_hit (i : Input) (b : Searched) (hit : SplitWorkerMir.search i = some b) (stop : retry i = false) :
    SplitWorkerMir.search (shortInput i) = some b := by
  rw [SplitWorkerMir.search_eq] at hit ⊢
  have choice : retry (shortInput i) = false := stop
  split at hit <;> try contradiction
  rename_i guard
  change (if i.depth = 0 ∨ (i.pv = true ∧ i.index = 0) then none else some (SplitWorkerMir.issued (shortInput i))) = some b
  rw [ite_eq_right guard]
  have eq : SplitWorkerMir.scoutQuery (shortInput i) = SplitWorkerMir.scoutQuery i := rfl
  simp only [SplitWorkerMir.issued,choice,Bool.false_eq_true,ite_false,eq]
  simp only [SplitWorkerMir.issued,stop,Bool.false_eq_true,ite_false] at hit
  exact hit

theorem short_safe (i : Input) (safe : Safe i) (no : retry i = false) : Safe (shortInput i) :=
  ⟨safe.alphaLow,safe.alphaHigh,safe.scout,fun h => by change retry i = true at h; simp [no] at h,
    fun h => by change retry i = true at h; simp [no] at h,fun h => by change retry i = true at h; simp [no] at h⟩

def reads (p : Interval pv beta initial earlier seed final) (pre post : List Write)
    (x : Posted rc s journal stream earlier r seq) (t : Release x act) (safe : Safe x.input)
    (tag : x.input.pv = pv) (upper : x.input.beta.toInt = beta)
    (scout : Load history (pre ++ final ++ post) seed.event x.lock.scoutLoad x.input.alpha)
    (wide : retry x.input = true → Load history (pre ++ final ++ post) seed.event x.lock.wideLoad x.input.fresh)
    (next : ∀ i : Fin (pre ++ final ++ post).length, i.val = (p.last pre post).val + 1 →
      HappensBefore history x.lock.acquired ((pre ++ final ++ post).get i).event) :
    Receipt history pv beta initial earlier r x.input x.lock.scoutLoad x.lock.wideLoad x.lock.acquired := by
  let probe := scout.probe p pre post (t.before x false (by simp)) next
  have selected : SplitWorkerMir.selected x.input x.body = r := by
    have emitted := x.credited
    simpa only [SplitWorkerMir.recordLocked_eq,x.running,Bool.false_eq_true,ite_false,Option.some.injEq] using emitted
  have build : Receipt history x.input.pv x.input.beta.toInt initial earlier (SplitWorkerMir.selected x.input x.body)
      x.input x.lock.scoutLoad x.lock.wideLoad x.lock.acquired := by
    have stores : StoreWindow _ (p.first pre post) (p.last pre post) x.input.pv x.input.beta.toInt initial
        (earlier.map Report.score) := by rw [tag,upper]; exact p.window pre post
    cases ret : retry x.input with
    | true =>
      let fresh := wide ret
      let load := fresh.probe p pre post (t.before x true (fun _ => ret)) next
      exact both x.input x.body x.searched safe _ _ scout.read fresh.read scout.wordRead fresh.wordRead
        probe stores (fun _ => load) rfl (fun _ => rfl)
    | false =>
      let i := shortInput x.input
      have no : retry i = false := ret
      let out := both i x.body (short_hit x.input x.body x.searched ret) (short_safe x.input safe ret) _ _
        scout.read scout.read scout.wordRead scout.wordRead probe stores (fun h => by simp [no] at h)
        rfl (fun h => by simp [no] at h)
      exact ⟨out.reads,rfl,out.alphaWord,out.scoutWord,by simp [ret],by simp [ret],out.wideWord,out.probeLock,out.wideLock⟩
  exact (tag ▸ upper ▸ selected ▸ build : Receipt history pv beta initial earlier r x.input _ _ _)

structure Loads (x : Posted rc s journal stream earlier r seq) (history : History) (writes : List Write) (seed : Event) where
  safe : Safe x.input
  scout : Load history writes seed x.lock.scoutLoad x.input.alpha
  wide : retry x.input = true → Load history writes seed x.lock.wideLoad x.input.fresh

structure Modifications (p : Interval pv beta initial earlier seed final) (x : Posted rc s journal stream earlier r seq)
    (history : History) (pre post : List Write) extends Loads x history (pre ++ final ++ post) seed.event where
  next : ∀ i : Fin (pre ++ final ++ post).length, i.val = (p.last pre post).val + 1 →
    HappensBefore history x.lock.acquired ((pre ++ final ++ post).get i).event

structure Occurrence (history : History) (act : Activation mask) (pv : Bool) (beta : Int)
    (initial : SplitWindow.State) (work : Worker → SplitReturn.Work) (x : Posted rc s journal stream earlier r seq) where
  receipt : Receipt history pv beta initial earlier r x.input x.lock.scoutLoad x.lock.wideLoad x.lock.acquired
  publication : { pub : SplitReturn.Observation.ReportPublication act //
    r ∈ (work pub.worker).reports ∧ pub.acquired = receipt.reads.probe.acquire ∧
      ∀ needed, pub.acquired = (receipt.reads.reread needed).acquire }
  worker : publication.val.worker.val = x.lock.worker
  acquired : publication.val.acquired = x.lock.acquired
  checked : publication.val.checked = x.lock.polled
  published : publication.val.published = x.lock.published

/-- A report occurrence for the work function of Iterator.dispatches. Both read and release
    events use the lock row at its physical journal rank. -/
noncomputable def occurrence (rc : List Move.Move) {s : State w}
    (run : MoveList.ConcurrentTrace.Execution rc.length start s) (fits : rc.length < 2 ^ w)
    (workers : List Worker) (out : Worker → WorkerResult)
    (covers : ∀ op ∈ s.writes, op.worker ∈ workers.map Fin.val)
    (ran : ∀ i ∈ workers, SplitWorkerMir.Iterator.worker (source journal stream) s.writes i.val = some (out i))
    (a : SplitWorkerMir.FieldStores.low32 aStore = seed.alpha) (b : SplitWorkerMir.FieldStores.low32 bStore = seed.best)
    (m : SplitWorkerMir.FieldStores.low8 mStore = encode seed.picked)
    (ok : IteratorFields.replay rc s.writes journal stream workers traffic pv beta encode
      (SplitWorkerMir.FieldStores.initialLog seed aStore bStore mStore) = some final)
    (x : Posted rc s journal stream earlier r (ranked workers out))
    (part : SplitWorkerMir.Iterator.lockReports out workers = earlier ++ r :: later) (release : Release x act)
    (pre post : List Write)
    (raw : Modifications (interval rc workers out ran a b m ok x part) x history pre post) :
    Occurrence history act pv beta seed.state (fun i => (out i).completed) x := by
  let slice := interval rc workers out ran a b m ok x part
  have bounds := IteratorFields.posted_window covers ran ok x
  let record := reads slice pre post x release raw.safe bounds.1 bounds.2 raw.scout raw.wide raw.next
  have stamps := release.stamps
  have belongs := in_work rc run fits workers out ran x release.worker release.actor
  exact ⟨record,⟨release.publication,stamps.1 ▸ belongs,stamps.2.1.trans record.probeLock.symm,
      fun needed => stamps.2.1.trans (record.wideLock needed).symm⟩,
    (congrArg Fin.val stamps.1).trans release.actor,stamps.2.1,stamps.2.2.1,stamps.2.2.2⟩

def lockOps (r : IteratorFrames.Observation) : List SpinLock.Operation :=
  [.acquire r.acquired,.release r.published]

private theorem cut_at (xs : List α) (found : xs[n]? = some a) :
    xs = xs.take n ++ a :: xs.drop (n+1) := by
  obtain ⟨inside,same⟩ := List.getElem?_eq_some_iff.mp found
  rw [← same,List.getElem_cons_drop inside,List.take_append_drop]

/-- Two journal ordinals give the release/acquire order for the next alpha store, even
    with intervening sections that issued no stores. -/
theorem unlock_before_later (x : Posted rc s journal stream earlier r seq) (trace : SpinLock.Trace history)
    (mutex : (journal.flatMap lockOps).Sublist trace.operations)
    {later : IteratorFrames.Observation} (newer : x.rank < k) (row : journal[k]? = some later) :
    HappensBefore history x.lock.published later.acquired := by
  let tail := journal.drop (x.rank+1)
  let j := k - (x.rank+1)
  have index : x.rank+1+j = k := by dsimp [j]; omega
  have cell : tail[j]? = some later := by simpa [tail,index] using row
  have two : journal = journal.take x.rank ++ x.lock :: (tail.take j ++ later :: tail.drop (j+1)) := by
    calc
      journal = journal.take x.rank ++ x.lock :: tail := cut_at _ x.atLock
      _ = _ := congrArg (fun xs => journal.take x.rank ++ x.lock :: xs) (cut_at _ cell)
  have pair : [SpinLock.Operation.release x.lock.published, .acquire later.acquired].Sublist trace.operations := by
    apply List.Sublist.trans _ mutex
    conv => rhs; rw [two, List.flatMap_append]
    apply List.Sublist.trans _ (List.sublist_append_right ..)
    simp only [List.flatMap_cons, lockOps, List.flatMap_append, List.cons_append, List.nil_append]
    apply List.Sublist.cons
    apply List.Sublist.cons_cons
    simp
  obtain ⟨a,b,whole,unlocked,acquired⟩ := List.cons_sublist_iff.mp pair
  obtain ⟨pre,between,first⟩ := List.append_of_mem unlocked
  obtain ⟨gap,post,last⟩ := List.append_of_mem (List.singleton_sublist.mp acquired)
  apply SpinLock.handoffContract_proven history trace _ _ pre (between ++ gap) post
  simp [whole,first,last,List.append_assoc]

end NeuralReversi.PvsParallel.IteratorReports

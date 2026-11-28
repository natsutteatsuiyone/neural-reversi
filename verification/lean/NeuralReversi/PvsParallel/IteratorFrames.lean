module
public import NeuralReversi.PvsParallel.NativeWorkerFeed
public import NeuralReversi.PvsParallel.SplitWorkerMir.Iterator

@[expose] public section

namespace NeuralReversi.PvsParallel.IteratorFrames

open MoveList.ConcurrentTrace (State Write Execution forWorker)
open SplitWorkerMir (Input Locked Ticket Searched FrameFeed)
open SplitWorkerMir.Iterator (LockedCall WorkerResult)
open Threading.SplitResult (Report)
open Threading.AtomicWord (Event)
open Threading.SplitCompletion (Worker)
open PvsBoardTree (Position GeneratedSquare)
open NativeWorkerFeed (Prefix received labels)
open NativeFrames (Record Capture)

private theorem held_snapshot (run : MoveList.ConcurrentTrace.Execution len start s) (member : op ∈ s.writes) :
    ∃ t, MoveList.ConcurrentTrace.Execution len start t ∧ t.localAt op.worker = .held op := by
  induction run with
  | init => contradiction
  | load _ _ _ _ ih | failed _ _ _ _ ih | decline _ _ _ ih | returnedSome _ _ ih | returnedNone _ _ ih =>
    exact ih member
  | @success before worker opening wanted run active proposed ih =>
    rcases List.mem_append.mp member with old | newest
    · exact ih old
    · simp only [List.mem_singleton] at newest
      subst op
      exact ⟨_,MoveList.ConcurrentTrace.Execution.success run active proposed,by simp [MoveList.ConcurrentTrace.replace]⟩

theorem issued_move (rc : List Move.Move) {op : Write w} (run : MoveList.ConcurrentTrace.Execution rc.length start s)
    (member : op ∈ s.writes) (fits : rc.length < 2 ^ w) :
    ∃ mv, rc[op.old.toNat]? = some mv ∧
      NativeWorkerFeed.ticket rc op.old = some (mv,op.stored.toNat) ∧ op.stored.toNat = op.old.toNat + 1 := by
  obtain ⟨before,ran,held⟩ := held_snapshot run member
  obtain ⟨_,mv,atPos,_,one⟩ := ran.move_at_return rc held fits
  have write := MoveList.ConcurrentTrace.issued _ run.replayed member
  exact ⟨mv,atPos,by simp [NativeWorkerFeed.ticket,MoveList.Concurrent.next,write,atPos],one⟩

/-- The lock log is in the physical critical-section order. Each entry names the iterator
    opening for this worker and the observed acquire, stop check and publication. -/
structure Observation where
  worker : Nat
  opening : Nat
  dispatch : Nat
  nextClaim : Option Nat
  scoutOpening : Nat
  wideOpening : Nat
  scoutReturn : Event
  wideReturn : Event
  scoutLoad : Event
  wideLoad : Event
  acquired : Event
  polled : Event
  published : Event
  task : NativeWorkerFeed.Task
  alpha : SplitWorkerMir.Word
  fresh : SplitWorkerMir.Word
  locked : Locked

def Observation.ends (r : Observation) (wide : Bool) := if wide then r.wideReturn else r.scoutReturn

def Observation.stamp (r : Observation) : SplitWorkerMir.Step → Event
  | .load false => r.scoutLoad
  | .load true => r.wideLoad
  | .returned q => r.ends q.pv
  | .lock => r.acquired
  | .poll => r.polled
  | .unlock => r.published

def Observation.prefix (r : Observation) : Prefix :=
  ⟨r.worker,r.dispatch,r.opening,r.nextClaim,r.scoutOpening,r.wideOpening,r.stamp⟩

def findLock (journal : List Observation) (worker opening : Nat) : Option (Nat × Observation) :=
  (journal.zipIdx.find? (fun cell => decide (cell.1.worker = worker ∧ cell.1.opening = opening))).map Prod.swap

theorem lock_at (found : findLock journal worker opening = some (k,r)) :
    journal[k]? = some r ∧ r.worker = worker ∧ r.opening = opening := by
  unfold findLock at found
  obtain ⟨⟨obs,n⟩,actual,eq⟩ := Option.map_eq_some_iff.mp found
  cases eq
  have member := List.mem_of_find?_eq_some actual
  have tag := List.find?_some actual
  exact ⟨by simpa only [List.mk_mem_zipIdx_iff_getElem?] using member,by simpa using tag⟩

def source (journal : List Observation) (stream : List Record) : SplitWorkerMir.Iterator.Feed :=
  fun worker opening => do
    let (k,r) ← findLock journal worker opening
    let results ← NativeWorkerFeed.returns worker stream
    if NativeWorkerFeed.completeControl r.prefix r.task results r.alpha
        (fun wide => r.stamp (.load wide)) r.ends r.acquired r.locked.stopped then
      return ⟨⟨received r.task r.prefix results (0#32) r.alpha r.fresh,r.locked⟩,k⟩
    else none

theorem acquire_control {op : Write w}
    (got : SplitWorkerMir.Iterator.acquired (source journal stream) op = some call) :
    ∃ r results, findLock journal op.worker op.opening = some (call.order,r) ∧
      NativeWorkerFeed.returns op.worker stream = some results ∧
      call.ticket = ⟨received r.task r.prefix results op.old r.alpha r.fresh,r.locked⟩ ∧
      NativeWorkerFeed.completeControl r.prefix r.task results r.alpha
        (fun wide => r.stamp (.load wide)) r.ends r.acquired r.locked.stopped = true := by
  unfold SplitWorkerMir.Iterator.acquired at got
  cases cell : findLock journal op.worker op.opening with
  | none => simp [source,cell] at got
  | some pair =>
    cases seen : NativeWorkerFeed.returns op.worker stream with
    | none => simp [source,cell,seen] at got
    | some results =>
      cases control : NativeWorkerFeed.completeControl pair.2.prefix pair.2.task results pair.2.alpha
          (fun wide => pair.2.stamp (.load wide)) pair.2.ends pair.2.acquired pair.2.locked.stopped <;>
        simp [source,cell,seen,control] at got
      cases got
      exact ⟨pair.2,results,rfl,rfl,rfl,control⟩

private theorem published_by (calls : List LockedCall) (ran : SplitWorkerMir.Iterator.retire calls = some result)
    (pub : (k,r) ∈ result.published) :
    ∃ c ∈ calls, c.order = k ∧ ∃ b, SplitWorkerMir.search c.ticket.input = some b ∧
      SplitWorkerMir.recordLocked c.ticket.input b c.ticket.locked = some r := by
  induction calls generalizing result with
  | nil => cases ran; contradiction
  | cons head tail ih =>
    unfold SplitWorkerMir.Iterator.retire at ran
    obtain ⟨b,body,remaining⟩ := Option.bind_eq_some_iff.mp ran
    dsimp only at remaining
    cases ex : (SplitWorkerMir.commit head.ticket.input head.ticket.locked b.chosen).exit with
    | stopped =>
      cases tail <;> simp [ex] at remaining
      subst result
      contradiction
    | cutoff =>
      cases tail <;> simp [ex] at remaining
      subst result
      simp only [List.mem_singleton,Prod.mk.injEq] at pub
      exact ⟨head,by simp,pub.1.symm,b,body,by simp [SplitWorkerMir.recordLocked,ex,pub.2]⟩
    | again =>
      rw [ex] at remaining
      obtain ⟨next,ret,out⟩ := Option.bind_eq_some_iff.mp remaining
      cases out
      rcases List.mem_cons.mp pub with eq | late
      · cases eq
        exact ⟨head,by simp,rfl,b,body,by simp [SplitWorkerMir.recordLocked,ex]⟩
      · obtain ⟨c,mem,origin⟩ := ih ret late
        exact ⟨c,List.mem_cons_of_mem _ mem,origin⟩

private theorem map_member (f : A → Option B) (xs : List A) (ran : xs.mapM f = some ys) (mem : y ∈ ys) :
    ∃ x ∈ xs, f x = some y := by
  induction xs generalizing ys with
  | nil => cases ran; contradiction
  | cons x xs ih =>
    rw [List.mapM_cons] at ran
    obtain ⟨v,one,next⟩ := Option.bind_eq_some_iff.mp ran
    obtain ⟨vs,more,endOf⟩ := Option.bind_eq_some_iff.mp next
    cases endOf
    rcases List.mem_cons.mp mem with eq | tail
    · exact ⟨x,by simp,eq ▸ one⟩
    · obtain ⟨other,member,returned⟩ := ih more tail
      exact ⟨other,List.mem_cons_of_mem _ member,returned⟩



def ranked (workers : List Worker) (out : Worker → WorkerResult) :=
  (workers.flatMap (fun i => (out i).published)).mergeSort (fun a b => decide (a.1 ≤ b.1))

structure Posted (rc : List Move.Move) (s : State w) (journal : List Observation) (stream : List Record)
    (earlier : List Report) (report : Report) (sequence : List (Nat × Report)) where
  op : Write w
  issued : op ∈ s.writes
  rank : Nat
  sorted : sequence[earlier.length]? = some (rank,report)
  lock : Observation
  foundLock : findLock journal op.worker op.opening = some (rank,lock)
  returned : List NativeWorkerFeed.Return
  returns : NativeWorkerFeed.returns op.worker stream = some returned
  control : NativeWorkerFeed.completeControl lock.prefix lock.task returned lock.alpha
    (fun wide => lock.stamp (.load wide)) lock.ends lock.acquired lock.locked.stopped = true
  move : Move.Move
  cell : rc[op.old.toNat]? = some move
  ticket : NativeWorkerFeed.ticket rc op.old = some (move,op.stored.toNat)
  ordinal : op.stored.toNat = op.old.toNat+1
  body : Searched
  searched : SplitWorkerMir.search (received lock.task lock.prefix returned op.old lock.alpha lock.fresh) = some body
  credited : SplitWorkerMir.recordLocked (received lock.task lock.prefix returned op.old lock.alpha lock.fresh) body lock.locked = some report

theorem Posted.atLock (x : Posted rc s journal stream earlier r seq) : journal[x.rank]? = some x.lock :=
  (lock_at x.foundLock).1

theorem Posted.worker (x : Posted rc s journal stream earlier r seq) : x.lock.worker = x.op.worker :=
  (lock_at x.foundLock).2.1

theorem Posted.opening (x : Posted rc s journal stream earlier r seq) : x.lock.opening = x.op.opening :=
  (lock_at x.foundLock).2.2

theorem posted (rc : List Move.Move) {s : State w} (run : MoveList.ConcurrentTrace.Execution rc.length start s)
    (fits : rc.length < 2 ^ w) (workers : List Worker) (out : Worker → WorkerResult)
    (runs : ∀ i ∈ workers, SplitWorkerMir.Iterator.worker (source journal stream) s.writes i.val = some (out i))
    (shape : SplitWorkerMir.Iterator.lockReports out workers = earlier ++ r :: later) :
    Nonempty (Posted rc s journal stream earlier r (ranked workers out)) := by
  have idx : (ranked workers out)[earlier.length]?.map Prod.snd = some r := by
    rw [← List.getElem?_map]
    change (SplitWorkerMir.Iterator.lockReports out workers)[earlier.length]? = some r
    simp [shape]
  obtain ⟨⟨k,value⟩,found,eq⟩ := Option.map_eq_some_iff.mp idx
  subst r
  have position := List.mem_of_getElem? found
  have members := (List.mergeSort_perm _ _).mem_iff.mp position
  obtain ⟨i,owns,pub⟩ := List.mem_flatMap.mp members
  obtain ⟨calls,mapped,retired⟩ := Option.bind_eq_some_iff.mp (runs i owns)
  obtain ⟨call,member,same,body,searched,credit⟩ := published_by _ retired pub
  obtain ⟨op,claimed,acquired⟩ := map_member _ _ mapped member
  obtain ⟨issued,_⟩ := List.mem_filter.mp claimed
  obtain ⟨mv,cell,ticket,ordinal⟩ := issued_move rc run issued fits
  obtain ⟨lock,returns,sourceLock,observed,values,control⟩ := acquire_control acquired
  rw [values] at searched credit
  exact ⟨⟨op,issued,k,found,lock,same ▸ sourceLock,returns,observed,control,mv,cell,ticket,
    ordinal,body,searched,credit⟩⟩

abbrev Posted.input {s : State w} (x : Posted rc s journal stream earlier report sequence) : SplitWorkerMir.Input :=
  received x.lock.task x.lock.prefix x.returned x.op.old x.lock.alpha x.lock.fresh

theorem Posted.acquired (x : Posted rc s journal stream earlier r seq) :
    SplitWorkerMir.Iterator.acquired (source journal stream) x.op =
      some ⟨⟨x.input,x.lock.locked⟩,x.rank⟩ := by
  simp [SplitWorkerMir.Iterator.acquired, source, x.foundLock, x.returns, x.control, input, NativeWorkerFeed.received]

theorem Posted.index (x : Posted rc s journal stream earlier r seq) : r.index = x.op.old.toNat := by
  have c := x.credited
  unfold SplitWorkerMir.recordLocked at c
  split at c
  · contradiction
  · exact congrArg Report.index (Option.some.inj c).symm

theorem find_claim (key : α → Nat) (xs : List α) (unique : (xs.map key).Nodup)
    (member : x ∈ xs) : xs.find? (fun y => key y == key x) = some x := by
  induction xs with
  | nil => contradiction
  | cons y ys ih =>
    obtain ⟨fresh,distinct⟩ := List.nodup_cons.mp unique
    rcases List.mem_cons.mp member with rfl | later
    · simp
    · have other : key y ≠ key x := fun same => fresh (same ▸ List.mem_map_of_mem later)
      simp [other,ih distinct later]

structure Decoded (w : Nat) where
  op : Write w
  rank : Nat
  lock : Observation
  returned : List NativeWorkerFeed.Return
  move : Move.Move
  body : Searched

def decode (rc : List Move.Move) (s : State w) (journal : List Observation) (stream : List Record)
    (ordinal index : Nat) (seq : List (Nat × Report)) : Option (Decoded w) := do
  let (rank,_) ← seq[ordinal]?
  let op ← s.writes.find? (fun op => op.old.toNat == index)
  let lock ← journal[rank]?
  let returned ← NativeWorkerFeed.returns op.worker stream
  let move ← rc[op.old.toNat]?
  let body ← SplitWorkerMir.search (received lock.task lock.prefix returned op.old lock.alpha lock.fresh)
  return ⟨op,rank,lock,returned,move,body⟩

/-- The posted ordinal names the physical lock row. A claim's old iterator word uniquely names
    its successful write, even when several workers returned the same score. -/
def recover {s : State w} (distinct : (s.writes.map (fun op => op.old.toNat)).Nodup)
    (existsPost : Nonempty (Posted rc s journal stream earlier r seq)) : Posted rc s journal stream earlier r seq := by
  let parsed := decode rc s journal stream earlier.length r.index seq
  have decoded (x : Posted rc s journal stream earlier r seq) :
      parsed = some ⟨x.op,x.rank,x.lock,x.returned,x.move,x.body⟩ := by
    have found := find_claim (fun op : Write w => op.old.toNat) s.writes distinct x.issued
    simp [parsed,decode,x.sorted,x.index,found,x.atLock,x.returns,x.cell,x.searched]
  let d := parsed.get (by obtain ⟨x⟩ := existsPost; simp [decoded x])
  have same (x : Posted rc s journal stream earlier r seq) :
      d = ⟨x.op,x.rank,x.lock,x.returned,x.move,x.body⟩ :=
    Option.some.inj ((Option.some_get _).trans (decoded x))
  refine ⟨d.op,?_,d.rank,?_,d.lock,?_,d.returned,?_,?_,d.move,?_,?_,?_,d.body,?_,?_⟩ <;>
    obtain ⟨x⟩ := existsPost <;> rw [same x] <;> first
    | exact x.issued | exact x.sorted | exact x.foundLock | exact x.returns | exact x.control
    | exact x.cell | exact x.ticket | exact x.ordinal | exact x.searched | exact x.credited

def Posted.frameWords (x : Posted rc s journal stream earlier r seq) (mc : Bitboard.Moves.X86Config)
    (fc : Cutoffs.FlipConfig) (b : Position) (order : List (GeneratedSquare mc b.val))
    (rows : List NativeFrames.Row) (clock : Prefix) : Bool :=
  let steps := x.body.steps ++ SplitWorkerMir.lockedSteps
  NativeWorkerFeed.frozen mc fc b rc order && NativeWorkerFeed.ordered clock.stops steps &&
    steps.all (NativeWorkerFeed.stepOk mc fc b order rc x.op.old rows x.returned clock)

theorem Posted.check_from_return (x : Posted rc s journal stream earlier r seq)
    (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (order : List (GeneratedSquare mc b.val))
    (rows : List NativeFrames.Row) (clock : Prefix) (actor : clock.worker = x.op.worker) :
    NativeWorkerFeed.check mc fc b rc order x.op.old rows stream clock x.input
      (x.body.steps ++ SplitWorkerMir.lockedSteps).length = x.frameWords mc fc b order rows clock := by
  unfold NativeWorkerFeed.check frameWords
  have idx : x.input.index = x.op.old.toNat := rfl
  rw [x.searched,actor,x.returns,x.ticket]
  simp only [idx,x.ordinal,and_self,decide_true,Bool.and_true,List.take_of_length_le (Nat.le_refl _)]

def Posted.checked {pool : EndgameSearch.CacheHistory.Pool} {mc fc b rc order} {s : State w}
    {earlier r later} {journal stream seq}
    (x : Posted rc s journal stream earlier r seq)
    (split : SplitExecution (labels mc fc b order).length pv alpha beta score)
    (same : split.reports = earlier ++ r :: later) (cap : Capture pool) : Bool :=
  let raw := split.reportReads earlier r later same
  let pub := (split.reportPublications earlier r later same).val
  let clock := NativeWorkerFeed.forReport split earlier r later same x.op.worker x.lock.dispatch x.op.opening
    x.lock.nextClaim x.lock.scoutOpening x.lock.wideOpening x.lock.ends
  decide (x.frameWords mc fc b order cap.rows clock = true ∧
      x.lock.worker = pub.worker.val ∧ x.lock.acquired = pub.acquired ∧ x.lock.polled = pub.checked ∧
      x.lock.published = pub.published ∧ x.lock.scoutLoad = raw.scoutLoad.event ∧
      x.input.pv = pv ∧ x.input.beta.toInt = beta ∧ x.input.alpha.toInt = raw.alpha ∧ x.input.scout.toInt = raw.scout ∧
      -2147483648 < x.input.alpha.toInt ∧ x.input.alpha.toInt < 2147483647 ∧ -2147483648 < x.input.scout.toInt ∧
      (SplitWorkerMir.retry x.input = true → x.lock.wideLoad = raw.wideLoad.event ∧
        x.input.fresh.toInt = raw.fresh ∧ x.input.wide.toInt = raw.wide ∧ -2147483648 < x.input.beta.toInt ∧
        -2147483648 < x.input.fresh.toInt ∧ -2147483648 < x.input.wide.toInt))

def Posted.segment {pool : EndgameSearch.CacheHistory.Pool} {mc fc b rc order} {s : State w}
    {earlier r later} {journal stream seq}
    (x : Posted rc s journal stream earlier r seq)
    (split : SplitExecution (labels mc fc b order).length pv alpha beta score)
    (same : split.reports = earlier ++ r :: later)
    (cap : Capture pool) (identify before sequenced) :
    Option (SplitWorkerMir.Segment (moves := labels mc fc b order) split earlier r later same identify
      (NativeFrames.logOf mc fc cap.rows cap.table identify before sequenced).events) :=
  let clock := NativeWorkerFeed.forReport split earlier r later same x.op.worker x.lock.dispatch x.op.opening
    x.lock.nextClaim x.lock.scoutOpening x.lock.wideOpening x.lock.ends
  let budget := (x.body.steps ++ SplitWorkerMir.lockedSteps).length
  if checked : x.checked split same cap = true then
    some (by
      have h := of_decide_eq_true checked
      obtain ⟨accepted,_,_,_,_,_,node,bound,load,scout,lo,hi,safeScout,wide⟩ := h
      have sourceCheck := (x.check_from_return mc fc b order cap.rows clock rfl).trans accepted
      have children := NativeWorkerFeed.feed sourceCheck x.body x.searched cap.table identify before sequenced
      have all : FrameFeed (labels mc fc b order) identify (NativeFrames.logOf mc fc cap.rows cap.table identify before sequenced).events
          clock.stops (x.body.steps ++ SplitWorkerMir.lockedSteps) := by
        simpa only [budget,List.take_of_length_le (Nat.le_refl _)] using children
      exact ⟨x.input,x.body,x.lock.locked,x.searched,x.credited,
        ⟨lo,hi,safeScout,fun t => (wide t).2.2.2.1,fun t => (wide t).2.2.2.2.1,fun t => (wide t).2.2.2.2.2⟩,
        x.lock.ends,all,node,bound,load,scout,fun t => (wide t).2.1,fun t => (wide t).2.2.1⟩)
  else none

theorem Posted.running (x : Posted rc s journal stream earlier r seq) : x.lock.locked.stopped = false := by
  cases status : x.lock.locked.stopped with
  | false => rfl
  | true =>
    have paid := x.credited
    simp [SplitWorkerMir.recordLocked_eq,status] at paid

theorem Posted.stamps {pool : EndgameSearch.CacheHistory.Pool} {mc fc b order} {s : State w}
    {earlier r later} (x : Posted rc s journal stream earlier r seq)
    (split : SplitExecution (labels mc fc b order).length pv alpha beta score)
    (same : split.reports = earlier ++ r :: later) (cap : Capture pool) (identify before sequenced)
    {joined : SplitWorkerMir.Segment (moves := labels mc fc b order) split earlier r later same identify
      (NativeFrames.logOf mc fc cap.rows cap.table identify before sequenced).events}
    (ok : x.segment split same cap identify before sequenced = some joined) :
    let pub := (split.reportPublications earlier r later same).val
    journal[x.rank]? = some x.lock ∧ x.lock.worker = pub.worker.val ∧ x.lock.acquired = pub.acquired ∧
      x.lock.polled = pub.checked ∧ x.lock.published = pub.published ∧ joined.input.index = x.op.old.toNat ∧
      joined.locked = x.lock.locked := by
  unfold Posted.segment at ok
  dsimp only at ok
  split at ok
  next accepted =>
    have good := of_decide_eq_true accepted
    cases ok
    exact ⟨x.atLock,good.2.1,good.2.2.1,good.2.2.2.1,good.2.2.2.2.1,rfl,rfl⟩
  next => contradiction

/-- Independently supplied call streams may re-use a ticket opening, although the native
    physical clock cannot. Check both domains before joining the shared lock and call logs. -/
def unique (writes : List (Write w)) (journal : List Observation) : Bool :=
  decide ((writes.map (fun op => (op.worker,op.opening))).Nodup ∧
    (journal.map (fun lock => (lock.worker,lock.opening))).Nodup)

def Posted.fromNative (pool : EndgameSearch.CacheHistory.Pool) {mc fc b order} {s : State w}
    {earlier r later} (x : Posted rc s journal stream earlier r seq)
    (inputs : Nat → Nat → Option NativeFrames.Args)
    (split : SplitExecution (labels mc fc b order).length pv alpha beta score)
    (same : split.reports = earlier ++ r :: later) (identify before sequenced) :
    Option ((cap : Capture pool) × SplitWorkerMir.Segment (moves := labels mc fc b order)
      split earlier r later same identify (NativeFrames.logOf mc fc cap.rows cap.table identify before sequenced).events) := do
  if !unique s.writes journal then none else
    let cap ← NativeFrames.decode pool inputs stream
    let joined ← x.segment split same cap identify before sequenced
    return ⟨cap,joined⟩

theorem Posted.native_stamps {pool : EndgameSearch.CacheHistory.Pool} {mc fc b order} {s : State w}
    {earlier r later} (x : Posted rc s journal stream earlier r seq) {inputs}
    (split : SplitExecution (labels mc fc b order).length pv alpha beta score)
    (same : split.reports = earlier ++ r :: later) (identify before sequenced)
    {cap : Capture pool} {joined : SplitWorkerMir.Segment (moves := labels mc fc b order) split earlier r later same identify
      (NativeFrames.logOf mc fc cap.rows cap.table identify before sequenced).events}
    (ok : x.fromNative pool inputs split same identify before sequenced = some ⟨cap,joined⟩) :
    NativeFrames.decode pool inputs stream = some cap ∧ journal[x.rank]? = some x.lock ∧
      x.lock.acquired = (split.reportPublications earlier r later same).val.acquired ∧
      x.lock.published = (split.reportPublications earlier r later same).val.published ∧
      joined.input.index = x.op.old.toNat ∧ joined.locked = x.lock.locked := by
  unfold Posted.fromNative at ok
  split at ok
  · contradiction
  · obtain ⟨c,decoded,afterCapture⟩ := Option.bind_eq_some_iff.mp ok
    obtain ⟨segment,assembled,result⟩ := Option.bind_eq_some_iff.mp afterCapture
    cases result
    have clocks := x.stamps split same cap identify before sequenced assembled
    exact ⟨decoded,clocks.1,clocks.2.2.1,clocks.2.2.2.2.1,clocks.2.2.2.2.2⟩

theorem reportCalls {pool : EndgameSearch.CacheHistory.Pool} {mc fc b rc order} {s : State w}
    {journal stream} (inputs : Nat → Nat → Option NativeFrames.Args) (cap : Capture pool)
    (capture : NativeFrames.decode pool inputs stream = some cap)
    (run : MoveList.ConcurrentTrace.Execution rc.length start s) (fits : rc.length < 2 ^ w)
    (workers : List Worker) (out : Worker → WorkerResult)
    (completed : ∀ i ∈ workers, SplitWorkerMir.Iterator.worker (source journal stream) s.writes i.val = some (out i))
    (split : SplitExecution (labels mc fc b order).length pv alpha beta score)
    (reports : split.reports = SplitWorkerMir.Iterator.lockReports out workers)
    (uncollided : unique s.writes journal = true)
    (raw : ∀ earlier r later (part : split.reports = earlier ++ r :: later)
      (x : Posted rc s journal stream earlier r (ranked workers out)), x.checked split part cap = true)
    (identify before sequenced) (consumer : Nat)
    (clock : ∀ a z, Threading.CutoffVisibility.HappensBefore split.history a z → before (identify a) (identify z))
    (after : before (identify split.result.event) consumer) :
    ReportCalls (ConcurrentReplay.UsedChild identify before
      (NativeFrames.logOf mc fc cap.rows cap.table identify before sequenced).events consumer) (labels mc fc b order) split := by
  apply SplitWorkerMir.reportCalls_of_worker_frames split identify before _ consumer clock after
  intro earlier r later same
  let x := Classical.choice (posted rc run fits workers out completed (reports.symm.trans same))
  have ok := raw earlier r later same x
  have existsNative : ∃ j, x.fromNative pool inputs split same identify before sequenced = some ⟨cap,j⟩ := by
    unfold Posted.fromNative
    simp only [uncollided,Bool.not_true,Bool.false_eq_true,↓reduceIte,capture]
    cases assembled : x.segment split same cap identify before sequenced with
    | none => unfold Posted.segment at assembled; simp [ok] at assembled
    | some j => exact ⟨j,by simp [assembled]⟩
  exact Classical.choose existsNative

end NeuralReversi.PvsParallel.IteratorFrames

import NeuralReversi.PvsParallel.Interleaving.RawSchedule
import NeuralReversi.PvsParallel.NativeStopReturns.Poll

namespace NeuralReversi.PvsParallel.Interleaving.StopSchedule

open Threading
open AtomicWord (Event Write Read SequencedBefore)
open CutoffVisibility (History FlagWrite FlagRead HappensBefore Coherent)
open SplitCompletion (RetainedThrough)
open SplitReturn.Observation (ExternalStop OwnerCheck)
open NativeStopReturns (Memory InWorker)
open Tickets (Post State Step)

structure Departed (hist : History) (atPoll wait : Event) where
  mask : List Write
  act : SplitCompletion.Activation mask
  acquired : SplitCompletion.Retirement hist mask act
  finish : Fin act.actions.length
  reset : (act.actions.get finish).release = true
  before : SequencedBefore atPoll (act.actions.get finish).event
  waited : acquired.read.event = wait

theorem Departed.keeps (d : Departed hist atPoll wait) (within : InWorker atPoll e) : RetainedThrough hist e wait := by
  rw [← d.waited]
  exact .wait d.acquired d.finish d.reset .here ⟨within.1.trans d.before.1,Nat.lt_of_le_of_lt within.2 d.before.2⟩

/-- The physical cutoff word is cleared once and all cutoff swaps in this generation write true. -/
structure Flags (h : History) (slot : Nat) where
  pre : List FlagWrite
  clear : Event
  marks : List Event
  future : List FlagWrite
  words : h.flags slot = pre ++ ⟨clear,false⟩ :: (marks.map (fun e => ⟨e,true⟩) ++ future)

def Flags.first (f : Flags h slot) : Fin (h.flags slot).length := ⟨f.pre.length,by rw [f.words]; simp⟩

private theorem Flags.cleared (f : Flags h slot) : (h.flags slot).get f.first = ⟨f.clear,false⟩ := by
  have get : (h.flags slot)[f.pre.length]? = some ⟨f.clear,false⟩ := by rw [f.words]; simp
  exact (List.getElem?_eq_some_iff.mp get).2

private theorem Flags.future_of_false (f : Flags h slot) (i : Fin (h.flags slot).length)
    (after : f.first.val < i.val) (clear : ((h.flags slot).get i).value = false) :
    (h.flags slot).get i ∈ f.future := by
  have get : (h.flags slot)[i.val]? = some ((h.flags slot).get i) := List.getElem?_eq_getElem i.isLt
  change f.pre.length < i.val at after
  have get := (congrArg (fun ws : List FlagWrite => ws[i.val]?) f.words).symm.trans get
  rw [List.getElem?_append_right (Nat.le_of_lt after)] at get
  have pos : i.val - f.pre.length = (i.val - f.pre.length - 1)+1 := by omega
  rw [pos,List.getElem?_cons_succ] at get
  rcases List.mem_append.mp (List.mem_of_getElem? get) with mark | later
  · obtain ⟨e,_,eq⟩ := List.mem_map.mp mark
    rw [←eq] at clear
    contradiction
  · exact later

/-- Replays the flag stores in actual publication order. -/
def marks (stamp : Nat → Event) (ps : List Post) : List Event :=
  (ps.filter Post.cut).map (fun p => stamp p.report.index)

private theorem marks_nil (none : ¬ ∃ p ∈ ps, p.cut = true) : marks stamp ps = [] := by
  apply List.eq_nil_iff_forall_not_mem.mpr
  intro e mem
  obtain ⟨p,inList,_⟩ := List.mem_map.mp mem
  exact none ⟨p,List.mem_filter.mp inList⟩

/-- A cached true read cannot be credited to an earlier use of this slot or a later lock holder. -/
private theorem current {r : FlagRead h.flags} (f : Flags h r.slot) (co : Coherent h)
    (recorded : r ∈ h.flagReads) (yes : r.value = true) (same : f.marks = marks stamp ps)
    (initialized : HappensBefore h f.clear r.event)
    (future : ∀ i : Fin (h.flags r.slot).length, i.val = f.pre.length+1+f.marks.length →
      HappensBefore h r.event ((h.flags r.slot).get i).event) : ∃ p ∈ ps, p.cut = true := by
  apply Classical.byContradiction
  intro no
  have noMarks := same.trans (marks_nil no)
  have lo := co.flagWriteRead r recorded f.first (f.cleared ▸ initialized)
  change f.pre.length ≤ r.source.val at lo
  have eq : r.source = f.first := by
    apply Fin.ext
    change r.source.val = f.pre.length
    by_cases bound : f.pre.length < r.source.val
    · let next : Fin (h.flags r.slot).length := ⟨f.pre.length+1,by have := r.source.isLt; omega⟩
      have edge := co.flagReadWrite r recorded next (future next (by simp [next,noMarks]))
      change r.source.val < f.pre.length+1 at edge
      omega
    · omega
  unfold FlagRead.value at yes
  rw [eq,f.cleared] at yes
  contradiction

structure Parent (h : History) (scan : SplitReturn.Observation.AncestorScan h) (r : FlagRead h.flags) where
  era : Flags h r.slot
  initialized : HappensBefore h era.clear r.event
  endAt : Event
  held : ∀ x ∈ scan.reads, x.slot = r.slot → RetainedThrough h x.event endAt
  next : ∀ x ∈ era.future, x.value = false → SequencedBefore endAt x.event

private theorem sameWord {f : Nat → List FlagWrite} {i j} (same : i = j)
    (idx : Fin (f i).length) : (f j).get ⟨idx.val,same ▸ idx.isLt⟩ = (f i).get idx := by
  subst j
  rfl

private theorem parent {r : FlagRead h.flags} (p : Parent h check.ancestors r) (co : Coherent h)
    (rr : ∀ a ∈ h.flagReads, ∀ b ∈ h.flagReads, a.slot = b.slot → HappensBefore h a.event b.event → a.source.val ≤ b.source.val)
    (mem : r ∈ h.flagReads) (inChain : r.slot ∈ check.ancestors.chain) (yes : r.value = true)
    (d : Departed h atPoll wait) (seen : InWorker atPoll r.event) (recheck : SequencedBefore wait check.ancestors.event) :
    ExternalStop h check wait := by
  have lower := co.flagWriteRead r mem p.era.first (p.era.cleared ▸ p.initialized)
  refine .ancestor r co ⟨d.keeps seen,recheck,yes,inChain,fun x src slot => rr r mem x (check.ancestors.afterStart x src).1 slot.symm,?_⟩
  intro x src slot i after cleared
  let k : Fin (h.flags r.slot).length := ⟨i.val,slot ▸ i.isLt⟩
  have word : (h.flags r.slot).get k = (h.flags x.slot).get i := sameWord slot i
  have future := p.era.future_of_false k (Nat.lt_of_le_of_lt lower after) (word ▸ cleared)
  exact ⟨p.endAt,p.held x src slot,word ▸ p.next _ future (word ▸ cleared)⟩

/-- Modification order between consecutive search generations contains only request RMWs. -/
structure AbortEra (writes : List Write) where
  begin : Fin writes.length
  final : Fin writes.length
  rmw : ∀ i : Fin writes.length, begin.val < i.val → i.val ≤ final.val →
    (writes.get i).value = SearchAbort.request (writes.get ⟨i.val-1,by have := i.isLt; omega⟩).value

structure Shared (m : Memory) (check : OwnerCheck m.history) (wait : Event) where
  raw : Option (Read m.abort)
  observed : check.shared = raw.map (fun r => ⟨m.abort,r⟩)
  era : AbortEra m.abort
  coherent : ∀ r ∈ m.abortReads, SplitRebookingMir.Coherence m.history m.abort r
  readRead : ∀ r ∈ m.abortReads, ∀ q ∈ m.abortReads, HappensBefore m.history r.event q.event → r.source.val ≤ q.source.val
  owner : ∀ q, raw = some q → q ∈ m.abortReads ∧ SequencedBefore wait q.event ∧
    ∀ i : Fin m.abort.length, i.val = era.final.val+1 → HappensBefore m.history q.event (m.abort.get i).event

private theorem shared {m : Memory} {check : OwnerCheck m.history} (e : Shared m check wait) (d : Departed m.history atPoll wait)
    (r : Read m.abort) (mem : r ∈ m.abortReads) (before : InWorker atPoll r.event)
    (init : HappensBefore m.history (m.abort.get e.era.begin).event r.event)
    (yes : SearchAbort.Stopped (m.abort.get r.source).value) : ExternalStop m.history check wait := by
  refine .abort ?_
  intro obs site
  rw [e.observed] at site
  obtain ⟨q,found,rfl⟩ := Option.map_eq_some_iff.mp site
  obtain ⟨logged,retest,next⟩ := e.owner q found
  have lower := (e.coherent r mem).writeRead e.era.begin init
  refine .seen r e.era.final.val ⟨d.keeps before,retest,e.readRead r mem q logged,next,(e.coherent q logged).readWrite,?_⟩ yes
  intro i after upto
  exact ⟨⟨i.val-1,by have := i.isLt; omega⟩,by simp; omega,e.era.rmw i (by omega) upto⟩

/-- Parent pointers in the retained SplitPointTask values. The worker starts from
    its assigned Arc. Finalization restores the exact parent field of that slot. -/
inductive Chain (parent : Nat → Option Nat) : Option Nat → List Nat → Prop where
  | root : Chain parent none []
  | active : Chain parent (parent slot) tail → Chain parent (some slot) (slot :: tail)

private theorem Chain.unique (a : Chain links p xs) (b : Chain links p ys) : xs = ys := by
  induction a generalizing ys with
  | root => cases b; rfl
  | active _ ih => cases b with | active next => rw [ih next]

structure Ancestry (slot : Nat) (ancestors : List Nat) where
  links : Nat → Option Nat
  cachedChain : List Nat
  assigned : Chain links (some slot) cachedChain
  restored : Chain links (links slot) ancestors

structure CacheAt (m : Memory) (atPoll : Event) (cachedChain : List Nat) where
  cache : SplitCutoff.Cache
  reads : List (FlagRead m.history.flags)
  assignment : CutoffVisibility.StopSource.Execution m.history cachedChain cache reads
  localClock : ∀ r ∈ reads, InWorker atPoll r.event

structure PollMemory (m : Memory) (check : OwnerCheck m.history) (wait : Event) (slot : Nat) (ps : List Post)
    (atPoll : Event) (reads : List (FlagRead m.history.flags)) where
  loads : NativeStopReturns.Loads m atPoll
  recheck : SequencedBefore wait check.ancestors.event
  coherent : Coherent m.history
  readRead : ∀ a ∈ m.history.flagReads, ∀ b ∈ m.history.flagReads, a.slot = b.slot →
    HappensBefore m.history a.event b.event → a.source.val ≤ b.source.val
  currentEra : Flags m.history slot
  stamp : Nat → Event
  published : currentEra.marks = marks stamp ps
  localStart : ∀ r ∈ reads, r.slot = slot → InWorker atPoll r.event →
    HappensBefore m.history currentEra.clear r.event
  localEnd : ∀ r ∈ reads, r.slot = slot → InWorker atPoll r.event →
    ∀ i : Fin (m.history.flags slot).length, i.val = currentEra.pre.length+1+currentEra.marks.length →
      HappensBefore m.history r.event ((m.history.flags slot).get i).event
  sharedState : Shared m check wait
  abortStart : ∀ r, loads.abort = some r → HappensBefore m.history (m.abort.get sharedState.era.begin).event r.event
  helper : ∀ r, loads.helper = some r →
    ∃ clear, SplitReturn.Observation.HelperJobRead m.history r clear

structure PollContext (m : Memory) (check : OwnerCheck m.history) (wait : Event) (slot : Nat) (ps : List Post)
    (atPoll : Event) (reads : List (FlagRead m.history.flags)) extends PollMemory m check wait slot ps atPoll reads where
  parentEra : ∀ r ∈ reads, r.slot ∈ check.ancestors.chain → InWorker atPoll r.event →
    Parent m.history check.ancestors r

structure PollOperands (m : Memory) (check : OwnerCheck m.history) (wait : Event) (slot : Nat) (ps : List Post)
    (atPoll : Event) (cachedChain : List Nat) extends CacheAt m atPoll cachedChain,
      PollContext m check wait slot ps atPoll toCacheAt.reads where
  positive : NativeStopReturns.poll (.cached cachedChain cache reads assignment localClock) loads = some true

structure Samples (m : Memory) (check : OwnerCheck m.history) (wait : Event) (slot : Nat) (ps : List Post)
    (atPoll : Event) extends Ancestry slot check.ancestors.chain,
      PollOperands m check wait slot ps atPoll toAncestry.cachedChain

theorem Samples.cause (s : Samples m check wait slot ps atPoll) (departed : Departed m.history atPoll wait) :
    (∃ p ∈ ps, p.cut = true) ∨ ExternalStop m.history check wait := by
  let chain : NativeStopReturns.ChainRead m atPoll :=
    .cached s.cachedChain s.cache s.reads s.assignment s.localClock
  have yes : NativeStopReturns.poll chain s.loads = some true := s.positive
  cases bit : chain.set with
  | true =>
    obtain ⟨r,inReads,mem,inChain,localRead,on⟩ := chain.flag bit
    change r.slot ∈ s.cachedChain at inChain
    rw [s.assigned.unique (.active s.restored)] at inChain
    rcases List.mem_cons.mp inChain with same | parentSlot
    · subst slot
      exact .inl (current s.currentEra s.coherent mem on s.published
        (s.localStart r inReads rfl localRead) (s.localEnd r inReads rfl localRead))
    · exact .inr (parent (s.parentEra r inReads parentSlot localRead) s.coherent s.readRead
        mem parentSlot on departed localRead s.recheck)
  | false =>
    cases ar : s.loads.abort with
    | none => simp [NativeStopReturns.poll,bit,ar] at yes
    | some r =>
      obtain ⟨mem,_,localRead,_⟩ := s.loads.abortRecorded r ar
      cases hr : s.loads.helper with
      | none =>
        have stopped : NativeStopReturns.abortBit m r = true := by simpa [NativeStopReturns.poll,bit,ar,hr] using yes
        exact .inr (shared s.sharedState departed r mem localRead (s.abortStart r ar)
          ((SearchAbort.stopped_iff _).2 (beq_iff_eq.mp stopped)))
      | some helper =>
        have hot : NativeStopReturns.helperBit m atPoll.worker helper = some true := by
          simp only [NativeStopReturns.poll,bit,ar,hr] at yes
          split at yes <;> simp_all
        obtain ⟨clear,seen⟩ := s.helper helper hr
        unfold NativeStopReturns.helperBit at hot
        rw [SplitReturn.Observation.helperJobReadContract_proven _ _ _ clear seen] at hot
        contradiction

/-- A latched cutoff short-circuits both abort loads. Its explanation therefore
    needs only the observed flag and its local publication or retained ancestor. -/
inductive FlagCause (h : History) (check : OwnerCheck h) (wait : Event)
    (slot : Nat) (ps : List Post) (r : FlagRead h.flags) : Type where
  | here : r.slot = slot → (∃ p ∈ ps, p.cut = true) → FlagCause h check wait slot ps r
  | ancestor : r.slot ∈ check.ancestors.chain → Parent h check.ancestors r → Coherent h →
      (∀ a ∈ h.flagReads, ∀ b ∈ h.flagReads, a.slot = b.slot →
        HappensBefore h a.event b.event → a.source.val ≤ b.source.val) →
      SequencedBefore wait check.ancestors.event → FlagCause h check wait slot ps r

inductive Evidence (m : Memory) (check : OwnerCheck m.history) (wait : Event)
    (slot : Nat) (ps : List Post) (poll : Event) : Type where
  | full : Samples m check wait slot ps poll → Evidence m check wait slot ps poll
  | flag (cache : CacheAt m poll (slot :: check.ancestors.chain)) (r : FlagRead m.history.flags) :
      cache.cache.latched = true → r ∈ cache.reads → r ∈ m.history.flagReads → r.value = true →
      FlagCause m.history check wait slot ps r → Evidence m check wait slot ps poll

theorem Evidence.cause (ev : Evidence m check wait slot ps poll) (departed : Departed m.history poll wait) :
    (∃ p ∈ ps, p.cut = true) ∨ ExternalStop m.history check wait := by
  cases ev with
  | full samples => exact samples.cause departed
  | flag cache r _ read logged yes why =>
    cases why with
    | here _ hit => exact .inl hit
    | ancestor target era co rr after =>
      exact .inr (parent era co rr logged target yes departed (cache.localClock r read) after)

private def ordinary : Tickets.Action → Bool
  | .take .. | .empty .. | .report .. => true
  | _ => false

def discard (s : State w) (left right : List (Nat × Nat)) (worker index : Nat) : State w :=
  {s with
    pending := left ++ right
    discarded := index :: s.discarded
    booking := false
    departing := worker :: s.departing
    outside := if s.reports.any Post.cut then s.outside else true}

structure Release (actions : List SplitCompletion.Action) (poll : Event) where
  pos : Fin actions.length
  worker : SplitCompletion.Worker
  event : Event
  reset : actions.get pos = .finish worker event
  who : worker.val = poll.worker
  before : SequencedBefore poll event

def Release.departed {a : SplitCompletion.Activation ws} (r : SplitCompletion.Retirement hist ws a)
    (e : Release a.actions poll) : Departed hist poll r.read.event :=
  ⟨ws,a,r,e.pos,by rw [e.reset]; rfl,by rw [e.reset]; exact e.before,rfl⟩

/-- Take, exhaustion and lock publications have their original steps. A discard must
    consume its worker's real positive stop poll. The caller cannot insert a free-standing
    external stop or use a child return of zero to explain a discard. -/
inductive Run (m : Memory) (check : OwnerCheck m.history) (wait : Event) (slot : Nat)
    (valid : State w → Post → Prop) (len : Nat) (start : BitVec w) (owner : SplitCompletion.Worker) (complete : List SplitCompletion.Action) :
    List SplitCompletion.Action → State w → Prop where
  | initial : Run m check wait slot valid len start owner complete [] (Tickets.initial start owner.val)
  | work : Run m check wait slot valid len start owner complete actions s → Step len s a t →
      ordinary a = true → (∀ p ∈ a.posts, valid s p) → Run m check wait slot valid len start owner complete actions t
  | book (i : SplitCompletion.Worker) (stamp : Event) : Run m check wait slot valid len start owner complete actions s →
      Step len s (.book i.val) t → Run m check wait slot valid len start owner complete (actions ++ [.book i stamp]) t
  | finish (i : SplitCompletion.Worker) (stamp : Event) : Run m check wait slot valid len start owner complete actions s →
      Step len s (.leave i.val) t → Run m check wait slot valid len start owner complete (actions ++ [.finish i stamp]) t
  | stop (stamp : Event) (i : Nat) : Run m check wait slot valid len start owner complete actions s →
      s.pending = left ++ (stamp.worker,i) :: right → Evidence m check wait slot s.reports stamp → Release complete stamp →
      Run m check wait slot valid len start owner complete actions (discard s left right stamp.worker i)

private theorem Run.refine (run : Run m check wait slot valid len start owner complete actions s)
    (cause : ∀ ps atPoll, Evidence m check wait slot ps atPoll → Release complete atPoll → (∃ p ∈ ps, p.cut = true) ∨ outside) :
    Retirement.ExecutionWith outside valid len start owner actions s := by
  induction run with
  | initial => exact .initial
  | @work _ _ a _ _ op event checked ih =>
    exact .work ih op (by cases a <;> simp_all [ordinary,Retirement.effect])
      (fun eq => by subst_vars; contradiction) checked
  | book i e _ op ih => exact .book i e ih op
  | finish i e _ op ih => exact .finish i e ih op
  | @stop actions s left right stamp i _ held polled reset ih =>
    have quiet (i : Nat) : Retirement.effect (.discard i) = none := rfl
    have checkDiscard (s : State _) (i : Nat) : ∀ p ∈ (Tickets.Action.discard i).posts, valid s p := by simp [Tickets.Action.posts]
    cases cut : s.reports.any Post.cut with
    | true =>
      have stopped : s.stopped := .inr (List.any_eq_true.mp cut)
      simpa [discard,cut] using (Retirement.ExecutionWith.work ih (Step.discard held stopped) (quiet _) (by intro h; cases h) (checkDiscard s _))
    | false =>
      have why := (cause _ _ polled reset).resolve_left (fun found => by rw [List.any_eq_true.mpr found] at cut; contradiction)
      have outside := Retirement.ExecutionWith.work ih Step.externalStop rfl (fun _ => why) (by simp [Tickets.Action.posts])
      have stopped : ({s with outside := true} : State _).stopped := Or.inl rfl
      simpa [discard,cut] using (Retirement.ExecutionWith.work outside (Step.discard held stopped) (quiet _)
        (by intro h; cases h) (checkDiscard _ _))

private theorem Run.with_wait {m : Memory} {check : OwnerCheck m.history} {a : SplitCompletion.Activation ws}
    (r : SplitCompletion.Retirement m.history ws a) (stamp : r.read.event = atWait)
    (run : Run m check atWait slot valid len start a.owner a.actions a.actions s) :
    Retirement.ExecutionWith (ExternalStop m.history check atWait) valid len start a.owner a.actions s := by
  apply run.refine
  intro ps event polled reset
  exact polled.cause (stamp ▸ Release.departed r reset)

/-- First replay the mask and worker path with an unconstrained termination cause.
    All stopped polls then use the single acquire returned by that path's physical
    guard loop. Their release indices refer to its own RMWs. -/
theorem Run.project {m : Memory} {check : OwnerCheck m.history} (o : Mask.Observed m.history prior owner g next)
    (run : Run m check o.waitAt slot valid len start owner g.actions g.actions s) :
    Retirement.ExecutionWith (ExternalStop m.history check o.waitAt) valid len start owner g.actions s := by
  have shape := run.refine (outside := True) (fun _ _ _ _ => .inr trivial)
  obtain ⟨r,atWait,_⟩ := o.wait shape
  exact Run.with_wait r atWait run

abbrev Scored (m : Memory) (check : OwnerCheck m.history) (wait : Event) (slot : Nat) (words : List Write)
    (task : Schedule.Task) (done_ : PvsMoveLoop.ChildRelation Nat) (len : Nat) (start : BitVec w) (owner : SplitCompletion.Worker) (actions : List SplitCompletion.Action) (s : State w) :=
  Run m check wait slot (fun s p => ∃ c : RawSchedule.Commit m.history words task s.reports, p = c.post ∧ c.calls done_) len start owner actions actions s

theorem join {m : Memory} (o : Mask.Observed m.history prior owner g next) (check : OwnerCheck m.history)
    (run : Scored m check o.waitAt slot words task done_ len start owner g.actions s)
    (window : Aspiration.Window task.entry.alpha task.beta) (cap : task.entry.alpha ≤ max original task.entry.best) :
    ∃ j : Join task done_ len start, j.state = s ∧ j.wait.read.event = o.waitAt ∧
      j.wait.read.source.val = prior.length+1+g.actions.length ∧ j.activation.actions = g.actions ∧
      j.answer = SplitReturn.checked check.stopped (task.locked s.reports).best :=
  RawSchedule.join o check (Run.project o run) window cap

/-- The same projection for original modification order containing redundant
    resets. Every poll is discharged against the one effective worker release and the
    very same acquire event as the final raw idle-loop observation. -/
theorem join_raw {m : Memory} {ws : List Write} (g : SlotFrameMir.RawMask ws)
    (exit : SlotFrameMir.OwnerExit m.history g) (gates : SlotFrameMir.GateHistory m.history g)
    (mark : (ws.get g.mark).event = exit.clock.atStep 1)
    (sync : ∀ pre src rest, exit.log = pre ++ .mask src :: rest → ∀ i, AtomicWord.InReleaseSequence ws i src →
      m.history.otherSync (ws.get i).event (exit.clock.atStep (4+pre.length)))
    (load : Read scoreWords) (scoreSite : load.event = exit.clock.atStep (exit.log.length+5))
    (coherent : SplitRebookingMir.Coherence m.history scoreWords load)
    (check : OwnerCheck m.history)
    (run : Scored m check (exit.clock.atStep (3+exit.log.length)) slot words task done_ len start
      g.owner (SlotFrameMir.RawProjection.actions g) s)
    (window : Aspiration.Window task.entry.alpha task.beta) (cap : task.entry.alpha ≤ max original task.entry.best) :
    ∃ (j : Join task done_ len start) (pre : List (SplitSchedulerMir.Guard ws)) (src : Fin ws.length),
      exit.log = pre ++ [.mask src] ∧ (ws.get src).value = 0 ∧
      g.mark.val ≤ src.val ∧ src.val ≤ g.mark.val+g.actions.length ∧
      j.state = s ∧ j.wait.read.event = exit.clock.atStep (3+exit.log.length) ∧
      j.wait.read.source.val = (SlotFrameMir.RawProjection.rf g src).val ∧
      j.answer = SplitReturn.checked check.stopped (task.locked s.reports).best := by
  let a := SlotFrameMir.RawProjection.activation g gates.closed
  have mem := SlotFrameMir.RawProjection.memory exit gates.closed mark sync scoreWords load.source
    ⟨fun i hb => coherent.writeRead i (by simpa only [scoreSite,SplitSchedulerMir.scoreLoad] using hb),
     fun i hb => coherent.readWrite i (by simpa only [scoreSite,SplitSchedulerMir.scoreLoad] using hb)⟩
  obtain ⟨r,_,_,_,site⟩ := SplitSchedulerMir.retired_after_assignment m.history _ _ a _ _ load.source
    (SlotFrameMir.RawProjection.exit_returned exit) (SlotFrameMir.RawProjection.open_pool g exit.poolOpen) mem
  simp only [List.length_map] at site
  have path : RawSchedule.Run m.history words (ExternalStop m.history check _) task done_ len start
      g.owner (SlotFrameMir.RawProjection.actions g) s := Run.with_wait r site run
  exact Mask.join_raw g exit gates mark sync load scoreSite coherent check (path.schedule window cap)

end NeuralReversi.PvsParallel.Interleaving.StopSchedule

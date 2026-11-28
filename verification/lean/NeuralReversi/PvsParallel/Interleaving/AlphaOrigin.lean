module
public import NeuralReversi.PvsParallel.Interleaving.ScoreOrigin
public import NeuralReversi.PvsParallel.SlotFrameMir.RawHandoff
meta import NeuralReversi.PvsParallel.Interleaving.ScoreOrigin
meta import NeuralReversi.PvsParallel.SlotFrameMir.RawHandoff
meta import NeuralReversi.Threading.AtomicWord

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.AlphaOrigin

open Threading SplitWorkerMir
open NativeSplit (Tickets)
open AtomicWord (Write)
open SlotFrameMir.RawProjection (PhysicalOwner Reports subsequent)
open TranspositionTable.Packing (encodeBestMove)

variable {t : NativeSplit.Tickets rc pv beta}
variable {g : SlotFrameMir.RawMask mask} {p : PhysicalOwner g hist time slot t}

abbrev words (physical : Reports p) :=
  physical.pre ++ t.final.alpha ++ subsequent p.exit slot .alpha p.later

/-- The current generation contributes its initialization and exactly the alpha
    stores emitted by the locked instruction replay, in modification order. -/
theorem writes (physical : Reports p) :
    words physical = physical.pre ++ (t.a :: t.calls.flatMap (fun c => c.ops.filterMap FieldStores.Physical.alpha?)) ++
      subsequent p.exit slot .alpha p.later := by
  have same := (FieldStores.modification_orders t.calls (IteratorFields.field_replay t.replay)).1
  change t.final.alpha = [t.a] ++ _ at same
  simp only [words,same,List.singleton_append]

theorem publication (p : PhysicalOwner g hist time slot t) (write : Write)
    (issued : write ∈ t.calls.flatMap (fun packet => packet.ops.filterMap FieldStores.Physical.alpha?)) :
    Nonempty (SlotFrameMir.WorkerStore hist time g slot write.event) := by
  obtain ⟨packet,found,member⟩ := List.mem_flatMap.mp issued
  obtain ⟨op,atOp,emitted⟩ := List.mem_filterMap.mp member
  have alpha : op = .writeAlpha write := by cases op <;> simp_all [FieldStores.Physical.alpha?]
  obtain ⟨locked,_,rfl⟩ := List.mem_map.mp found
  change op ∈ t.traffic locked.order at atOp
  obtain ⟨obs,addr,site,fields⟩ := SplitSchedulerMir.WorkerFields.traffic_source atOp
  have reached : op ∈ addr.traffic := fields ▸ atOp
  obtain ⟨_,reset,_,_,same⟩ := p.reset site (addr.needs_reset_of_access reached)
  have before := addr.store_before_unlock reached (Or.inl alpha)
  have released := addr.unlock_before_reset
  rw [←same] at released
  exact ⟨⟨reset,before.1.trans released.1,Nat.lt_trans before.2 released.2⟩⟩

/-- The raw source is a modification-order index. Repeated words on different
    slots or at another instruction remain distinct observations. -/
structure Register where
  slot : SlotFrameMir.Slot
  field : SplitSchedulerMir.OwnerFields.Field
  event : AtomicWord.Event
  source : Nat
  word : BitVec 32
  acquire : Bool

def Register.read (r : Register) (target : SlotFrameMir.Slot) (mo : List Write) : Option (AtomicWord.Read mo) :=
  match r.field with
  | .alpha => if h : r.slot = target ∧ r.source < mo.length then
      if !r.acquire && (mo.get ⟨r.source,h.2⟩).value.setWidth 32 == r.word then
        some ⟨r.event,⟨r.source,h.2⟩,false⟩ else none
    else none
  | _ => none

def Register.Coherent (r : Register) (hist : CutoffVisibility.History) (mo : List Write) : Prop :=
  (∀ i : Fin mo.length, CutoffVisibility.HappensBefore hist (mo.get i).event r.event → i.val ≤ r.source) ∧
  (∀ i : Fin mo.length, CutoffVisibility.HappensBefore hist r.event (mo.get i).event → r.source < i.val)

def registers (target : SlotFrameMir.Slot) (mo : List Write) (rows : List Register) : List (AtomicWord.Read mo) :=
  if (rows.map Register.event).Nodup then rows.filterMap (fun r => r.read target mo) else []

/-- Project the uniformly addressed load bus onto the physical split's alpha
    field. Caller evidence consists of ordinary per-location coherence. -/
def memory (target : SlotFrameMir.Slot) (mo : List Write) (rows : List Register)
    (coherent : ∀ r ∈ rows, r.slot = target → r.field = .alpha → r.Coherent hist mo) :
    NativeSplit.AlphaLoads.Memory hist mo where
  reads := registers target mo rows
  coherent := by
    intro read member
    unfold registers at member
    split at member <;> try contradiction
    obtain ⟨r,inRows,hit⟩ := List.mem_filterMap.mp member
    unfold Register.read at hit
    split at hit <;> try contradiction
    rename_i alpha
    split at hit <;> try contradiction
    rename_i present
    split at hit <;> try contradiction
    cases hit
    exact ⟨(coherent r inRows present.1 alpha).1,(coherent r inRows present.1 alpha).2⟩

structure Addressed (p : PhysicalOwner g hist time slot t) where
  pre : List Write
  mutex : SplitSchedulerMir.Mutex.Capture hist
  checked : SplitSchedulerMir.Mutex.covers mutex.ops (t.journal.map fun row => (row.acquired,row.published))
  stateLocks : SplitSchedulerMir.Startup.StateLocks hist
  started : t.dispatches.all fun d => !(d.sections.any SplitSchedulerMir.WorkerFields.Section.needsReset) ||
    SplitSchedulerMir.Startup.ready mutex.ops (SplitSchedulerMir.Startup.stateOps stateLocks)
      (p.exit.timeline.atStep (SplitSchedulerMir.initPhases.idxOf .assign)) d.entry
  rows : List Register
  coherent : ∀ r ∈ rows, r.slot = slot → r.field = .alpha →
    r.Coherent hist (pre ++ t.final.alpha ++ subsequent p.exit slot .alpha p.later)
  operands : t.state.writes.all (NativeSplit.AlphaLoads.ticket (memory _ _ rows coherent).reads t.journal t.stream)

def Addressed.reports (s : Addressed p) : Reports p :=
  ⟨s.pre,s.mutex,s.checked,s.stateLocks,s.started,memory _ _ s.rows s.coherent,s.operands⟩

def sample (physical : Reports p) (ps : List Tickets.Post)
    (part : t.reports = ps.map Tickets.Post.report ++ r :: tail)
    (got : IteratorReports.Load hist (words physical) t.a.event event word)
    (before : AtomicWord.SequencedBefore event (t.post part).lock.acquired) :
    RawSchedule.Sample hist (words physical) (ScoreOrigin.task t root cut depth) ps := by
  let interval := t.interval part
  let later := subsequent p.exit slot .alpha p.later
  exact ⟨interval.first physical.pre later,interval.last physical.pre later,got.read,
    ⟨by rw [interval.seed_at]; exact got.published,(t.post part).lock.acquired,before,
      (physical.alpha.loads part).next,got.writeRead,got.readWrite⟩,interval.window physical.pre later⟩

def scout (physical : Reports p) (ps : List Tickets.Post)
    (part : t.reports = ps.map Tickets.Post.report ++ r :: tail) :
    RawSchedule.Sample hist (words physical) (ScoreOrigin.task t root cut depth) ps :=
  sample physical ps part (physical.loads part).scout ((physical.alpha.finish part).before _ false (by simp))

def wide (physical : Reports p) (ps : List Tickets.Post)
    (part : t.reports = ps.map Tickets.Post.report ++ r :: tail) (used : retry (t.post part).input = true) :
    RawSchedule.Sample hist (words physical) (ScoreOrigin.task t root cut depth) ps :=
  sample physical ps part ((physical.loads part).wide used) ((physical.alpha.finish part).before _ true (fun _ => used))

def prior (t : NativeSplit.Tickets rc pv beta)
    (part : t.reports = earlier ++ r :: tail) :
    { fields : FieldStores.Fields // fields.state = SplitReturn.stateAfter pv beta t.initial.state earlier } := by
  let x := t.post part
  have found := IteratorFields.before_posted rc t.workers t.result t.ran t.replay x part
  let row := IteratorReports.prefixLog found
  let mid := row.val
  obtain ⟨before,passed,_⟩ := row.property
  have inv := (FieldStores.replay_preserves _
    (FieldStores.seed t.initial t.a t.b t.m t.seedAlpha t.seedBest t.seedMove) before).state
  have same : mid.reports = earlier := passed
  change mid.fields.state = SplitWindow.scan pv beta t.initial.state (mid.reports.map SplitResult.Report.score) at inv
  exact ⟨mid.fields,by simpa only [same,SplitReturn.stateAfter] using inv⟩

/-- Child return words come from the two calls decoded for this physical ticket.
    The wide operand is only consumed on the native re-search arm. -/
def children (i : Input) : PvsMoveLoop.ChildSearch Nat :=
  fun _ useWide _ _ => if useWide then i.wide.toInt else i.scout.toInt

def commit (physical : Reports p) (ps : List Tickets.Post)
    (part : t.reports = ps.map Tickets.Post.report ++ r :: tail) :
    RawSchedule.Commit hist (words physical) (ScoreOrigin.task t root cut depth) ps := by
  let x := t.post part
  let init := prior t part
  let a := scout (root := root) (cut := cut) (depth := depth) physical ps part
  let again := if used : retry x.input then some (wide (root := root) (cut := cut) (depth := depth) physical ps part used) else none
  have asRead : a.value = x.input.alpha.toInt := (physical.loads part).scout.wordRead.symm
  have pivot := (IteratorFields.posted_window t.covers t.ran t.replay x).1
  exact ⟨x.input.index,a,again,children x.input,⟨false,init.val.alpha,init.val.best⟩,rfl,
    congrArg SplitWindow.State.alpha init.property,congrArg SplitWindow.State.best init.property,by
      change again.isSome = SplitWindow.reSearch pv a.value ((children x.input) x.input.index false _ _)
      rw [asRead]
      change again.isSome = SplitWindow.reSearch pv x.input.alpha.toInt x.input.scout.toInt
      have againWord := SplitWorkerMir.retry_eq (physical.loads part).safe
      rw [pivot] at againWord
      rw [←againWord]
      cases used : retry x.input <;> simp [again,used]⟩

deriving instance DecidableEq for SplitResult.Report

structure Next (reports : List SplitResult.Report) (posts : List Tickets.Post) where
  report : SplitResult.Report
  later : List SplitResult.Report
  position : reports = posts.map Tickets.Post.report ++ report :: later

def next (reports : List SplitResult.Report) (posts : List Tickets.Post) : Option (Next reports posts) := do
  let height := posts.length
  match hs : reports.drop height with
  | [] => none
  | r :: tail =>
    if prior : reports.take height = posts.map Tickets.Post.report then
      some ⟨r,tail,by rw [←prior,←hs,List.take_append_drop]⟩
    else none

/-- At each live lock section the cursor computes the next physical report
    from the posted prefix. Stopped tickets never consume a report row. -/
def atNext (physical : Reports p) (posts : List Tickets.Post) :
    Option (RawSchedule.Commit hist (words physical) (ScoreOrigin.task t root cut depth) posts) :=
  (next t.reports posts).map fun row => commit physical posts row.position

def source (physical : Reports p) : ScoreOrigin.Source hist (words physical) (ScoreOrigin.task t root cut depth) :=
  ⟨fun ps => PLift (next t.reports ps).isSome,fun ps found =>
    commit physical ps ((next t.reports ps).get found.down).position⟩

/-- The worker schedule asks the next lock packet for its commit. No caller can
    substitute a different slot's alpha array or an earlier report for that event. -/
structure SplitOperands (t : NativeSplit.Tickets rc pv beta) where
  memory : NativeStopReturns.Memory
  mask : List Write
  generation : SlotFrameMir.RawMask mask
  clock : AtomicWord.Event → Nat
  bankIndex : Nat
  flagAddress : SlotFrameMir.Slot → Nat
  seed : ScoreOrigin.Seed t.b

structure SetupOperands (t : NativeSplit.Tickets rc pv beta) extends SplitOperands t where
  owner : SlotFrameMir.RawProjection.OwnerSetup generation memory.history clock (generation.owner,bankIndex) t

structure OwnerOperands (t : NativeSplit.Tickets rc pv beta) extends SplitOperands t where
  owner : PhysicalOwner generation memory.history clock (generation.owner,bankIndex) t

def OwnerOperands.setup (b : OwnerOperands t) : SetupOperands t := ⟨b.toSplitOperands,b.owner.toOwnerSetup⟩

structure Operands (t : NativeSplit.Tickets rc pv beta) extends OwnerOperands t where
  registers : Addressed owner

structure Capture (t : NativeSplit.Tickets rc pv beta) (root cut : Bool) (depth : Nat)
    (child : PvsMoveLoop.ChildRelation Nat) extends Operands t where
  state : Tickets.State t.width
  run : ScoreOrigin.RunWith t memory owner.check (owner.exit.clock.atStep (3+owner.exit.log.length))
    (flagAddress (generation.owner,bankIndex)) (words registers.reports) (ScoreOrigin.task t root cut depth)
    child generation.owner (SlotFrameMir.RawProjection.actions generation) (source registers.reports)
    (SlotFrameMir.RawProjection.actions generation) t.calls state

def Capture.withReports (c : Capture t root cut depth old)
    (reports : ScoreOrigin.ReportCalls t (source (root := root) (cut := cut) (depth := depth) c.registers.reports) child t.calls) :
    Capture t root cut depth child :=
  {c with run := c.run.withReports reports}

def Capture.score (c : Capture t root cut depth child) : ScoreOrigin.Capture t root cut depth child :=
  ⟨c.memory,c.mask,c.generation,c.clock,c.bankIndex,c.flagAddress,c.owner,c.seed,words c.registers.reports,c.state,c.run.toRun⟩

theorem window (physical : Reports p) (posts : List Tickets.Post)
    (part : t.reports = posts.map Tickets.Post.report ++ r :: tail)
    (observed : RawSchedule.Sample hist (words physical) (ScoreOrigin.task t root cut depth) posts)
    (site : observed.first = (t.interval part).first physical.pre (subsequent p.exit slot .alpha p.later))
    (end_ : observed.last = (t.interval part).last physical.pre (subsequent p.exit slot .alpha p.later)) :
    physical.pre.length ≤ observed.load.source.val ∧
    observed.load.source.val < physical.pre.length + (t.interval part).current.length := by
  obtain ⟨lo,hi⟩ := observed.order.window.bounds
  rw [site] at lo
  rw [end_] at hi
  have nonempty := (t.interval part).seeded.length_le
  change physical.pre.length ≤ _ at lo
  change _ ≤ physical.pre.length + (t.interval part).current.length - 1 at hi
  simp at nonempty
  exact ⟨lo,by omega⟩

theorem scout_source (physical : Reports p) (posts : List Tickets.Post)
    (part : t.reports = posts.map Tickets.Post.report ++ r :: tail) :
    let load := (scout (root := root) (cut := cut) (depth := depth) physical posts part).load
    physical.pre.length ≤ load.source.val ∧
    load.source.val < physical.pre.length + (t.interval part).current.length :=
  window physical posts part _ rfl rfl

theorem wide_source (physical : Reports p) (posts : List Tickets.Post)
    (part : t.reports = posts.map Tickets.Post.report ++ r :: tail) (again : retry (t.post part).input = true) :
    let load := (wide (root := root) (cut := cut) (depth := depth) physical posts part again).load
    physical.pre.length ≤ load.source.val ∧
    load.source.val < physical.pre.length + (t.interval part).current.length :=
  window physical posts part _ rfl rfl

/-- A live scout consumes the source rank recorded for its particular alpha-load instruction. -/
theorem commit_scout_register (physical : Reports p) (ps : List Tickets.Post)
    (part : t.reports = ps.map Tickets.Post.report ++ r :: tail)
    {row : AtomicWord.Read (words physical)} (site : physical.memory.reads.filter (fun read => read.event == (t.post part).lock.scoutLoad) = [row]) :
    (commit (root := root) (cut := cut) (depth := depth) physical ps part).scout.load.source = row.source :=
  physical.memory.load_source _ _ site

theorem commit_wide_register (physical : Reports p) (ps : List Tickets.Post)
    (part : t.reports = ps.map Tickets.Post.report ++ r :: tail) (again : retry (t.post part).input = true)
    {row : AtomicWord.Read (words physical)} (site : physical.memory.reads.filter (fun read => read.event == (t.post part).lock.wideLoad) = [row]) :
    ((commit (root := root) (cut := cut) (depth := depth) physical ps part).wide.map
      (fun s => s.load.source)) = some row.source := by
  let value := wide (root := root) (cut := cut) (depth := depth) physical ps part again
  have picked : (commit (root := root) (cut := cut) (depth := depth) physical ps part).wide = some value := by
    change (if h : retry (t.post part).input = true then some (wide physical ps part h) else none) = some value
    simp [again,value]
  rw [picked,Option.map_some]
  exact congrArg some (physical.memory.load_source _ _ site)

namespace Regression

private def history : CutoffVisibility.History := ⟨[],fun _ => [],[],[],fun _ _ => False⟩
private abbrev changes : List Write := [⟨⟨0,0⟩,0,false,false⟩,⟨⟨1,2⟩,9,false,false⟩,⟨⟨0,4⟩,0,false,false⟩]
private def field : Register := ⟨(0,3),.alpha,⟨0,1⟩,0,0,false⟩
private def got (rows : List Register) :=
  NativeSplit.AlphaLoads.operand (registers (0,3) changes rows) field.event field.word

example : got [field] = true ∧ got [{field with slot := (0,4)}] = false ∧
    got [{field with field := .score}] = false ∧ got [{field with acquire := true}] = false ∧
    got [field,{field with slot := (0,4)}] = false := by decide

private def load (source : Fin changes.length) : AtomicWord.Read changes := ⟨⟨0,1⟩,source,false⟩

private theorem only_local (h : CutoffVisibility.HappensBefore history a b) : AtomicWord.SequencedBefore a b := by
  induction h with
  | sequenced relation => exact relation
  | synchronized at_ => exact Fin.elim0 at_
  | other h => contradiction
  | trans _ _ first second =>
    exact ⟨first.1.trans second.1,Nat.lt_trans first.2 second.2⟩

private def stale : SplitWindow.AlphaRead history (load 0) 0 1 where
  published := .own (by unfold AtomicWord.SequencedBefore; decide)
  acquire := ⟨0,3⟩
  beforeAcquire := by unfold AtomicWord.SequencedBefore; decide
  afterAcquire := by
    intro i next
    have idx : i = 2 := Fin.ext next
    subst i
    exact .sequenced (by unfold AtomicWord.SequencedBefore; decide)
  writeRead := by
    intro i before
    have sb := only_local before
    have : ∀ ix : Fin changes.length, AtomicWord.SequencedBefore (changes.get ix).event (load 0).event → ix.val ≤ 0 := by unfold AtomicWord.SequencedBefore; decide
    exact this i sb
  readWrite := by
    intro i after
    have sb := only_local after
    have : ∀ ix : Fin changes.length, AtomicWord.SequencedBefore (load 0).event (changes.get ix).event → 0 < ix.val := by unfold AtomicWord.SequencedBefore; decide
    exact this i sb

/-- Coherence admits a seed read concurrent with a raise to nine. The next
    generation reuses zero, but its write cannot source the very same load. -/
example : AtomicWord.signed32At changes 0 = AtomicWord.signed32At changes 2 ∧
    Nonempty (SplitWindow.AlphaRead history (load 0) 0 1) ∧
    ¬ Nonempty (SplitWindow.AlphaRead history (load 2) 0 1) := by
  refine ⟨rfl,⟨stale⟩,?_⟩
  rintro ⟨later⟩
  have out := later.window.bounds.2
  exact (by decide : ¬ 2 ≤ 1) out

-- Both live workers return zero. The second posted ordinal must use worker one's claim and child return.
private def lock (w : Nat) : IteratorFrames.Observation :=
  ⟨w,3,2,none,5,0,⟨w,7⟩,⟨w,0⟩,⟨w,4⟩,⟨w,0⟩,⟨w,8⟩,⟨w,9⟩,⟨w,10⟩,
    ⟨4,false,false,false,256⟩,0,0,⟨false,0,0⟩⟩
private def call (w : Nat) : List NativeFrames.Record :=
  [.localCall w (.entry 2 0 .splitSearch .none none),.localCall w (.entry 5 0 .pvs .none none),
   .localCall w (.leave 7 5 .pvs none (some ⟨0⟩))]
private def claims : MoveList.ConcurrentTrace.State 8 :=
  ⟨2,[⟨0,3,0,1⟩,⟨1,3,1,2⟩],fun _ => .idle⟩
private def moves : List Move.Move := [⟨0,0,4⟩,⟨0,0,11⟩]
private def posted (i : Nat) : SplitResult.Report := ⟨i,0,0⟩
private def returns : List NativeWorkerFeed.Return := [⟨2,0,5,7,⟨0⟩,true⟩]
private def second : IteratorFrames.Posted moves claims [lock 0,lock 1] (call 0 ++ call 1)
    [posted 0] (posted 1) [(0,posted 0),(1,posted 1)] := by
  let obs := lock 1
  let i := NativeWorkerFeed.received obs.task obs.prefix returns (1#8) obs.alpha obs.fresh
  exact ⟨⟨1,3,1,2⟩,by decide,1,rfl,obs,rfl,returns,rfl,by decide,
    ⟨0,0,11⟩,rfl,rfl,rfl,issued i,rfl,by decide +revert⟩

private def decoded := IteratorFrames.recover (by decide) ⟨second⟩
example : decoded.rank = 1 ∧ decoded.op.worker = 1 ∧ decoded.move.sq = 11 ∧
    decoded.lock.scoutLoad = ⟨1,4⟩ ∧ decoded.body.chosen = 0 := by decide

private def initialFields := FieldStores.initialLog ⟨0,-1,none⟩
  ⟨⟨0,0⟩,0,false,false⟩ ⟨⟨0,1⟩,(-1#32).setWidth 64,false,false⟩ ⟨⟨0,2⟩,0,false,false⟩
private def beforeSecond := FieldStores.replay false 256 encodeBestMove (IteratorFields.atIndex moves) initialFields
  [⟨{decoded.input with index := 0},false,[.readBest (-1),.writeBest ⟨⟨0,9⟩,0,false,false⟩,.readAlpha 0]⟩]
private def priorRow := IteratorReports.prefixLog (out := beforeSecond) (p := fun log => log.reports = [posted 0])
  ⟨_,rfl,by decide⟩
#guard priorRow.val.fields.best == 0 && priorRow.val.best.length == 2

private def equalAlpha : List Write := [⟨⟨1,0⟩,0,false,false⟩,⟨⟨0,1⟩,0,false,false⟩]
private def observations : List (AtomicWord.Read equalAlpha) := [⟨⟨1,2⟩,⟨0,by decide⟩,false⟩,⟨⟨1,4⟩,⟨1,by decide⟩,false⟩]

private def alphaMemory : NativeSplit.AlphaLoads.Memory history equalAlpha where
  reads := observations
  coherent := by
    intro read mem
    obtain ⟨k,rfl⟩ := List.get_of_mem mem
    have bounds : ∀ k : Fin observations.length, ∀ i : Fin equalAlpha.length,
        (AtomicWord.SequencedBefore (equalAlpha.get i).event (observations.get k).event →
          i.val ≤ (observations.get k).source.val) ∧
        (AtomicWord.SequencedBefore (observations.get k).event (equalAlpha.get i).event →
          (observations.get k).source.val < i.val) := by unfold AtomicWord.SequencedBefore; decide
    exact ⟨fun i edge => (bounds k i).1 (only_local edge),fun i edge => (bounds k i).2 (only_local edge)⟩

private def observed := alphaMemory.load (event := decoded.lock.scoutLoad) (word := decoded.input.alpha)
  (seed := ⟨1,0⟩) (by decide) (.own (by unfold AtomicWord.SequencedBefore; decide))

#guard observed.source.val == 1 && observed.read.event == ⟨1,4⟩ && observed.read.acquire == false

end Regression

end NeuralReversi.PvsParallel.Interleaving.AlphaOrigin

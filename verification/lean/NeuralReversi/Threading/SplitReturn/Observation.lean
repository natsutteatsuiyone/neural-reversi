module
public import NeuralReversi.Threading.SearchAbort

@[expose] public section

namespace NeuralReversi.Threading.SplitReturn.Observation

open AtomicWord (Event Write Read ReadWindow SequencedBefore signed32At)
open CutoffVisibility (Coherent FlagRead HappensBefore History Walk)
open SearchAbort (Stopped request)
open SplitCompletion (Activation Retirement RetainedThrough)

structure ScoreRead (history : History) {mask : List Write} {activation : Activation mask}
    (retirement : Retirement history mask activation) {writes : List Write} (load : Read writes)
    (last : Fin writes.length) : Prop where
  finished : SequencedBefore (writes.get last).event retirement.read.event ∨
    ∃ i : Fin activation.actions.length, (activation.actions.get i).release = true ∧
      SequencedBefore (writes.get last).event (activation.actions.get i).event
  beforeLoad : SequencedBefore retirement.read.event load.event
  beforeReuse : ∀ i : Fin writes.length, i.val = last.val + 1 →
    SequencedBefore load.event (writes.get i).event
  writeRead : ∀ i : Fin writes.length, HappensBefore history (writes.get i).event load.event →
    i.val ≤ load.source.val
  readWrite : ∀ i : Fin writes.length, HappensBefore history load.event (writes.get i).event →
    load.source.val < i.val

def ScoreReadContract : Prop :=
  ∀ (history : History) (writes : List Write) (load : Read writes) (last : Fin writes.length)
    {mask : List Write} {activation : Activation mask}
    (retirement : Retirement history mask activation), ScoreRead history retirement load last →
    load.source = last ∧ signed32At writes load.source = signed32At writes last

theorem scoreReadContract_proven : ScoreReadContract := by
  intro history writes load last mask activation retirement read
  have finished : HappensBefore history (writes.get last).event retirement.read.event := by
    rcases read.finished with owner | ⟨i, released, beforeRelease⟩
    · exact .sequenced owner
    · exact .trans (.sequenced beforeRelease)
        (SplitCompletion.handoffContract_proven history mask activation retirement i released)
  have window : ReadWindow load (HappensBefore history) last last.val :=
    ⟨.trans finished (.sequenced read.beforeLoad),
      fun i next => .sequenced (read.beforeReuse i next), read.writeRead, read.readWrite⟩
  have exactSource : load.source = last := Fin.ext (Nat.le_antisymm window.bounds.2 window.bounds.1)
  exact ⟨exactSource, congrArg (signed32At writes) exactSource⟩

structure ReportPublication (activation : Activation mask) where
  worker : SplitCompletion.Worker
  acquired : Event
  checked : Event
  published : Event
  stopped : Bool
  running : stopped = false
  afterAcquire : SequencedBefore acquired checked
  afterCheck : SequencedBefore checked published
  finish : Fin activation.actions.length
  finished : (activation.actions.get finish) = .finish worker (activation.actions.get finish).event
  sameWorker : published.worker = worker.val
  beforeFinish : SequencedBefore published (activation.actions.get finish).event

theorem ReportPublication.beforeOwnerRead {mask : List Write} {activation : Activation mask}
    (publication : ReportPublication activation)
    (retirement : Retirement history mask activation)
    (afterWait : SequencedBefore retirement.read.event ownerRead) :
    HappensBefore history publication.acquired ownerRead := by
  have released : (activation.actions.get publication.finish).release = true := by
    rw [publication.finished]
    rfl
  have acquired := SplitCompletion.handoffContract_proven history mask activation retirement
    publication.finish released
  exact .trans (.sequenced publication.afterAcquire) (.trans (.sequenced publication.afterCheck)
    (.trans (.sequenced publication.beforeFinish) (.trans acquired (.sequenced afterWait))))

structure SharedAbortRead (history : History) {writes : List Write}
    (worker owner : Read writes) (last : Nat) (wait : Event) : Prop where
  retained : RetainedThrough history worker.event wait
  beforeTest : SequencedBefore wait owner.event
  readRead : HappensBefore history worker.event owner.event → worker.source.val ≤ owner.source.val
  beforeNext : ∀ i : Fin writes.length, i.val = last + 1 →
    HappensBefore history owner.event (writes.get i).event
  readWrite : ∀ i : Fin writes.length, HappensBefore history owner.event (writes.get i).event →
    owner.source.val < i.val
  requests : ∀ i : Fin writes.length, worker.source.val < i.val → i.val ≤ last →
    ∃ prior : Fin writes.length, prior.val + 1 = i.val ∧
      (writes.get i).value = request (writes.get prior).value

def SharedAbortContract : Prop :=
  ∀ (history : History) (writes : List Write) (worker owner : Read writes) (last : Nat) (wait : Event),
    SharedAbortRead history worker owner last wait →
    Stopped (writes.get worker.source).value → Stopped (writes.get owner.source).value

theorem sharedAbortContract_proven : SharedAbortContract := by
  intro history writes worker owner last wait read stopped
  have ordered := SplitCompletion.retainedAccessContract_proven
    history worker.event wait owner.event read.retained read.beforeTest
  have lower := read.readRead ordered
  by_cases same : owner.source = worker.source
  · exact same.symm ▸ stopped
  · have later : worker.source.val < owner.source.val := by
      apply Nat.lt_of_le_of_ne lower
      exact fun eq => same (Fin.ext eq.symm)
    have beforeNext : owner.source.val ≤ last := by
      by_cases within : owner.source.val ≤ last
      · exact within
      · let next : Fin writes.length := ⟨last + 1, by have := owner.source.isLt; omega⟩
        have excluded := read.readWrite next (read.beforeNext next rfl)
        change owner.source.val < last + 1 at excluded
        omega
    obtain ⟨prior, _, wrote⟩ := read.requests owner.source later beforeNext
    rw [wrote]
    exact SearchAbort.request_stops (writes.get prior).value

structure AncestorScan (history : History) where
  event : Event
  chain : List Nat
  reads : List (FlagRead history.flags)
  walk : Walk history.flags chain reads
  afterStart : ∀ read ∈ reads, read ∈ history.flagReads ∧ SequencedBefore event read.event

def AncestorScan.stopped (scan : AncestorScan history) : Bool :=
  (scan.reads.map FlagRead.value).any id

structure AncestorRead (history : History) (flag : FlagRead history.flags)
    (owner : AncestorScan history) (wait : Event) : Prop where
  retained : RetainedThrough history flag.event wait
  beforeScan : SequencedBefore wait owner.event
  seen : flag.value = true
  target : flag.slot ∈ owner.chain
  readRead : ∀ r ∈ owner.reads, r.slot = flag.slot →
    HappensBefore history flag.event r.event → flag.source.val ≤ r.source.val
  retainedUntilClear : ∀ r ∈ owner.reads, r.slot = flag.slot →
    ∀ i : Fin (history.flags r.slot).length, flag.source.val < i.val →
      ((history.flags r.slot).get i).value = false →
      ∃ wait, RetainedThrough history r.event wait ∧
        SequencedBefore wait ((history.flags r.slot).get i).event

def AncestorStopContract : Prop :=
  ∀ (history : History) (_ : Coherent history) (flag : FlagRead history.flags)
    (owner : AncestorScan history) (wait : Event),
    AncestorRead history flag owner wait → owner.stopped = true

theorem ancestorStopContract_proven : AncestorStopContract := by
  intro history coherent flag owner wait matched
  have earlier := SplitCompletion.retainedAccessContract_proven history flag.event wait owner.event
    matched.retained matched.beforeScan
  apply List.any_eq_true.mpr
  refine ⟨true, ?_, rfl⟩
  apply CutoffVisibility.walk_finds owner.walk flag.slot matched.target
  intro read member sameSlot
  obtain ⟨recorded, afterLoad⟩ := owner.afterStart read member
  have later := matched.readRead read member sameSlot (.trans earlier (.sequenced afterLoad))
  cases read with
  | mk slot event source =>
    change slot = flag.slot at sameSlot
    subst slot
    by_cases same : source = flag.source
    · exact same.symm ▸ matched.seen
    · cases current : ((history.flags flag.slot).get source).value
      · have newer : flag.source.val < source.val := by
          apply Nat.lt_of_le_of_ne later
          exact fun eq => same (Fin.ext eq.symm)
        obtain ⟨finished, kept, after⟩ := matched.retainedUntilClear _ member rfl source newer current
        have beforeClear := SplitCompletion.retainedAccessContract_proven history _ finished _ kept after
        have excluded := coherent.flagReadWrite _ recorded source beforeClear
        exact False.elim ((Nat.lt_irrefl _) excluded)
      · exact current

structure AbortObservation where
  writes : List Write
  load : Read writes

def AbortObservation.stopped (observed : AbortObservation) : Bool :=
  let word := (observed.writes.get observed.load.source).value
  SearchAbort.generation word == SearchAbort.requested word

inductive AbortCause (history : History) (owner : AbortObservation) (wait : Event) : Prop
  | seen (worker : Read owner.writes) (last : Nat) :
      SharedAbortRead history worker owner.load last wait →
      Stopped (owner.writes.get worker.source).value → AbortCause history owner wait
  | unwind (request : Fin owner.writes.length) (last : Nat) :
      SearchAbort.FailedSplitCheck history owner.load request last → AbortCause history owner wait

private theorem AbortCause.stopped (cause : AbortCause history owner wait) : owner.stopped = true := by
  have observed : Stopped (owner.writes.get owner.load.source).value := by
    cases cause with
    | seen worker last matched stopped =>
      exact sharedAbortContract_proven history owner.writes worker owner.load last wait matched stopped
    | unwind request last matched =>
      exact SearchAbort.observeFailedSplitContract_proven history owner.writes owner.load request last matched
  simpa [AbortObservation.stopped, SearchAbort.stopped_iff] using observed

structure OwnerCheck (history : History) where
  ancestors : AncestorScan history
  shared : Option AbortObservation
  checksShared : shared = none → ancestors.stopped = true
  helperJob : Bool

def OwnerCheck.stopped (check : OwnerCheck history) : Bool :=
  check.ancestors.stopped || (check.shared.map AbortObservation.stopped).getD false || check.helperJob

inductive ExternalStop (history : History) (owner : OwnerCheck history) (wait : Event) : Prop
  | ancestor (flag : FlagRead history.flags) : Coherent history →
      AncestorRead history flag owner.ancestors wait → ExternalStop history owner wait
  | abort : (∀ obs, owner.shared = some obs → AbortCause history obs wait) →
      ExternalStop history owner wait

def CancellationGateContract : Prop :=
  ∀ (history : History) (owner : OwnerCheck history) (wait : Event),
    ExternalStop history owner wait → owner.stopped = true

theorem cancellationGateContract_proven : CancellationGateContract := by
  intro history owner wait stop
  cases stop with
  | ancestor flag coherent read =>
    have found := ancestorStopContract_proven history coherent flag owner.ancestors wait read
    simp [OwnerCheck.stopped, found]
  | abort matched =>
    cases read : owner.shared with
    | none => simp [OwnerCheck.stopped, owner.checksShared read]
    | some obs => simp [OwnerCheck.stopped, read, (matched obs read).stopped]

structure HelperJobRead (history : History) {writes : List Write} (load : Read writes)
    (clear : Fin writes.length) : Prop where
  cleared : (writes.get clear).value = 0
  published : SplitPublication.PublishedThrough history (writes.get clear).event load.event
  writeRead : ∀ i : Fin writes.length, HappensBefore history (writes.get i).event load.event →
    i.val ≤ load.source.val
  readWrite : ∀ i : Fin writes.length, HappensBefore history load.event (writes.get i).event →
    load.source.val < i.val
  laterCancellation : ∀ i : Fin writes.length, clear.val < i.val → (writes.get i).value ≠ 0 →
    ∃ (wait : Event) (messages : List Write) (sent : Fin messages.length) (received : Read messages),
      RetainedThrough history load.event wait ∧ SequencedBefore wait (messages.get sent).event ∧
        SplitShutdown.MessageRead history sent received ∧
        HappensBefore history received.event (writes.get i).event

def HelperJobReadContract : Prop :=
  ∀ (history : History) (writes : List Write) (load : Read writes) (clear : Fin writes.length),
    HelperJobRead history load clear → (writes.get load.source).value = 0

theorem helperJobReadContract_proven : HelperJobReadContract := by
  intro history writes load clear read
  have published := SplitPublication.initializationContract_proven _ _ _ read.published
  have lower := read.writeRead clear published
  by_cases stopped : (writes.get load.source).value = 0
  · exact stopped
  · have newer : clear.val < load.source.val := by
      apply Nat.lt_of_le_of_ne lower
      intro same
      exact stopped (Fin.ext same ▸ read.cleared)
    obtain ⟨wait, messages, sent, received, kept, sentAfter, message, future⟩ :=
      read.laterCancellation load.source newer stopped
    have afterRead := SearchAbort.resultBoundaryContract_proven history messages sent received
      load.event wait (writes.get load.source).event kept sentAfter message future
    exact False.elim ((Nat.lt_irrefl _) (read.readWrite load.source afterRead))

example : (SplitCutoff.poll (SplitCutoff.reset (0#64)) ⟨0#64, [true]⟩).latched = true ∧
    (SplitCutoff.poll (SplitCutoff.reset (0#64)) ⟨0xFFFFFFFFFFFFFFFF#64, [true]⟩).latched = false := by
  decide

end NeuralReversi.Threading.SplitReturn.Observation

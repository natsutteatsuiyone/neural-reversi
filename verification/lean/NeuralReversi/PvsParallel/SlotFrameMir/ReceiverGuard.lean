module
public import NeuralReversi.PvsParallel.SlotFrameMir.ReceiverMemory

@[expose] public section

namespace NeuralReversi.PvsParallel.SlotFrameMir.Receiver

open Threading
open AtomicWord (Event Write Read SequencedBefore)
open CutoffVisibility (History HappensBefore)
open SplitRebookingMir (Field Coherence)

structure Mutation where
  stamp : Event
  value : Option Slot

/-- The actual intervening writes to the worker's active Arc object. Local
    instructions carry a call record; external writes carry the target worker
    recheck and the serialization of its lock with the capture. -/
structure ArcRead (hist : History) (clock : Event → Nat) (start : Prefix hist ws w) (captured : Slot) where
  atRead : Event
  beforeFirst : SequencedBefore atRead start.first
  afterCapture : SequencedBefore start.capture atRead
  between : List Mutation
  recorded : ∀ m ∈ between, clock start.capture < clock m.stamp ∧ clock m.stamp < clock atRead
  origins : ∀ m ∈ between, (m.stamp,.effect (.replace m.value)) ∈ untilSplit start.ops ∨
    ∃ p : Posting hist ws m.stamp, Nonempty (SpinLock.OrderedAccess hist m.stamp start.capture) ∨
      Nonempty (SpinLock.OrderedAccess hist start.capture p.read.event)

def ArcRead.value {start : Prefix hist ws w} {captured : Slot} (a : ArcRead hist clock start captured) : Option Slot :=
  a.between.foldl (fun _ m => m.value) (some captured)

theorem ArcRead.same {start : Prefix hist ws w} {captured : Slot} (a : ArcRead hist clock start captured)
    (causal : ∀ x y, HappensBefore hist x y → clock x < clock y) :
    guardReceiver a.value=some captured := by
  suffices empty : a.between=[] by simp [ArcRead.value,empty,guardReceiver]
  cases split : a.between with
  | nil => rfl
  | cons m ms =>
    have mem : m∈a.between := by simp [split]
    have where_:=a.recorded m mem
    rcases a.origins m mem with localOp | ⟨posted,locked⟩
    · exact ((start.executed.untouched.1 _ _) localOp).elim
    · rcases posted.side start locked a.beforeFirst with early | late
      · have order:=causal _ _ early;omega
      · have order:=causal _ _ late;omega

abbrev Words := Slot → Field → List Write

/-- The pinned PVS/can_split MIR reads the flag and optional count from the
    state of the Arc actually selected by the worker's active Option. -/
inductive Probe (hist : History) (words : Words) (atRead first : Event) : Option Slot → Type
  | missing : Probe hist words atRead first none
  | closed (s : Slot) (r : Read (words s .accepting)) (pool depth : Nat) :
      SplitRebookingMir.nested pool depth false 0=true → ((words s .accepting).get r.source).value=0 →
      Coherence hist _ r → SequencedBefore atRead r.event → SequencedBefore r.event first →
      Probe hist words atRead first (some s)
  | counted (s : Slot) (r : Read (words s .mask)) (pool depth : Nat) :
      SplitRebookingMir.nested pool depth true (AtomicBitSet.count ((words s .mask).get r.source).value)=true →
      Coherence hist _ r → SequencedBefore atRead r.event → SequencedBefore r.event first →
      Probe hist words atRead first (some s)

structure ObjectOrder (hist : History) (words : Words) (s : Slot) (q : Recheck hist w book) (capture : Event) where
  contents : ∀ field : Field, SplitRebookingMir.Prefix (words s field) field q.mask q.acts
  published : ∀ field : Field, HappensBefore hist ((words s field).get (contents field).first).event capture
  next : ∀ field : Field, ∀ i : Fin (words s field).length, i.val=(contents field).last.val+1 →
    SplitRebookingMir.AtOrAfter hist book ((words s field).get i).event

theorem extendSB {a b c : Event} (short : SequencedBefore a b)
    (same : b.worker=c.worker) (order : b.step≤c.step) : SequencedBefore a c :=
  ⟨short.1.trans same,Nat.lt_of_lt_of_le short.2 order⟩

/-- The earliest chronological nested split suffices, including when it has
    already returned and the false store belongs to a later nested call. -/
def earlier_sample {q : Recheck hist who book} {start : Prefix hist ws w}
    (arc : ArcRead hist clock start s) (causal : ∀ x y, HappensBefore hist x y → clock x < clock y)
    (probe : Probe hist words arc.atRead start.first (guardReceiver arc.value))
    (object : ObjectOrder hist words s q start.capture) (f : WorkerFrameMir.Frame)
    (same : start.first.worker=f.entered.worker) (order : start.first.step≤f.entered.step) :
    WorkerFrameMir.GuardSample hist q.mask q.acts book f := by
  have real : Probe hist words arc.atRead start.first (some s) := arc.same causal ▸ probe
  cases real with
  | closed _ r pool d ok zero mem stamp prior =>
    exact .closed _ r (object.contents .accepting) pool d ok zero mem
      ((object.published _).trans ((HappensBefore.sequenced arc.afterCapture).trans (.sequenced stamp)))
      (extendSB prior same order) (object.next .accepting)
  | counted _ r pool d ok mem stamp prior =>
    exact .counted _ r (object.contents .mask) pool d ok mem
      ((object.published _).trans ((HappensBefore.sequenced arc.afterCapture).trans (.sequenced stamp)))
      (extendSB prior same order) (object.next .mask)

/-- Complete call/return stream of this worker's dispatch. The bridge asks
    for the first split's unlabelled active-pointer load and physical field
    arrays. It does not assume that its guard examined the booked slot. -/
structure GuardOrigins (hist : History) (clock : Event → Nat) (q : Recheck hist w book)
    (root : WorkerFrameMir.Frame) (span : WorkerFrameMir.Span root q.searching) (s : Slot) where
  ops : List EventOp
  onWorker : ∀ x ∈ ops, x.1.worker=w.val
  ordered : ops.Pairwise (fun x y => x.1.step < y.1.step)
  stack : ∀ st, WorkerFrameMir.Record.localWrite st ∈ span.records → ∀ f,
    WorkerFrameMir.FirstInner st.frames f → StackCut ops st.frames
  observed : ∀ first, firstSplit ops=some first →
    ∃ (start : Prefix hist q.searching w.val) (arc : ArcRead hist clock start s) (words : Words),
      start.ops=ops ∧ start.first=first ∧ Nonempty (ObjectOrder hist words s q start.capture) ∧
      Nonempty (Probe hist words arc.atRead start.first (guardReceiver arc.value))

noncomputable def GuardOrigins.sample {q : Recheck hist w book} {root : WorkerFrameMir.Frame}
    {span : WorkerFrameMir.Span root q.searching} (origin : GuardOrigins hist clock q root span s)
    (causal : ∀ x y, HappensBefore hist x y → clock x < clock y) (st)
    (member : WorkerFrameMir.Record.localWrite st ∈ span.records) (f)
    (nested : WorkerFrameMir.FirstInner st.frames f) : WorkerFrameMir.GuardSample hist q.mask q.acts book f := by
  classical
  apply Classical.choice
  have loc:=(origin.stack st member f nested).contains nested
  obtain ⟨earliest,first,le⟩:=first_before origin.ops Event.step origin.ordered loc
  obtain ⟨start,arc,words,_eq,atFirst,⟨object⟩,⟨probe⟩⟩:=origin.observed earliest first
  exact ⟨earlier_sample arc causal probe object f
    (start.onThread.trans (origin.onWorker (f.entered,.call .split) loc).symm) (atFirst ▸ le)⟩

noncomputable def GuardOrigins.entered {q : Recheck hist w book} {root : WorkerFrameMir.Frame}
    {span : WorkerFrameMir.Span root q.searching} (origin : GuardOrigins hist clock q root span s)
    (causal : ∀ x y, HappensBefore hist x y → clock x < clock y) (anchor : EntryAnchor q root span)
    (finish : ∀ st, WorkerFrameMir.Record.localWrite st ∈ span.records → ∀ d, st.frames=[d] → d.kind=.detach →
      ∃ sec ∈ q.order.prior++q.order.after, ∃ e, sec.action=.finish w e ∧ SequencedBefore sec.lock st.time)
    (abort : ∀ st, WorkerFrameMir.Record.localWrite st ∈ span.records → ∀ d parent rest,
      st.frames=d::parent::rest → d.kind=.detach → parent.kind=.abandon →
      ∃ i : Fin q.faults.length, SequencedBefore (q.faults.get i) st.time) : EnteredFrame q :=
  ⟨root,span,anchor,⟨origin.sample causal,finish,abort⟩⟩

end NeuralReversi.PvsParallel.SlotFrameMir.Receiver

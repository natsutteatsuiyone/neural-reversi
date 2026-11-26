import NeuralReversi.PvsParallel.SlotFrameMir.Dispatch
import NeuralReversi.PvsParallel.SlotFrameMir.Previous

namespace NeuralReversi.PvsParallel.SlotFrameMir

open Threading
open AtomicWord (Event Read Write SequencedBefore ReadWindow InReleaseSequence)
open CutoffVisibility (History HappensBefore)
open SplitCompletion (MaskState Action)
open SplitRebookingMir (Coherence)

/-- Registers of a successful target-thread-locked can_join recheck, its slot
    actions in mutex order, and the two atomic read sources. -/
structure Recheck (hist : History) (w : Worker) (book : Event) where
  mask : MaskState
  acts : List Action
  searching : List Write
  busy : Read searching
  coherent : Coherence hist searching busy
  acquire : busy.acquire=true
  sync : ∀ i, busy.acquire=true → InReleaseSequence searching i busy.source →
    hist.otherSync (searching.get i).event busy.event
  beforeBook : SequencedBefore busy.event book
  birth : Event
  faults : List Event
  failed : Read (WorkerFrameMir.failures birth faults)
  failMemory : Coherence hist _ failed
  afterRead : SequencedBefore busy.event failed.event
  order : WorkerFrameMir.SlotOrder hist acts book
  late : Bool
  opened : (SplitCompletion.run mask acts).accepting=true
  capacity : SplitRebookingMir.room late (AtomicBitSet.count (SplitCompletion.run mask acts).bits)=true
  hasSlot : Bool
  deepest : Bool
  tested : Bool
  grants : WorkerFrameMir.canJoin (decide ((searching.get busy.source).value=1))
    (decide (((WorkerFrameMir.failures birth faults).get failed.source).value=1)) hasSlot deepest tested=true

/-- The opening busy-word source is the actual idle-loop Acquire, which need
    not have observed the original booking. The target-worker lock orders the
    successful active-pointer snapshot before the competing locked recheck. -/
structure EntryAnchor (q : Recheck hist w book) (root : WorkerFrameMir.Frame)
    (span : WorkerFrameMir.Span root q.searching) where
  guard : Read q.searching
  index : guard.source=span.first
  release : (q.searching.get guard.source).release=true
  acquire : guard.acquire=true
  sync : ∀ i, guard.acquire=true → InReleaseSequence q.searching i guard.source →
    hist.otherSync (q.searching.get i).event guard.event
  captured : Event
  entered : SequencedBefore guard.event captured
  order : SpinLock.OrderedAccess hist captured q.busy.event
  next : ∀ i : Fin q.searching.length, i.val=span.last.val+1 →
    SequencedBefore q.busy.event (q.searching.get i).event ∨
    Nonempty (SpinLock.OrderedAccess hist q.busy.event (q.searching.get i).event)

theorem EntryAnchor.range {q : Recheck hist w book} {root : WorkerFrameMir.Frame}
    {span : WorkerFrameMir.Span root q.searching} (a : EntryAnchor q root span) :
    ReadWindow q.busy (HappensBefore hist) span.first span.last.val := by
  have seq : InReleaseSequence q.searching a.guard.source a.guard.source :=
    ⟨a.release,Nat.le_refl _,fun _ lower upper => by omega⟩
  have fromGuard := HappensBefore.other (a.sync _ a.acquire seq)
  refine ⟨?_,?_,q.coherent.writeRead,q.coherent.readWrite⟩
  · rw [← a.index]
    exact fromGuard.trans ((HappensBefore.sequenced a.entered).trans a.order.orders)
  · intro i pos
    rcases a.next i pos with same | locked
    · exact .sequenced same
    · exact locked.elim fun order => order.orders

/-- The dynamic frame supplies its searching stores and call stacks, the first
    nested can_split guard and the detach mutex sections. -/
structure EnteredFrame (q : Recheck hist w book) where
  root : WorkerFrameMir.Frame
  span : WorkerFrameMir.Span root q.searching
  anchor : EntryAnchor q root span
  events : WorkerFrameMir.Bindings hist root span q.mask q.acts w book q.faults q.order

theorem Recheck.rejects_entered (q : Recheck hist w book) (frame : EnteredFrame q) : False := by
  have rejected := WorkerFrameMir.no_booking_from_frame hist q.mask q.acts w frame.root q.searching
    frame.span q.busy book frame.anchor.range q.coherent q.acquire q.sync q.beforeBook q.birth q.faults
    q.failed q.failMemory q.afterRead q.order frame.events q.late q.opened q.capacity
  have yes := (Bool.and_eq_true_iff.mp q.grants).1
  rw [rejected] at yes
  cases yes

abbrev Tick := Event × Instruction

/-- Most recent instruction at the head. -/
def machine : List Tick → Resources
  | [] => .zero
  | (e,op)::past => (machine past).exec e op

private theorem untouched (past : List Tick) (s : Slot) (none : ∀ x ∈ past, x.2.maskSlot ≠ some s) :
    (machine past).mask s=0 := by
  induction past with
  | nil => rfl
  | cons x xs ih =>
    obtain ⟨e,op⟩:=x
    exact ((machine xs).unmodified e op (none (e,op) (by simp))).trans
      (ih (fun x mem => none x (List.mem_cons_of_mem _ mem)))

inductive Source (hist : History) (past : List Tick) (e : Event) : Instruction → Type
  | first (s : Slot) : (∀ x ∈ past, x.2.maskSlot ≠ some s) → Source hist past e (.openSlot s)
  | reused (s : Slot) : Previous hist ((machine past).mask s) → Source hist past e (.openSlot s)
  | reserve (s : Slot) (w : Worker) (q : Recheck hist w e) :
      (s ∈ (machine past).frames w → EnteredFrame q) → Source hist past e (.reserve s w)
  | capture (s : Slot) (w : Worker) (initial : Pointer) (posts : List IdlePost) :
      IdleEntry hist initial posts → (machine past).active w=activeAfter initial posts →
      (activeAfter initial posts).address=some s → Source hist past e (.capture s w)
  | restore (parent : Option Slot) (w : Worker) : Source hist past e (.restore parent w)
  | release (s : Slot) (w : Worker) (taken : Option Slot) :
      workerReset ((machine past).frames w).head? taken=some s → Source hist past e (.release s w)
  | abandon (w : Worker) (taken : Option Slot) :
      taken=((machine past).active w).address →
      (∀ s, unwindReset none taken=some s → Nonempty (OwnerRead hist s.1 w)) →
      Source hist past e (.abandon taken w)

theorem Source.checked (src : Source hist past e op) (acyclic : ∀ ev, ¬HappensBefore hist ev ev) :
    AtInstruction (machine past) op := by
  cases src with
  | first s fresh => exact untouched past s fresh
  | reused s prev => exact prev.empty
  | reserve s w q resolve => exact fun mem => q.rejects_entered (resolve mem)
  | capture s w init posts origin current addr => exact current.trans (dispatch_fresh origin _ addr acyclic)
  | restore => trivial
  | release s w taken read => exact read
  | abandon w taken _ protects => exact fun s eq => (protects s eq).elim fun read => read.same

/-- One linear extension of the raw causality graph. Local call/return order
    fixes frame operations; receiver mutex order fixes competing bookings. -/
inductive Scheduled (hist : History) (clock : Event → Nat) : List Tick → Prop
  | empty : Scheduled hist clock []
  | step : Scheduled hist clock past → Source hist past e op →
      (∀ x ∈ past, clock x.1 < clock e) → Scheduled hist clock ((e,op)::past)

theorem Scheduled.retains (trace : Scheduled hist clock past)
    (causal : ∀ x y, HappensBefore hist x y → clock x < clock y) : (machine past).Retains := by
  induction trace with
  | empty => exact initial_retains
  | step _ src _ ih =>
    exact Resources.preserves _ ih _ _ (src.checked (fun x cyc => Nat.lt_irrefl _ (causal x x cyc)))

/-- Every registered worker that has entered the subtree stays visible to the
    retiring owner. This conclusion permits duplicate unconsumed reservations. -/
theorem Scheduled.no_live_at_zero (trace : Scheduled hist clock past)
    (causal : ∀ x y, HappensBefore hist x y → clock x < clock y)
    (s : Slot) (w : Worker) (zero : (machine past).mask s=0) : s ∉ (machine past).frames w := by
  intro present
  have bit:=(trace.retains causal).2.1 s w present
  rw [zero] at bit;simp at bit

theorem Scheduled.capture_bit (trace : Scheduled hist clock past)
    (causal : ∀ x y, HappensBefore hist x y → clock x < clock y)
    (entry : Source hist past e (.capture s w)) : ((machine past).mask s).getLsbD w.val=true := by
  have ready:=entry.checked (fun x cyc => Nat.lt_irrefl _ (causal x x cyc))
  exact ((trace.retains causal).1 s w ready).1

theorem Scheduled.reuse_has_no_older_worker (trace : Scheduled hist clock past)
    (causal : ∀ x y, HappensBefore hist x y → clock x < clock y)
    (init : Source hist past e (.openSlot s)) (w : Worker) : s ∉ (machine past).frames w :=
  trace.no_live_at_zero causal s w (init.checked (fun x cyc => Nat.lt_irrefl _ (causal x x cyc)))

theorem Scheduled.normal_release_is_set (trace : Scheduled hist clock past)
    (causal : ∀ x y, HappensBefore hist x y → clock x < clock y)
    (finish : Source hist past e (.release s w)) : ((machine past).mask s).getLsbD w.val=true := by
  have head:=finish.checked (fun x cyc => Nat.lt_irrefl _ (causal x x cyc))
  obtain ⟨tail,same⟩:=List.head?_eq_some_iff.mp head
  exact (trace.retains causal).2.1 s w (by rw [same];simp)

end NeuralReversi.PvsParallel.SlotFrameMir

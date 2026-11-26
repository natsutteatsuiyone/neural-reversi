import NeuralReversi.PvsParallel.Interleaving.NestedRetention
import NeuralReversi.PvsParallel.Interleaving.TaskAncestry

namespace NeuralReversi.PvsParallel.Interleaving.TaskRetention

open Threading
open AtomicWord (Event SequencedBefore)
open CutoffVisibility (History HappensBefore FlagRead)
open SplitCompletion (Action RetainedThrough)
open TaskAncestry (Generation)

private def put [DecidableEq α] (f : α → β) (k : α) (v : β) (x : α) := if x=k then v else f x

structure Dispatch where
  gen : Generation
  entered : Event
  deriving DecidableEq

structure MaskOp where
  event : Event
  participant : Nat
  booking : Bool
  oldWord : BitVec 64
  newWord : BitVec 64
  deriving DecidableEq

/-- The copyParent register names an enclosing call of dispatch_search. The
    worker executing the inner split may have been booked by another worker. -/
structure State where
  born : Nat → Option Generation
  active : Nat → Option Generation
  calls : Nat → List Dispatch
  copied : Nat → Option (Nat × Option Dispatch)
  links : Generation → Option (Option Dispatch)
  mods : Generation → List MaskOp
  previous : Nat → Option Nat

def empty : State := ⟨fun _ => none,fun _ => none,fun _ => [],fun _ => none,fun _ => none,fun _ => [],fun _ => none⟩

abbrev Address := Nat → Nat → Option Nat

private def task (s : State) (e : Event) : Joint.TaskWord → Option State
  | .copyParent slot word => do
    let ptr := s.active e.worker
    guard (ptr.map Generation.slot == word)
    let enclosing ← match ptr with
      | none => some none
      | some g => (s.calls e.worker).find? (fun c => c.gen == g) |>.map some
    return {s with copied := put s.copied e.worker (some (slot,enclosing))}
  | .install slot word => do
    let (building,call) ← s.copied e.worker
    let g ← s.born slot
    guard (building == slot && g.mark.worker == e.worker && call.map (fun c => c.gen.slot) == word)
    return {s with links := put s.links g (some call), copied := put s.copied e.worker none}
  | .ownerActive slot => do
    let g ← s.born slot
    return {s with active := put s.active e.worker (some g)}
  | .book helper slot => do
    let g ← s.born slot
    return {s with active := put s.active helper (some g)}
  | .detach old => do
    guard ((s.active e.worker).map Generation.slot == some old)
    return {s with active := put s.active e.worker none}
  | .detachEmpty => do
    guard ((s.active e.worker).isNone)
    return s
  | .restore slot word => do
    let g ← s.born slot
    let call ← s.links g
    guard (g.mark.worker == e.worker && call.map (fun c => c.gen.slot) == word)
    return {s with active := put s.active e.worker (call.map Dispatch.gen)}
  | _ => some s

private def step (addr : Address) (s : State) : Joint.Event Op → Option State
  | .taskWord w t op => task s ⟨w,t⟩ op
  | .splitWord w t (.mark k) => do
    let slot ← addr w k
    let g := ⟨slot,⟨w,t⟩⟩
    return {s with born := put s.born slot (some g)}
  | .splitWord w t (.maskRmw owner k who book old stored) => do
    let slot ← addr owner k
    let g ← s.born slot
    return {s with mods := put s.mods g (s.mods g ++ [⟨⟨w,t⟩,who,book,old,stored⟩])}
  | .engine (.localCall w (.entry t _ .dispatch ..) _) => do
    let g ← s.active w
    return {s with calls := put s.calls w (⟨g,⟨w,t⟩⟩ :: s.calls w)}
  | .engine (.localCall w (.leave _ opening .dispatch ..) _) => do
    let c :: tail := s.calls w | none
    guard (c.entered == ⟨w,opening⟩)
    return {s with calls := put s.calls w tail}
  | _ => some s

private def site : Joint.Event Op → Option Event
  | .taskWord w t _ | .splitWord w t _ => some ⟨w,t⟩
  | .engine (.localCall w (.entry t _ .dispatch ..) _)
  | .engine (.localCall w (.leave t _ .dispatch ..) _) => some ⟨w,t⟩
  | _ => none

private def tick (addr : Address) (s : State) (row : Joint.Event Op) : Option State := do
  let some e := site row | return s
  guard ((s.previous e.worker).all (· < e.step))
  let next ← step addr s row
  return { next with previous := put next.previous e.worker (some e.step) }

def capture (addr : Address) (es : Joint.Tape Op) : Option State := es.foldlM (tick addr) empty

def atEvent (addr : Address) (clock : Event → Nat) (es : Joint.Tape Op) (e : Event) :=
  capture addr (es.filter fun row => (site row).any (fun s => clock s ≤ clock e))

private def actor : Action → Nat
  | .book w _ | .finish w _ => w.val

def physical (g : NestedRetention.Generation h) : List MaskOp :=
  List.ofFn fun i : Fin g.mask.actions.length =>
    let a := g.mask.actions.get i
    ⟨a.event,actor a,!a.release,(g.writes.get (g.mask.before i)).value,(g.writes.get (g.mask.at i)).value⟩

def membership (w : Nat) (was : Bool) (action : MaskOp) : Bool :=
  if action.participant == w then action.booking else was

/-- The last real booking/reset of this worker determines its bit even when
    the raw RMW list contains repeated bookings or redundant resets. -/
private theorem membership_at (g : NestedRetention.Generation h) (w : SplitCompletion.Worker)
    (n : Nat) (bound : n ≤ g.mask.actions.length) :
    ((physical g).take n).foldl (membership w.val) (w == g.mask.owner) =
      (g.writes.get ⟨g.mask.mark.val+n,by have := g.mask.bounded; omega⟩).value.getLsbD w.val := by
  induction n with
  | zero =>
    change (w == g.mask.owner) = (g.writes.get g.mask.mark).value.getLsbD w.val
    rw [g.mask.initial,Bitboard.squareMask_bit]; by_cases eq : w=g.mask.owner <;> simp [eq,Fin.val_inj]
  | succ n ih =>
    have lt : n < g.mask.actions.length := by omega
    have physicalBound : n < (physical g).length := by simpa [physical] using lt
    rw [List.take_succ_eq_append_getElem physicalBound,List.foldl_append,List.foldl_cons,List.foldl_nil,ih (by omega)]
    let i : Fin g.mask.actions.length := ⟨n,lt⟩
    have update := (g.mask.rmw i).2.2.2
    change (g.writes.get (g.mask.at i)).value = (g.mask.actions.get i).updateBits (g.writes.get (g.mask.before i)).value at update
    change membership w.val ((g.writes.get (g.mask.before i)).value.getLsbD w.val) ((physical g)[n]) =
      (g.writes.get (g.mask.at i)).value.getLsbD w.val
    rw [update]
    simp only [physical,List.getElem_ofFn,membership]
    change (if (actor (g.mask.actions.get i) == w.val) = true then !(g.mask.actions.get i).release else _) = _
    let state : SplitCompletion.MaskState := ⟨(g.writes.get (g.mask.before i)).value,true⟩
    cases action : g.mask.actions.get i with
    | book who e =>
      change (if (who.val == w.val) = true then true else state.has w) =
        (SplitCompletion.step state (.book who e)).has w
      rw [SplitCompletion.book_has]
      by_cases same : w=who
      · subst who; simp
      · simp [same, Fin.val_ne_of_ne (Ne.symm same)]
    | finish who e =>
      change (if (who.val == w.val) = true then false else state.has w) =
        (SplitCompletion.step state (.finish who e)).has w
      rw [SplitCompletion.finish_has]
      by_cases same : w=who
      · subst who; simp
      · simp [same, Fin.val_ne_of_ne (Ne.symm same)]

/-- Physics for each address/generation consists of its mask modification order
    and its own final acquire, constructed by NestedRetention.ownerRetirement. -/
structure Input (h : History) where
  address : Address
  clock : Event → Nat
  generation : Generation → Option (NestedRetention.Generation h)

structure Physics (h : History) where
  slot : SlotFrameMir.Slot
  writes : List AtomicWord.Write
  mask : SlotFrameMir.RawMask writes
  gates : SlotFrameMir.GateHistory h mask
  exit : SlotFrameMir.OwnerExit h mask
  mark : (writes.get mask.mark).event = exit.clock.atStep 1
  sync : ∀ pre src post, exit.log = pre ++ .mask src :: post → ∀ i,
    AtomicWord.InReleaseSequence writes i src →
      h.otherSync (writes.get i).event (exit.clock.atStep (4 + pre.length))
  actors : ∀ i w e, mask.actions.get i = .finish w e → e.worker = w.val

def Physics.generation (p : Physics h) : NestedRetention.Generation h :=
  ⟨p.slot,p.writes,p.mask,p.gates.closed,
    NestedRetention.retire p.exit p.gates.closed p.mark p.sync,p.actors⟩

theorem Physics.wait (p : Physics h) :
    p.generation.wait = p.exit.clock.atStep (3 + p.exit.log.length) :=
  rfl

structure Credit (clock : Event → Nat) (g : NestedRetention.Generation h) (inner : Event) where
  who : SplitCompletion.Worker
  actorAt : inner.worker = who.val
  enrolled : ((physical g).take (NestedRetention.pastUpdates clock inner g.mask.actions)).foldl
    (membership who.val) (who == g.mask.owner) = true

theorem Credit.pass {g : NestedRetention.Generation h} (c : Credit clock g inner)
    (causal : ∀ x y, HappensBefore h x y → clock x < clock y)
    (kept : RetainedThrough h access inner) : RetainedThrough h access g.wait := by
  let n := NestedRetention.pastUpdates clock inner g.mask.actions
  let cut := NestedRetention.currentCut g.mask clock inner
  have bound : n ≤ g.mask.actions.length := by
    have := cut.endAt
    change g.mask.mark.val + n ≤ g.mask.mark.val + g.mask.actions.length at this
    omega
  exact g.pass causal cut c.who c.actorAt ((membership_at g c.who n bound).symm.trans c.enrolled) kept

private def sameGeneration (i : Input h) (parent : Generation) (g : NestedRetention.Generation h) : Bool :=
  i.address g.slot.1.val g.slot.2 == some parent.slot && g.slot.1 == g.mask.owner &&
    parent.mark == (g.writes.get g.mask.mark).event && parent.mark.worker == g.mask.owner.val

/-- Match the parent call captured by the child's copyParent, the native
    dispatch stack at the scan/wait, and every physical RMW of this allocation. -/
def credit (i : Input h) (es : Joint.Tape Op) (full : State) (child parent : Generation)
    (g : NestedRetention.Generation h) (inner : Event) : Option (Credit i.clock g inner) := do
  guard (sameGeneration i parent g)
  let some source ← full.links child | none
  guard (source.gen == parent && source.entered.worker == inner.worker && inner.worker == child.mark.worker)
  let at_ ← atEvent i.address i.clock es inner
  guard ((at_.calls inner.worker).contains source && at_.born parent.slot == some parent)
  let all := physical g
  let n := NestedRetention.pastUpdates i.clock inner g.mask.actions
  let prefix_ := at_.mods parent
  if routed : prefix_ = all.take n then
    guard (full.mods parent == all)
    if inRange : inner.worker < 64 then
      let who : SplitCompletion.Worker := ⟨inner.worker,inRange⟩
      if set : prefix_.foldl (membership who.val) (who == g.mask.owner) = true then
        return ⟨who,rfl,routed ▸ set⟩
  none

inductive Route (h : History) (clock : Event → Nat) : Event → Event → Type
  | done (e) : Route h clock e e
  | hop {g : NestedRetention.Generation h} {inner last} : Credit clock g inner → Route h clock g.wait last → Route h clock inner last

theorem Route.held (route : Route h clock inner last)
    (causal : ∀ x y, HappensBefore h x y → clock x < clock y)
    (kept : RetainedThrough h access inner) : RetainedThrough h access last := by
  induction route with
  | done => exact kept
  | hop c _ ih => exact ih (c.pass causal kept)

/-- Consume the actual restored parent list, stopping at the generation that
    owned the flag load. Each subsequent hop starts at the preceding owner wait. -/
def route (i : Input h) (es : Joint.Tape Op) (full : State) (target : Generation) (last : Event)
    (child : Generation) (inner : Event) : TaskAncestry.Path → Option (Route h i.clock inner last)
  | [] => none
  | p :: ps => do
    let g ← i.generation p
    let c ← credit i es full child p g inner
    if p == target then
      if eq : g.wait = last then return .hop c (eq ▸ Route.done g.wait)
      else none
    else return .hop c (← route i es full target last p g.wait ps)

structure Era (h : History) (r : FlagRead h.flags) (g : NestedRetention.Generation h) where
  flags : StopSchedule.Flags h r.slot
  initialized : HappensBefore h flags.clear r.event
  next : ∀ x ∈ flags.future, x.value = false → SequencedBefore g.wait x.event

private structure Prepared (child : Generation) where
  tasks : TaskAncestry.Journal
  view : TaskAncestry.Window child
  full : State

private def fromLoad (v : TaskAncestry.View) (r : FlagRead flags) (row : TaskAncestry.Read) : Bool :=
  row.view.path == v.path && row.view.reset == v.reset && row.field.slot == r.slot &&
    v.path.contains row.field && row.word == r.value && row.source == ((flags r.slot).get r.source).event

private def prepare (i : Input h) (es : Joint.Tape Op) (child : Generation) (poll : Event)
    (scan : SplitReturn.Observation.AncestorScan h) : Option (Prepared child) := do
  let tasks ← TaskAncestry.capture i.address es
  let w ← TaskAncestry.window tasks child poll scan.event
  let walk ← tasks.walks scan.event
  guard (w.parents.map Generation.slot == scan.chain && walk.view.path == w.owner.path &&
    walk.view.reset == w.owner.reset && walk.reverseReads.reverse == scan.reads.map (·.event) &&
    walk.found == scan.stopped &&
    scan.reads.all (fun r => (tasks.loads r.event).any (fromLoad w.owner r)))
  return ⟨tasks,w,← capture i.address es⟩

/-- The selected generation comes from the load address and the saved task
    path. Reads-from is left to coherence; it need not identify the generation. -/
private def one (i : Input h) (es : Joint.Tape Op) (p : Prepared child)
    (scan : SplitReturn.Observation.AncestorScan h) (r : FlagRead h.flags)
    (era : (g : NestedRetention.Generation h) → Option (Era h r g))
    (causal : ∀ x y, HappensBefore h x y → i.clock x < i.clock y) :
    Option (StopSchedule.Parent h scan r) := do
  let load ← p.tasks.loads r.event
  guard (fromLoad p.view.view r load && p.view.parents.contains load.field)
  let key := load.field
  let ancestor ← i.generation key
  guard (sameGeneration i key ancestor && p.full.mods key == physical ancestor)
  let flags ← era ancestor
  let checks (x : FlagRead h.flags) := route i es p.full key ancestor.wait child x.event p.view.parents
  if ok : scan.reads.all (fun x => x.slot != r.slot || (checks x).isSome) then
    return { era := flags.flags, initialized := flags.initialized, endAt := ancestor.wait, next := flags.next
             held := fun x mem same => ((checks x).get (by
               have checked := List.all_eq_true.mp ok x mem
               simpa [same] using checked)).held causal .here }
  none

def parent (i : Input h) (es : Joint.Tape Op) (child : Generation) (poll : Event)
    (scan : SplitReturn.Observation.AncestorScan h) (r : FlagRead h.flags)
    (era : (g : NestedRetention.Generation h) → Option (Era h r g))
    (causal : ∀ x y, HappensBefore h x y → i.clock x < i.clock y) :
    Option (StopSchedule.Parent h scan r) := do
  one i es (← prepare i es child poll scan) scan r era causal

/-- Producer of StopSchedule.PollContext.parentEra from one joint tape. -/
def parentEra (i : Input h) (es : Joint.Tape Op) (child : Generation) (poll : Event)
    (scan : SplitReturn.Observation.AncestorScan h) (reads : List (FlagRead h.flags))
    (eras : ∀ r : FlagRead h.flags, (g : NestedRetention.Generation h) → Option (Era h r g))
    (causal : ∀ x y, HappensBefore h x y → i.clock x < i.clock y) :
    Option (∀ r ∈ reads, r.slot ∈ scan.chain → NativeStopReturns.InWorker poll r.event →
      StopSchedule.Parent h scan r) := do
  let p ← prepare i es child poll scan
  let get (r : FlagRead h.flags) := one i es p scan r (eras r) causal
  if all : reads.all (fun r => !(scan.chain.contains r.slot && r.event.worker == poll.worker &&
      decide (r.event.step ≤ poll.step)) || (get r).isSome) then
    return fun r mem slot onWorker => (get r).get (by
      have checked := List.all_eq_true.mp all r mem
      simpa [slot,onWorker.1,onWorker.2] using checked)
  none

namespace Regression

open SlotFrameMir RawProjection

private def addr : Address := fun owner bank => some (64*owner+bank)
private def a : Generation := ⟨0,⟨0,2⟩⟩
private def b : Generation := ⟨64,⟨1,12⟩⟩
private def c : Generation := ⟨128,⟨2,22⟩⟩

private def push (w t : Nat) : Joint.Event Unit :=
  .engine (.localCall w (.entry t 0 .dispatch .none none) none)
private def pop (w t opening : Nat) : Joint.Event Unit :=
  .engine (.localCall w (.leave t opening .dispatch none none) none)
private abbrev tw (w t : Nat) (op : Joint.TaskWord) : Joint.Event Unit := .taskWord w t op
private abbrev sw (w t : Nat) (op : Joint.SplitWord) : Joint.Event Unit := .splitWord w t op

def tape : Joint.Tape Unit :=
  [tw 0 1 (.copyParent 0 none),sw 0 2 (.mark 0),tw 0 3 (.install 0 none),tw 0 4 (.ownerActive 0),
   sw 0 5 (.maskRmw 0 0 1 true 1 3),tw 0 6 (.book 1 0),push 0 7,pop 0 8 7,tw 0 9 (.detach 0),
   sw 0 10 (.maskRmw 0 0 0 false 3 2),push 1 10,tw 1 11 (.copyParent 64 (some 0)),sw 1 12 (.mark 0),
   tw 1 13 (.install 64 (some 0)),tw 1 14 (.ownerActive 64),sw 1 15 (.maskRmw 1 0 2 true 2 6),
   tw 1 16 (.book 2 64),push 1 17,pop 1 18 17,tw 1 19 (.detach 64),sw 1 20 (.maskRmw 1 0 1 false 6 4),
   push 2 20,tw 2 21 (.copyParent 128 (some 64)),sw 2 22 (.mark 0),tw 2 23 (.install 128 (some 64)),
   tw 2 24 (.ownerActive 128),push 2 25,tw 2 26 .poll,pop 2 27 25,tw 2 28 (.detach 128),sw 2 29 (.maskRmw 2 0 2 false 4 0),
   tw 2 30 (.restore 128 (some 64)),tw 2 31 (.clearTask 128),tw 2 32 (.afterSplit (some 64)),
   tw 2 33 (.loadFlag 64 false ⟨1,14⟩),tw 2 34 (.loadFlag 0 false ⟨0,4⟩),tw 2 35 (.endWalk false),
   pop 2 48 20,tw 2 49 (.detach 64),sw 2 50 (.maskRmw 1 0 2 false 4 0),
   tw 1 56 (.restore 64 (some 0)),tw 1 57 (.clearTask 64),pop 1 70 10,
   tw 1 71 (.detach 0),sw 1 75 (.maskRmw 0 0 1 false 2 0)]

private def edges : List (Event × Event) :=
  [(⟨0,10⟩,⟨0,80⟩),(⟨1,75⟩,⟨0,80⟩),(⟨1,20⟩,⟨1,55⟩),(⟨2,50⟩,⟨1,55⟩)]
private def history : History := ⟨[],fun _ => [],[],[],fun x y => (x,y) ∈ edges⟩

private theorem time {x y : Event} (hb : HappensBefore history x y) : x.step < y.step := by
  induction hb with
  | sequenced sb => exact sb.2
  | synchronized i => exact Fin.elim0 i
  | other mem =>
    have ok : edges.all (fun p => decide (p.1.step < p.2.step)) := by decide
    exact of_decide_eq_true (List.all_eq_true.mp ok _ mem)
  | trans _ _ first later => exact Nat.lt_trans first later

private def checkedGeneration (raw : RawMask ws) (closed : ClosedAfterFirst raw)
    (actors : ∀ j w e, raw.actions.get j = .finish w e → e.worker = w.val)
    (wait : Event) (src : Fin (words raw).length) (zero : ((words raw).get src).value = 0)
    (last : ∀ j : Fin (words raw).length, j.val ≤ src.val)
    (after : SequencedBefore ((words raw).get (activation raw closed).start).event wait)
    (earlier : ∀ j : Fin (words raw).length, ((words raw).get j).event.step < wait.step)
    (sync : ∀ j : Fin (words raw).length, ((words raw).get j).release=true →
      (((words raw).get j).event,wait) ∈ edges) : NestedRetention.Generation history where
  slot := (raw.owner,0)
  writes := ws
  mask := raw
  actors := actors
  closed := closed
  waited := {
    read := ⟨wait,src,true⟩
    acquire := rfl
    zero := zero
    afterOwnerBit := after
    beforeReuse := fun i eq => by
      change i.val = 0+(actions raw).length+1 at eq
      have bound := Nat.lt_of_lt_of_eq i.isLt (length_words raw)
      rw [length_actions] at eq
      omega
    writeRead := fun i _ => last i
    readWrite := fun i hb => False.elim ((Nat.not_lt_of_ge (Nat.le_of_lt (earlier i))) (time hb))
    synchronize := fun i _ seq => sync i seq.1 }

private def gen0 : Mask.Generation :=
  ⟨fun n => 1+n, fun _ _ lt => by omega, [.book 1 ⟨0,5⟩,.finish 0 ⟨0,10⟩,.finish 1 ⟨1,75⟩]⟩
private def gen1 : Mask.Generation :=
  ⟨fun n => 11+n, fun _ _ lt => by omega, [.book 2 ⟨1,15⟩,.finish 1 ⟨1,20⟩,.finish 2 ⟨2,50⟩]⟩

private theorem actors_of (raw : RawMask ws)
    (ok : ∀ j : Fin raw.actions.length, (raw.actions.get j).release = true →
      (raw.actions.get j).event.worker = actor (raw.actions.get j)) :
    ∀ j w e, raw.actions.get j = .finish w e → e.worker=w.val := by
  intro j w e found
  have eq := ok j (by rw [found];rfl)
  rw [found] at eq; exact eq

private def outer : NestedRetention.Generation history :=
  checkedGeneration (Mask.raw [] 0 gen0 []) (by unfold ClosedAfterFirst; decide) (actors_of _ (by decide))
    ⟨0,80⟩ ⟨3,by decide⟩ (by decide) (by decide) (by unfold SequencedBefore; decide) (by decide) (by decide)
private def middle : NestedRetention.Generation history :=
  checkedGeneration (Mask.raw [] 1 gen1 []) (by unfold ClosedAfterFirst; decide) (actors_of _ (by decide))
    ⟨1,55⟩ ⟨3,by decide⟩ (by decide) (by decide) (by unfold SequencedBefore; decide) (by decide) (by decide)

private def inputs : Input history := ⟨addr, Event.step,fun g =>
  if g=a then some outer else if g=b then some middle else none⟩

private def check (es : Joint.Tape Unit) (target : Generation) (finish site : Event) (path : List Generation) :=
  (capture addr es).bind fun full => route inputs es full target finish c site path

private def recovered := do
  let tasks ← TaskAncestry.capture addr tape
  let w ← TaskAncestry.window tasks c ⟨2,26⟩ ⟨2,32⟩
  check tape a ⟨0,80⟩ ⟨2,34⟩ w.parents

theorem through_two_owners : RetainedThrough history ⟨2,34⟩ ⟨0,80⟩ := by
  have joined : recovered.isSome := by decide +kernel
  exact (recovered.get joined).held (fun _ _ => time) .here

private def missingBook := tape.filter fun row => match row with
  | .splitWord 1 15 (.maskRmw ..) => false
  | _ => true

private def wrongOld := tape.map fun row => match row with
  | .splitWord 1 15 (.maskRmw 1 0 2 true _ stored) => sw 1 15 (.maskRmw 1 0 2 true 0 stored)
  | _ => row

/-- At step 49 the booking bit is still set, but dispatch 20 returned at step
    48. Reuse or skipping the immediate parent also cannot credit this read. -/
theorem rejected :
    (check tape a ⟨0,80⟩ ⟨2,49⟩ [b,a]).isNone ∧
    (check missingBook a ⟨0,80⟩ ⟨2,34⟩ [b,a]).isNone ∧
    (check wrongOld a ⟨0,80⟩ ⟨2,34⟩ [b,a]).isNone ∧
    (check tape a ⟨0,80⟩ ⟨2,34⟩ [a]).isNone ∧
    (check tape a ⟨0,80⟩ ⟨2,34⟩ [⟨64,⟨1,112⟩⟩,a]).isNone := by
  decide +kernel

end Regression

end NeuralReversi.PvsParallel.Interleaving.TaskRetention

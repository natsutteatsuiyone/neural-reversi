import NeuralReversi.PvsParallel.Interleaving.CutoffPollReplay
import NeuralReversi.PvsParallel.Interleaving.SlotSynchronization

namespace NeuralReversi.PvsParallel.Interleaving.CutoffPollFixture

open Threading
open AtomicWord (Event Write)
open CutoffVisibility (FlagWrite FlagRead History)
open TaskAncestry (Generation)

private def parent := 100
private def child := 200
private def slots : Nat → Nat → Option Nat
  | 0,0 => some parent | 1,0 => some child | _,_ => none

private def tick (w n : Nat) : Event := ⟨w,n⟩
private def word (w n : Nat) (op : Joint.TaskWord) : Joint.Event Unit := .taskWord w n op
private def split (w n : Nat) : Joint.Event Unit := .splitWord w n (.mark 0)
private def initial := tick 3 0
private def oldEpoch := tick 2 33
private def newEpoch := tick 2 133

/-- Worker 0 owns the parent; worker 1 creates a child on its behalf. After
    detaching from its parent assignment, worker 0 helps the child. Worker 2
    cuts the parent while both allocations still hold their installed tasks. -/
private def use_ (base : Nat) (stale wrongCall wrongParent reversed : Bool) : Joint.Tape Unit :=
  let w := fun who off op => word who (base+off) op
  let t := fun who off => tick who (base+off)
  let epoch0 := if base = 0 then initial else oldEpoch
  let oldValue : BitVec 64 := if base = 0 then 0 else 1
  let addr := if wrongParent then child else parent
  let ancestorSource := if stale then tick 2 32 else t 2 32
  [w 0 1 (.copyParent parent none),split 0 (base+2),w 0 3 (.install parent none),
   w 0 4 (.storeFlag parent false),w 0 5 (.ownerActive parent),w 0 7 (.resetCache (t 0 6) epoch0 oldValue),
   w 0 8 (.book 1 parent),w 0 9 (.book 2 parent),
   w 1 10 (.copyParent child (some parent)),split 1 (base+11),w 1 12 (.install child (some parent)),
   w 1 13 (.storeFlag child false),w 1 14 (.ownerActive child),w 1 16 (.resetCache (t 1 15) epoch0 oldValue),
   w 0 17 (.detach parent),w 0 19 (.resetCache (t 0 18) epoch0 oldValue),w 1 20 (.book 0 child),
   w 0 21 (.latch false),w 0 23 (.epoch (t 0 22) epoch0 oldValue (oldValue-1)),
   w 0 24 (.beginWalk (some child)),w 0 25 (.loadFlag child false (t 1 13)),
   w 0 26 (.loadFlag parent false (t 0 4)),w 0 27 (.endWalk false),w 0 28 (.checkStop (base+21)),
   w 0 29 (.latch false),w 0 31 (.epoch (t 0 30) epoch0 oldValue oldValue),
   w 2 32 (.storeFlag parent true),
   w 0 34 (.latch false),w 0 36 (.epoch (t 0 35) (t 2 33) (oldValue+1) oldValue),
   w 0 37 (.beginWalk (some child)),w 0 38 (.loadFlag (if reversed then parent else child) false (t 1 13)),
   w 0 39 (.loadFlag parent true ancestorSource),w 0 40 (.endWalk true),w 0 41 (.checkStop (base+34)),
   w 0 42 (.latch true),w 0 43 (.checkStop (if wrongCall then base+34 else base+42)),
   w 0 44 (.detach child),w 0 46 (.resetCache (t 0 45) (t 2 33) (oldValue+1)),
   w 1 47 (.detach child),w 1 49 (.resetCache (t 1 48) (t 2 33) (oldValue+1)),
   w 1 50 (.restore child (some addr)),w 1 51 (.clearTask child),w 1 52 (.afterSplit (some parent)),
   w 1 53 (.loadFlag parent true (t 2 32)),w 1 54 (.endWalk true),
   w 1 55 (.detach parent),w 2 56 (.detach parent),w 0 57 (.restore parent none),w 0 58 (.clearTask parent),
   w 0 59 (.afterSplit none),w 0 60 (.endWalk false)]

private def records (stale wrongCall wrongParent reversed : Bool) :=
  use_ 0 false false false false ++ use_ 100 stale wrongCall wrongParent reversed
private def epochs : List Write :=
  [⟨initial,0,false,false⟩,⟨oldEpoch,1,true,true⟩,⟨newEpoch,2,true,true⟩]
private def fields (slot : Nat) : List FlagWrite :=
  if slot = parent then [⟨tick 0 4,false⟩,⟨tick 2 32,true⟩,⟨tick 0 104,false⟩,⟨tick 2 132,true⟩]
  else if slot = child then [⟨tick 1 13,false⟩,⟨tick 1 113,false⟩] else []

local instance : NeZero epochs.length := ⟨by decide⟩
local instance : NeZero (fields child).length := ⟨by decide⟩

private def rparent (who pc : Nat) (src : Fin 4) : FlagRead fields := ⟨parent,tick who pc,src⟩
private def rchild (pc : Nat) : FlagRead fields := ⟨child,tick 0 pc,1⟩
private def history (stale : Bool) : History where
  epochs := epochs
  epochReads := [⟨tick 0 118,1,false⟩,⟨tick 0 122,1,true⟩,⟨tick 0 130,1,true⟩,⟨tick 0 135,2,true⟩]
  flags := fields
  flagReads := [rchild 125,rparent 0 126 2,rchild 138,rparent 0 139 (if stale then 1 else 3),rparent 1 153 3]
  otherSync := fun _ _ => False
private def memory (stale : Bool) : NativeStopReturns.Memory := ⟨history stale,[],[],fun _ => [],fun _ => []⟩

private def checked (stale wrongCall wrongParent reversed : Bool) (mark : Nat) : Option (List Nat × List Event × List Event) := do
  let es := records stale wrongCall wrongParent reversed
  let j ← TaskAncestry.capture slots es
  let w ← TaskAncestry.window j ⟨child,tick 1 mark⟩ (tick 0 143) (tick 1 152)
  let cache ← CutoffPollReplay.cacheAt (memory stale) j es (tick 0 143) w.view
  let scan ← CutoffPollReplay.ownerScan (history stale) j (tick 1 152)
  guard (cache.cache.latched && scan.stopped)
  return (w.ancestry.cachedChain,cache.reads.map (·.event),scan.reads.map (·.event))

private abbrev expected := ([child,parent],[tick 0 125,tick 0 126,tick 0 138,tick 0 139],[tick 1 153])
example : checked false false false false 111 = some expected := by decide +kernel

-- Both the raw register and the atomic history now claim an earlier positive
-- store at the same physical address. The numeric flags and final scores agree.
example : (checked true false false false 111).isNone ∧ (checked false true false false 111).isNone ∧
    (checked false false true false 111).isNone ∧ (checked false false false true 111).isNone ∧
    (checked false false false false 11).isNone := by decide +kernel

namespace LocalCut

private def trace : Joint.Tape Unit :=
  [word 0 1 (.copyParent parent none), split 0 2, word 0 3 (.install parent none),
   word 0 4 (.storeFlag parent false), word 0 5 (.ownerActive parent), word 0 6 (.book 1 parent),
   word 0 7 (.book 2 parent), word 1 10 (.storeFlag parent true),
   word 2 12 (.latch false), word 2 14 (.epoch (tick 2 13) (tick 1 11) 1 0),
   word 2 15 (.beginWalk (some parent)), word 2 16 (.loadFlag parent true (tick 1 10)),
   word 2 17 (.endWalk true),word 2 18 (.checkStop 12),
   word 1 20 (.detach parent),word 2 21 (.detach parent),word 0 22 (.detach parent),
   word 0 24 (.restore parent none),word 0 25 (.clearTask parent),word 0 26 (.afterSplit none),
   word 0 27 (.endWalk false)]

private def flag (s : Nat) : List FlagWrite := if s = parent then [⟨tick 0 4,false⟩,⟨tick 1 10,true⟩] else []
private def counter : List Write := [⟨initial,0,false,false⟩,⟨tick 1 11,1,true,true⟩]
private def hist : History := ⟨counter,flag,[⟨tick 2 13,⟨1,by decide⟩,true⟩],
  [⟨parent,tick 2 16,⟨1,by decide⟩⟩],fun _ _ => False⟩
private def abort : List Write := [⟨initial,1,false,false⟩]
private def abortRead : AtomicWord.Read abort := ⟨tick 0 28,⟨0,by decide⟩,true⟩
private def mem : NativeStopReturns.Memory := ⟨hist,abort,[abortRead],fun _ => [],fun _ => []⟩
private def gate (cut : Bool) : Option Bool := do
  let j ← TaskAncestry.capture slots trace
  let owner ← CutoffPollReplay.ownerScan hist j (tick 0 26)
  let check : SplitReturn.Observation.OwnerCheck hist :=
    ⟨owner,some ⟨abort,abortRead⟩,by simp,false⟩
  let ev ← CutoffPollReplay.flagged mem j trace ⟨parent,tick 0 2⟩ check (tick 0 23)
    [⟨⟨0,0,256⟩,cut⟩] (tick 2 18) (fun _ => none)
  match ev with
  | .flag _ _ _ _ _ _ (.here ..) => return check.stopped
  | _ => none

-- The helper's own cutoff discards its ticket and the owner continues.
-- With neither a posted cutoff nor an observed ancestor cause, the same
-- positive word cannot justify the discard.
example : gate true = some false ∧ gate false = none := by decide +kernel

end LocalCut

namespace ForeignEmptyPoll

private def t (w n : Nat) (op : Joint.TaskWord) : Joint.Event Unit := .taskWord w n op
private def epochs : List Write := [⟨⟨3,0⟩,0,false,false⟩,⟨⟨3,4⟩,1,true,true⟩,⟨⟨2,13⟩,2,true,true⟩]
private def flags (slot : Nat) : List FlagWrite :=
  if slot == 2 then [⟨⟨2,4⟩,false⟩,⟨⟨2,12⟩,true⟩] else []
private def hist (early : Bool) : History := {
  epochs := epochs
  flags := flags
  epochReads := [⟨⟨1,22⟩,⟨1,by decide⟩,true⟩,⟨⟨1,32⟩,⟨2,by decide⟩,true⟩]
  flagReads := (if early then [⟨2,⟨1,25⟩,⟨0,by decide⟩⟩] else []) ++ [⟨2,⟨1,35⟩,⟨1,by decide⟩⟩]
  otherSync := fun _ _ => False}
private def mem (early : Bool) : NativeStopReturns.Memory := ⟨hist early,[],[],fun _ => [],fun _ => []⟩
private def book : Joint.Tape Unit :=
  [t 2 1 (.copyParent 2 none),.splitWord 2 2 (.mark 0),t 2 3 (.install 2 none),
   t 2 4 (.storeFlag 2 false),t 2 5 (.ownerActive 2),t 2 8 (.book 1 2)]
private def tape (early : Bool) : Joint.Tape Unit :=
  (if early then book else []) ++
  [t 1 21 (.latch false),t 1 23 (.epoch ⟨1,22⟩ ⟨3,4⟩ 1 0),t 1 24 (.beginWalk (if early then some 2 else none))] ++
  (if early then [t 1 25 (.loadFlag 2 false ⟨2,4⟩)] else []) ++
  [t 1 26 (.endWalk false),t 1 27 (.checkStop 21)] ++ (if early then [] else book) ++
  [t 2 12 (.storeFlag 2 true),t 1 31 (.latch false),t 1 33 (.epoch ⟨1,32⟩ ⟨2,13⟩ 2 1),
   t 1 34 (.beginWalk (some 2)),t 1 35 (.loadFlag 2 true ⟨2,12⟩),t 1 36 (.endWalk true),t 1 37 (.checkStop 31)]

private def check (early : Bool) : Option (Bool × Nat) := do
  let es := tape early
  let j ← TaskAncestry.capture (fun _ _ => some 2) es
  let v ← j.polls ⟨1,37⟩
  let r ← CutoffPollReplay.cacheAt (mem early) j es ⟨1,37⟩ v
  return (r.cache.latched,r.cache.seen.toNat)

-- A Lazy SMP helper can scan an empty chain before it is booked. Both booking
-- orders retain the observed epoch and attribute the later cutoff to this slot.
example : check true = some (true,2) ∧ check false = some (true,2) := by decide +kernel

-- Changing both the writer and the supplied reads-from event would preserve
-- the flag value. Only a worker assigned to this generation may issue the swap.
private def assigned (worker : Nat) :=
  (TaskAncestry.capture (fun _ _ => some 2)
    (book ++ [t worker 12 (.storeFlag 2 true),t 1 34 (.beginWalk (some 2)),
      t 1 35 (.loadFlag 2 true ⟨worker,12⟩),t 1 36 (.endWalk true)])).isSome

example : assigned 1 = true ∧ assigned 2 = true ∧ assigned 3 = false := by decide +kernel

end ForeignEmptyPoll

namespace Publication

private def releases : List Write := [⟨⟨1,18⟩,1,true,true⟩]
private def acquire : AtomicWord.Read releases := ⟨⟨2,9⟩,⟨0,by decide⟩,true⟩
private def h : History := ⟨releases,fun _ => [],[acquire],[],
  fun a b => a = ⟨0,10⟩ ∧ b = ⟨1,7⟩⟩
private def lock : FlagSegment.Transfer h := .handoff ⟨0,10⟩ ⟨1,7⟩ ⟨rfl,rfl⟩
private def epoch : FlagSegment.Transfer h :=
  .epoch ⟨0,by decide⟩ acquire (List.mem_singleton.mpr rfl) rfl
    ⟨rfl,Nat.le_refl _,fun _ after before => False.elim (Nat.not_lt_of_ge before after)⟩
private def visible (ts : List (FlagSegment.Transfer h)) :=
  (FlagSegment.published ts ⟨0,4⟩ ⟨2,12⟩).isSome

-- The worker numbers and per-worker counters do not give a global event order.
example : visible [lock,epoch] ∧ !visible [epoch,lock] ∧ !visible [epoch] ∧ !visible [] := by decide +kernel

end Publication

namespace ClockOrder

private def handoffs : List (Event × Event) := [(tick 1 8,tick 2 3),(tick 0 100,tick 1 4)]
private def history : History := ⟨[],fun _ => [],[],[],fun a b => (a,b) ∈ handoffs⟩
private def sync : MaskClock.Sync history := ⟨handoffs,fun _ _ => Iff.rfl⟩
private def timeline := [tick 0 100,tick 1 4,tick 1 8,tick 2 3]

private def reach := do
  let c ← MaskClock.capture sync timeline
  let check (ts : List (FlagSegment.Transfer history)) (x y : Event) := (FlagSegment.published ts x y).isSome
  return (check sync.transfers (tick 0 97) (tick 2 9),
    check c.transfers (tick 0 97) (tick 2 9),check c.transfers (tick 2 9) (tick 0 101))

-- The last check is a different-worker pair with no causal path in that direction.
example : reach = some (false,true,false) ∧ (MaskClock.capture sync timeline.reverse).isNone := by
  cbv

private def cycle : List (Event × Event) := [(tick 0 10,tick 1 10),(tick 1 10,tick 0 10)]
private def cyclic : History := ⟨[],fun _ => [],[],[],fun a b => (a,b) ∈ cycle⟩
private def bad : MaskClock.Sync cyclic := ⟨cycle,fun _ _ => Iff.rfl⟩
example : (MaskClock.capture bad [tick 0 10,tick 1 10]).isNone ∧ (MaskClock.capture bad []).isNone := by
  decide +kernel

end ClockOrder

namespace EqualFlags

private def writes (_ : Nat) : List FlagWrite := [⟨tick 0 4,true⟩,⟨tick 0 8,true⟩]
private def history (ordersReaders : Bool) : History :=
  ⟨[],writes,[],[⟨0,tick 1 9,⟨1,by decide⟩⟩,⟨0,tick 2 12,⟨0,by decide⟩⟩],
    fun a b => (a,b) ∈ (if ordersReaders then [(tick 1 10,tick 2 11)] else [])⟩

private def accepted (ordersReaders : Bool) := do
  let s : MaskClock.Sync (history ordersReaders) :=
    ⟨if ordersReaders then [(tick 1 10,tick 2 11)] else [],fun _ _ => Iff.rfl⟩
  return CutoffPollReplay.flagMemoryOK (← MaskClock.capture s [])

-- Both readers get true. The older source at the second read is legal until a
-- handoff between the readers orders the two loads of the same atomic.
example : accepted false = some true ∧ accepted true = some false := by cbv

end EqualFlags

namespace NestedMask

private abbrev m (w t : Nat) (op : Joint.MaskWord) : Joint.Event Unit := .maskWord w t op

-- The dispatch and ancestor registers come from the existing two-parent fixture.
-- Both owners' real guard operands are interleaved with that same call stream.
private def tape (source : Event) : Joint.Tape Unit :=
  [m 0 0 (.clear 0),m 1 9 (.clear 0)] ++ (TaskRetention.Regression.tape.flatMap fun row =>
    match row with
    | .taskWord 1 56 (.restore ..) =>
      [m 1 52 (.shutdown 0 false),m 1 53 (.load 0 source 0 true),m 1 54 (.returned 0),row]
    | _ => [row]) ++ [m 0 78 (.shutdown 0 false),m 0 80 (.load 0 ⟨1,75⟩ 0 true),m 0 81 (.returned 0)]

private def edges (source : Event) (keep : Bool) : List (Event × Event) :=
  ((SlotSynchronization.recorded (tape source)).getD []).filter (fun pair =>
    keep || pair != (tick 2 50,tick 1 53))
private def h (source : Event) (keep : Bool) : History :=
  ⟨[],fun _ => [],[],[],fun a b => (a,b) ∈ edges source keep⟩
private def sync (source : Event) (keep : Bool) : MaskClock.Sync (h source keep) := ⟨edges source keep,fun _ _ => Iff.rfl⟩
private def addr : TaskRetention.Address := fun w bank => some (64*w+bank)

private def replay (source : Event) (middleRelease : Bool) :
    Option (PLift (SplitCompletion.RetainedThrough (h source middleRelease) (tick 2 34) (tick 0 80))) := do
  let chronology ← MaskClock.capture (sync source middleRelease) []
  let es := tape source
  let i := MaskGeneration.input chronology es addr
  let child : Generation := ⟨128,tick 2 22⟩
  let j ← TaskAncestry.capture addr es
  let w ← TaskAncestry.window j child (tick 2 26) (tick 2 32)
  let state ← TaskRetention.capture addr es
  let path ← TaskRetention.route i es state ⟨0,tick 0 2⟩ (tick 0 80) child (tick 2 34) w.parents
  return ⟨path.held chronology.before .here⟩

set_option maxRecDepth 16384 in
theorem throughTwoMasks : SplitCompletion.RetainedThrough (h (tick 2 50) true) (tick 2 34) (tick 0 80) := by
  have accepted : (replay (tick 2 50) true).isSome := by cbv
  exact ((replay (tick 2 50) true).get accepted).down

-- Reading the middle slot's initial zero or dropping the remote release-to-acquire
-- edge prevents the outer generation from being credited with the inner access.
set_option maxRecDepth 16384 in
example : (replay (tick 1 9) true).isNone ∧ (replay (tick 2 50) false).isNone := by cbv

end NestedMask

namespace LockOrder

private def sites : Joint.Tape Unit :=
  [.splitWord 2 1 (.lock 0 0),.splitWord 3 1 (.lock 0 1),
   .splitWord 2 5 (.unlock 0 0),.splitWord 3 8 (.unlock 0 1),
   .splitWord 3 10 (.lock 0 0),.splitWord 2 14 (.lock 0 1)]

-- The two mutexes overlap in time. Each successor observes its own bank's
-- release, including when the last critical section on each bank is still open.
example : (SlotSynchronization.recorded sites).any (fun es =>
    es.length == 2 && es.contains (tick 2 5,tick 3 10) && es.contains (tick 3 8,tick 2 14)) := by
  decide +kernel

end LockOrder

namespace EpochReuse

private def t (w n : Nat) (op : Joint.TaskWord) : Joint.Event Unit := .taskWord w n op

-- Write 1 and write 3 both store the exact same 64-bit counter value `1#64`.
-- A cache hit must match the physical write index `seenSource`, not only the numeric value.
private def epochs : List Write :=
  [⟨tick 3 0,0,false,false⟩,⟨tick 2 10,1,true,true⟩,⟨tick 2 20,2,true,true⟩,⟨tick 2 30,1,true,true⟩]
private def flags (slot : Nat) : List FlagWrite :=
  if slot == 2 then [⟨tick 0 4,false⟩] else []

local instance : NeZero epochs.length := ⟨by decide⟩

private def hist (hitIdx : Fin 4) : History where
  epochs := epochs
  flags := flags
  epochReads := [⟨tick 1 12,⟨1,by decide⟩,true⟩,⟨tick 1 42,hitIdx,true⟩]
  flagReads := [⟨2,tick 1 15,⟨0,by decide⟩⟩]
  otherSync := fun _ _ => False

private def mem (hitIdx : Fin 4) : NativeStopReturns.Memory := ⟨hist hitIdx,[],[],fun _ => [],fun _ => []⟩

private def tape (hitSrc : Event) : Joint.Tape Unit :=
  [t 0 1 (.copyParent 2 none),.splitWord 0 2 (.mark 0),t 0 3 (.install 2 none),
   t 0 4 (.storeFlag 2 false),t 0 5 (.ownerActive 2),t 0 6 (.book 1 2),
   t 1 11 (.latch false),t 1 13 (.epoch (tick 1 12) (tick 2 10) 1 0),
   t 1 14 (.beginWalk (some 2)),t 1 15 (.loadFlag 2 false (tick 0 4)),t 1 16 (.endWalk false),
   t 1 17 (.checkStop 11),
   t 1 41 (.latch false),t 1 43 (.epoch (tick 1 42) hitSrc 1 1),t 1 44 (.checkStop 41)]

private def checkHit (hitIdx : Fin 4) (hitSrc : Event) : Option Nat := do
  let es := tape hitSrc
  let j ← TaskAncestry.capture (fun _ _ => some 2) es
  let v ← j.polls (tick 1 44)
  let r ← CutoffPollReplay.cacheAt (mem hitIdx) j es (tick 1 44) v
  return r.hits.length

-- Hit reading the exact same write (index 1, tick 2 10) is accepted.
-- Hit reading a distinct write with the SAME 64-bit value 1 (index 3, tick 2 30) is rejected.
example : checkHit ⟨1,by decide⟩ (tick 2 10) = some 1 ∧
    checkHit ⟨3,by decide⟩ (tick 2 30) = none := by
  decide +kernel

end EpochReuse

end NeuralReversi.PvsParallel.Interleaving.CutoffPollFixture

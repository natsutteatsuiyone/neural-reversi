module
public import NeuralReversi.PvsParallel.SlotFrameMir.ReceiverStack
public import NeuralReversi.PvsParallel.SlotFrameMir.Assembly

@[expose] public section

namespace NeuralReversi.PvsParallel.SlotFrameMir.Receiver

open Threading
open AtomicWord (Event Write Read InReleaseSequence SequencedBefore)
open CutoffVisibility (History HappensBefore)
open SplitRebookingMir (Coherence)

inductive FlagWriter (worker : Nat) : Write → Type
  | birth (e : Event) : FlagWriter worker ⟨e,0,false,false⟩
  | external (e : Event) (site : Site) (bit : Bool) : site=.book ∨ site=.bookJob →
      emits site (.searching bit)=true → FlagWriter worker ⟨e,if bit then 1 else 0,true,false⟩
  | localStore (e : Event) (site : Site) (bit : Bool) : e.worker=worker →
      emits site (.searching bit)=true → FlagWriter worker ⟨e,if bit then 1 else 0,true,false⟩

private theorem writer_release {entry : Write} (origin : FlagWriter w entry) (on : entry.value=1) : entry.release=true := by
  cases origin <;> simp_all

private theorem writer_idle {entry : Write} (origin : FlagWriter w entry) (off : entry.value≠1) :
    entry.release=false ∨ (entry.release=true ∧ entry.event.worker=w ∧ entry.value=0) := by
  cases origin with
  | birth => exact Or.inl rfl
  | external e site bit kind emitted =>
    rcases kind with rfl | rfl <;> cases bit <;> simp_all [emits,searchingWriters]
  | localStore e site bit owner emitted =>
    cases bit
    · exact Or.inr ⟨rfl,owner,rfl⟩
    · exact (off rfl).elim

/-- The raw target searching word, the actual busy load dispatching this work,
    and the worker's sequencing interval. Birth is ordered by thread startup.
    Every local flag instruction in this interval is identified in the dynamic
    trace. The successful idle-to-dispatch prologue emits no flag instruction. -/
structure Prefix (hist : History) (ws : List Write) (worker : Nat) where
  ops : List EventOp
  executed : Calls [] ops
  first : Event
  chosen : firstSplit ops=some first
  onThread : first.worker=worker
  guard : Read ws
  running : (ws.get guard.source).value=1
  acquired : guard.acquire=true
  scheduledOn : guard.event.worker=worker
  capture : Event
  launched : SequencedBefore guard.event capture
  source : ∀ i : Fin ws.length, FlagWriter worker (ws.get i)
  distinct : ∀ i : Fin ws.length, (ws.get i).event≠guard.event
  initialized : ∀ i : Fin ws.length, (ws.get i).release=false → HappensBefore hist (ws.get i).event guard.event
  coherence : Coherence hist ws guard
  sync : ∀ i : Fin ws.length, guard.acquire=true → InReleaseSequence ws i guard.source →
    hist.otherSync (ws.get i).event guard.event
  localTrace : ∀ (i : Fin ws.length) (bit : Bool), (ws.get i).event.worker=worker →
    (ws.get i).value=(if bit then 1 else 0) →
    guard.event.step < (ws.get i).event.step → (ws.get i).event.step ≤ first.step →
    ((ws.get i).event,.effect (.searching bit)) ∈ untilSplit ops

structure Posting (hist : History) (ws : List Write) (stamp : Event) where
  read : Read ws
  acquire : read.acquire=true
  coherent : Coherence hist ws read
  sync : ∀ i : Fin ws.length, read.acquire=true → InReleaseSequence ws i read.source →
    hist.otherSync (ws.get i).event read.event
  hasSlot : Bool
  deepest : Bool
  helpful : Bool
  failed : Bool
  branch : WorkerFrameMir.canJoin (decide ((ws.get read.source).value=1)) failed hasSlot deepest helpful=true
  store : SequencedBefore read.event stamp

/-- For every competing booking that locks this worker after its capture, the
    first chronological nested split happens-before the new non-atomic Arc
    write. The posted target can be any physical slot. -/
theorem Posting.after_first {ws : List Write} (p : Posting hist ws stamp) (start : Prefix hist ws w)
    (afterCapture : SpinLock.OrderedAccess hist start.capture p.read.event) :
    HappensBefore hist start.first stamp := by
  have initialSeq : InReleaseSequence ws start.guard.source start.guard.source :=
    ⟨writer_release (start.source _) start.running,Nat.le_refl _,fun _ lo hi => by omega⟩
  have h := (HappensBefore.other (start.sync _ start.acquired initialSeq)).trans
    ((HappensBefore.sequenced start.launched).trans afterCapture.orders)
  have under := p.coherent.writeRead _ h
  have permit:=(Bool.and_eq_true_iff.mp (Bool.and_eq_true_iff.mp p.branch).1).1
  have off : (ws.get p.read.source).value≠1 := by simpa using permit
  have later : start.guard.source.val < p.read.source.val := by
    by_cases yes : start.guard.source=p.read.source
    · exact (off (yes ▸ start.running)).elim
    · have := Fin.val_ne_of_ne yes;omega
  rcases writer_idle (start.source p.read.source) off with birth | ⟨release,owner,zero⟩
  · have low:=start.coherence.writeRead _ (start.initialized _ birth);omega
  · have afterGuard : start.guard.event.step < (ws.get p.read.source).event.step := by
      by_cases prior : (ws.get p.read.source).event.step < start.guard.event.step
      · have prev := start.coherence.writeRead p.read.source (.sequenced ⟨owner.trans start.scheduledOn.symm,prior⟩)
        omega
      · have unequal : (ws.get p.read.source).event.step ≠ start.guard.event.step := by
          intro eq
          apply start.distinct p.read.source
          cases a : (ws.get p.read.source).event
          cases b : start.guard.event
          rw [a,b] at eq
          have ew := owner.trans start.scheduledOn.symm
          rw [a,b] at ew
          cases ew;cases eq;rfl
        omega
    have afterFirst : start.first.step < (ws.get p.read.source).event.step := by
      by_cases tooSoon : (ws.get p.read.source).event.step ≤ start.first.step
      · exact ((start.executed.untouched.2 _) (start.localTrace _ false owner zero afterGuard tooSoon)).elim
      · omega
    have seq : InReleaseSequence ws p.read.source p.read.source :=
      ⟨release,Nat.le_refl _,fun _ lo hi => by omega⟩
    exact (HappensBefore.sequenced ⟨start.onThread.trans owner.symm,afterFirst⟩).trans
      ((HappensBefore.other (p.sync _ p.acquire seq)).trans (.sequenced p.store))

/-- Two-sided ordering for the worker-locked external modification. The
    owner's unsynchronized guard read cannot race a competing active write. -/
theorem Posting.side {ws : List Write} (p : Posting hist ws stamp) (start : Prefix hist ws w)
    (order : Nonempty (SpinLock.OrderedAccess hist stamp start.capture) ∨
      Nonempty (SpinLock.OrderedAccess hist start.capture p.read.event))
    (readGuard : SequencedBefore actualGuard start.first) :
    HappensBefore hist stamp start.capture ∨ HappensBefore hist actualGuard stamp := by
  rcases order with before | later
  · exact before.elim fun lock => Or.inl lock.orders
  · exact later.elim fun lock => Or.inr ((HappensBefore.sequenced readGuard).trans (p.after_first start lock))

end NeuralReversi.PvsParallel.SlotFrameMir.Receiver

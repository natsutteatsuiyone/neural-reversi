import NeuralReversi.PvsParallel.SlotFrameMir.Generation
import NeuralReversi.PvsParallel.SplitSchedulerMir.OwnerFields

namespace NeuralReversi.PvsParallel.SlotFrameMir

open Threading
open AtomicWord (Event Read Write ReadWindow)
open CutoffVisibility (History HappensBefore)
open SplitCompletion (Action)

private def closeFlag (e : Event) : Write := ⟨e,0,false,false⟩

/-- All false stores to the protected flag in this generation have the same
    value. The following generation can reopen the physical flag. -/
structure FlagRead (hist : History) where
  earlier : List Write
  closes : List Event
  later : List Write
  load : Read (earlier ++ closes.map closeFlag ++ later)
  memory : SplitRebookingMir.Coherence hist _ load
  next : ∀ i : Fin (earlier ++ closes.map closeFlag ++ later).length,
    i.val=earlier.length+closes.length → HappensBefore hist load.event ((earlier ++ closes.map closeFlag ++ later).get i).event

def FlagRead.value (f : FlagRead hist) :=
  ((f.earlier ++ f.closes.map closeFlag ++ f.later).get f.load.source).value

theorem FlagRead.after_close (f : FlagRead hist) (i : Fin f.closes.length)
    (edge : HappensBefore hist (f.closes.get i) f.load.event) : f.value=0 := by
  let head := f.earlier ++ f.closes.map closeFlag
  let ws := head ++ f.later
  let pos : Fin ws.length := ⟨f.earlier.length+i.val,by have:=i.isLt;simp [ws,head];omega⟩
  have atClose : (ws.get pos).event=f.closes.get i := by simp [ws,head,pos,closeFlag]
  have lower := f.memory.writeRead pos (atClose ▸ edge)
  have upper : f.load.source.val < head.length := by
    by_cases inside : f.load.source.val < head.length
    · exact inside
    · have width:=f.load.source.isLt
      let j : Fin ws.length := ⟨head.length,by change f.load.source.val < ws.length at width;omega⟩
      have order:=f.memory.readWrite j (f.next j (by simp [j,head]))
      exact (inside order).elim
  have after : f.earlier.length ≤ f.load.source.val := by change f.earlier.length+i.val ≤ _ at lower;omega
  have get : ws[f.load.source.val]?=some (ws.get f.load.source) := List.getElem?_eq_getElem _
  change (head ++ f.later)[_]? = _ at get
  rw [List.getElem?_append_left upper] at get
  dsimp only [head] at get
  rw [List.getElem?_append_right after,List.getElem?_map] at get
  obtain ⟨stamp,_,same⟩:=Option.map_eq_some_iff.mp get
  change (ws.get f.load.source).value=0
  rw [←same];rfl

/-- The initial roster is in one critical section before any finishes. Each
    later successful booking read the flag in its own protected section, after
    all earlier detach sections that closed the flag. -/
structure GateHistory (hist : History) (g : RawMask ws) where
  initialCount : Nat
  initial : ∀ i : Fin g.actions.length, i.val < initialCount → (g.actions.get i).release=false
  late : ∀ j : Fin g.actions.length, initialCount ≤ j.val → ∀ w e, g.actions.get j=.book w e →
    ∃ (sample : FlagRead hist) (idle : Bool),
      SplitActivationMir.gate true (decide (sample.value=1)) idle = true ∧
      ∀ i : Fin g.actions.length, i.val < j.val → (g.actions.get i).release=true →
        ∃ idx : Fin sample.closes.length, HappensBefore hist (sample.closes.get idx) sample.load.event

theorem GateHistory.closed (trace : GateHistory hist g) : ClosedAfterFirst g := by
  intro i j less first
  cases eq : g.actions.get j with
  | finish => rfl
  | book w e =>
    have past : trace.initialCount ≤ j.val := by
      by_cases old : i.val<trace.initialCount
      · have := trace.initial i old;rw [first] at this;cases this
      · omega
    obtain ⟨flag,idle,allowed,ordering⟩ := trace.late j past w e eq
    obtain ⟨idx,prior⟩ := ordering i less first
    have gone := flag.after_close idx prior
    simp [SplitActivationMir.gate,gone] at allowed

/-- Actual per-load source and local event index for the owner idle loop. -/
structure OwnerExit (hist : History) (g : RawMask writes) where
  timeline : SplitSchedulerMir.Clock
  log : List (SplitSchedulerMir.Guard writes)
  returned : SplitSchedulerMir.Returned SplitSchedulerMir.idle writes 0 log
  poolOpen : SplitSchedulerMir.Guard.shutdown true ∉ log
  sampled : ∀ pre src post, log=pre ++ .mask src::post →
    ReadWindow (⟨(SplitSchedulerMir.OwnerFields.maskClock timeline).atStep (4+pre.length),src,true⟩ : Read writes) (HappensBefore hist)
      g.mark (g.mark.val+g.actions.length)

abbrev OwnerExit.clock {g : RawMask ws} (exit : OwnerExit hist g) : SplitSchedulerMir.Clock :=
  SplitSchedulerMir.OwnerFields.maskClock exit.timeline

structure Previous (hist : History) (physical : BitVec 64) where
  writes : List Write
  generation : RawMask writes
  gates : GateHistory hist generation
  ownerExit : OwnerExit hist generation
  observed : physical = (writes.get ⟨generation.mark.val+generation.actions.length,generation.bounded⟩).value

theorem Previous.empty (prev : Previous hist physical) : physical=0 := by
  obtain ⟨pre,src,post,log,zero⟩:=SplitSchedulerMir.idle_acquired_zero prev.ownerExit.returned prev.ownerExit.poolOpen
  have window:=prev.ownerExit.sampled pre src post log
  rw [prev.observed]
  exact prev.generation.stays_zero prev.gates.closed src _ window.bounds.1 window.bounds.2 (Nat.le_refl _) zero

end NeuralReversi.PvsParallel.SlotFrameMir

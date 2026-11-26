import NeuralReversi.PvsParallel.SlotFrameMir.Assembly

namespace NeuralReversi.PvsParallel.SlotFrameMir

open Threading
open AtomicWord (Event Read Write SequencedBefore InReleaseSequence signed32At)
open CutoffVisibility (History HappensBefore)
open SplitCompletion (Action)

/-- The normal worker epilogue, identified in this slot's physical mask history. -/
structure WorkerReset (hist : History) (clock : Event → Nat) (g : RawMask ws) (slot : Slot) where
  worker : Worker
  release : Event
  index : Fin g.actions.length
  history : List Tick
  scheduled : Scheduled hist clock history
  epilogue : Source hist history release (.release slot worker)
  maskWord : (machine history).mask slot=(ws.get (g.before index)).value
  observed : g.actions.get index=.finish worker release

structure WorkerStore (hist : History) (clock : Event → Nat) (g : RawMask ws) (slot : Slot) (store : Event)
    extends WorkerReset hist clock g slot where
  published : SequencedBefore store release

theorem WorkerStore.before_zero {g : RawMask ws} (p : WorkerStore hist clock g slot store)
    (causal : ∀ x y, HappensBefore hist x y → clock x < clock y)
    (gates : GateHistory hist g) (r : Read ws) (acquire : r.acquire=true)
    (window : AtomicWord.ReadWindow r (HappensBefore hist) g.mark (g.mark.val+g.actions.length))
    (zero : (ws.get r.source).value=0)
    (sync : ∀ i, r.acquire=true → InReleaseSequence ws i r.source → hist.otherSync (ws.get i).event r.event) :
    HappensBefore hist store r.event := by
  have on:=p.scheduled.normal_release_is_set causal p.epilogue
  rw [p.maskWord] at on
  exact (HappensBefore.sequenced p.published).trans
    (g.credited_release gates.closed hist r acquire window zero p.index p.worker p.release p.observed on sync)

/-- Score projection for the owner return, without requiring that a reset of
    another worker removed a set bit or that the zero source is the last RMW. -/
theorem returned_scan (hist : History) (clock : Event → Nat)
    (causal : ∀ x y, HappensBefore hist x y → clock x < clock y)
    (slot : Slot) (g : RawMask mask) (gates : GateHistory hist g) (exit : OwnerExit hist g)
    (scores : List Write) (source last : Fin scores.length)
    (sync : ∀ pre src post, exit.log=pre ++ .mask src::post → ∀ i,
      InReleaseSequence mask i src → hist.otherSync (mask.get i).event (exit.clock.atStep (4+pre.length)))
    (origin : (scores.get last).event=exit.clock.atStep 2 ∨
      Nonempty (WorkerStore hist clock g slot (scores.get last).event))
    (memory : SplitRebookingMir.Coherence hist scores (SplitSchedulerMir.scoreLoad scores source exit.clock exit.log))
    (next : ∀ i : Fin scores.length, i.val=last.val+1 →
      SequencedBefore (exit.clock.atStep (exit.log.length+5)) (scores.get i).event)
    (pv : Bool) (beta : Int) (initial : SplitWindow.State) (reports : List Int)
    (stores : SplitWindow.BestStoreWindow scores last initial.best reports) :
    (SplitSchedulerMir.ownerSteps exit.log)[exit.log.length+5]?=some .best ∧
      signed32At scores source=(SplitWindow.scan pv beta initial reports).best := by
  obtain ⟨pre,src,post,log,zero⟩:=SplitSchedulerMir.idle_acquired_zero exit.returned exit.poolOpen
  have site:=SplitSchedulerMir.physical_owner_indices pre post src
  refine ⟨log ▸ site.2.2.2,?_⟩
  let read : Read mask := ⟨exit.clock.atStep (4+pre.length),src,true⟩
  let score := SplitSchedulerMir.scoreLoad scores source exit.clock exit.log
  have afterZero : SequencedBefore read.event score.event := by
    refine ⟨rfl,exit.clock.advances _ _ ?_⟩
    simp [log];omega
  have before : HappensBefore hist (scores.get last).event score.event := by
    rcases origin with seeded | existsWorker
    · rw [seeded];exact .sequenced ⟨rfl,exit.clock.advances _ _ (by omega)⟩
    · obtain ⟨worker⟩:=existsWorker
      have edge:=worker.before_zero causal gates read rfl (exit.sampled pre src post log) zero
        (fun i _ seq => sync pre src post log i seq)
      exact edge.trans (.sequenced afterZero)
  have low : last.val ≤ source.val := memory.writeRead last before
  have exactSource : source=last := by
    apply Fin.ext
    by_cases after : last.val < source.val
    · let i : Fin scores.length := ⟨last.val+1,by have:=source.isLt;omega⟩
      have high : source.val < i.val := memory.readWrite i (.sequenced (next i rfl))
      change source.val < last.val+1 at high;omega
    · omega
  rw [exactSource]
  exact SplitWindow.bestStoreContract_proven scores last pv beta initial reports stores

end NeuralReversi.PvsParallel.SlotFrameMir

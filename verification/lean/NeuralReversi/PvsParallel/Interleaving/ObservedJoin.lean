module
public import NeuralReversi.PvsParallel.Interleaving.StopSchedule

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.ObservedJoin

open Threading
open AtomicWord (Write Read Event SequencedBefore signed32At)
open Aspiration (Window)
open PvsMoveLoop (ChildRelation)
open Tickets (Post State)
open SplitReturn.Observation (OwnerCheck)
open SplitCompletion (Worker)
open NativeStopReturns (Memory)

/-- Only a legal entry window is needed to translate raw alpha loads into indexed
    loads. Execution reads its score from memory before that translation. -/
structure Exit (t : Schedule.Task) (done_ : ChildRelation Nat) (len : Nat) (start : BitVec w) where
  answer : Option Int
  posts : List Post
  observes : ∀ original, Window t.entry.alpha t.beta → t.entry.alpha ≤ max original t.entry.best →
    ∃ j : Join t done_ len start, j.answer = answer ∧ j.state.reports = posts

structure Effective (t : Schedule.Task) (done_ : ChildRelation Nat) (len : Nat) (start : BitVec w) where
  memory : Memory
  previous : List Write
  owner : Worker
  generation : Mask.Generation
  following : List Mask.Generation
  observed : Mask.Observed memory.history previous owner generation following
  stop : OwnerCheck memory.history
  slot : Nat
  alphas : List Write
  state : State w
  run : StopSchedule.Scored memory stop observed.waitAt slot alphas t done_ len start owner generation.actions state
  scores : List Write
  load : Read scores
  last : Fin scores.length
  stores : SplitWindow.BestStoreWindow scores last t.entry.best ((Schedule.reports state).map SplitResult.Report.score)
  lastWriter : SequencedBefore (scores.get last).event observed.waitAt ∨
    ∃ i : Fin generation.actions.length, (generation.actions.get i).release = true ∧
      SequencedBefore (scores.get last).event (generation.actions.get i).event
  loadSite : load.event = (generation.clock owner).atStep (observed.log.length+5)
  beforeReuse : ∀ i : Fin scores.length, i.val = last.val+1 → SequencedBefore load.event (scores.get i).event
  coherent : SplitRebookingMir.Coherence memory.history scores load

def Effective.answer (trace : Effective t done_ len start) : Option Int :=
  SplitReturn.checked trace.stop.stopped (signed32At trace.scores trace.load.source)

theorem Effective.read (trace : Effective t done_ len start) :
    signed32At trace.scores trace.load.source = (t.locked trace.state.reports).best := by
  obtain ⟨wait,site,_⟩ := trace.observed.wait (StopSchedule.Run.project trace.observed trace.run)
  apply SplitReturn.ownerBestContract_proven trace.memory.history trace.scores trace.load trace.last wait
    t.pv t.beta t.entry (Schedule.reports trace.state) _ trace.stores
  refine ⟨?_,?_,trace.beforeReuse,trace.coherent.writeRead,trace.coherent.readWrite⟩
  · rcases trace.lastWriter with owner | helper
    · exact .inl (by rw [site]; exact owner)
    · exact .inr helper
  · rw [site,trace.loadSite]
    exact ⟨rfl,trace.generation.advances _ _ (by omega)⟩

def Effective.exit (trace : Effective t done_ len start) : Exit t done_ len start :=
  ⟨trace.answer,trace.state.reports,fun original w cap => by
    obtain ⟨j,state,_,_,_,score⟩ := StopSchedule.join trace.observed trace.stop trace.run w cap
    exact ⟨j,by rw [score,Effective.answer,trace.read],congrArg Tickets.State.reports state⟩⟩

/-- Raw modification order can include redundant bit resets. The returning mask
    read and the final score read retain their original physical source indices. -/
structure Physical (t : Schedule.Task) (done_ : ChildRelation Nat) (len : Nat) (start : BitVec w) where
  memory : Memory
  mask : List Write
  generation : SlotFrameMir.RawMask mask
  owner : SlotFrameMir.OwnerExit memory.history generation
  gates : SlotFrameMir.GateHistory memory.history generation
  mark : (mask.get generation.mark).event = owner.clock.atStep 1
  sync : ∀ pre src rest, owner.log = pre ++ .mask src :: rest → ∀ i, AtomicWord.InReleaseSequence mask i src →
    memory.history.otherSync (mask.get i).event (owner.clock.atStep (4+pre.length))
  stop : OwnerCheck memory.history
  slot : Nat
  alphas : List Write
  state : State w
  run : StopSchedule.Scored memory stop (owner.clock.atStep (3+owner.log.length)) slot alphas t done_ len start
    generation.owner (SlotFrameMir.RawProjection.actions generation) state
  scores : List Write
  last : Fin scores.length
  load : Read scores
  stores : SplitWindow.BestStoreWindow scores last t.entry.best ((Schedule.reports state).map SplitResult.Report.score)
  lastWriter : (scores.get last).event = owner.clock.atStep 2 ∨
    ∃ i : Fin (SlotFrameMir.RawProjection.actions generation).length,
      ((SlotFrameMir.RawProjection.actions generation).get i).release = true ∧
      SequencedBefore (scores.get last).event ((SlotFrameMir.RawProjection.actions generation).get i).event
  loadSite : load.event = owner.clock.atStep (owner.log.length+5)
  beforeReuse : ∀ i : Fin scores.length, i.val = last.val+1 → SequencedBefore load.event (scores.get i).event
  coherent : SplitRebookingMir.Coherence memory.history scores load

def Physical.answer (p : Physical t done_ len start) : Option Int :=
  SplitReturn.checked p.stop.stopped (signed32At p.scores p.load.source)

theorem Physical.read_source (p : Physical t done_ len start) : p.load.source = p.last := by
  let a := SlotFrameMir.RawProjection.activation p.generation p.gates.closed
  have mem := SlotFrameMir.RawProjection.memory p.owner p.gates.closed p.mark p.sync p.scores p.load.source
    ⟨fun i hb => p.coherent.writeRead i (by simpa only [p.loadSite,SplitSchedulerMir.scoreLoad] using hb),
     fun i hb => p.coherent.readWrite i (by simpa only [p.loadSite,SplitSchedulerMir.scoreLoad] using hb)⟩
  have selected := SplitSchedulerMir.returned_last_score p.memory.history _ p.scores a p.owner.clock _ p.load.source p.last
    (SlotFrameMir.RawProjection.exit_returned p.owner) (SlotFrameMir.RawProjection.open_pool _ p.owner.poolOpen) mem p.lastWriter
    (fun i later => by simpa only [SplitSchedulerMir.scoreLoad,List.length_map,←p.loadSite] using p.beforeReuse i later)
  exact selected.1

theorem Physical.read (p : Physical t done_ len start) :
    signed32At p.scores p.load.source = (t.locked p.state.reports).best := by
  rw [p.read_source]
  exact SplitWindow.bestStoreContract_proven _ p.last t.pv t.beta t.entry _ p.stores

def Physical.exit (p : Physical t done_ len start) : Exit t done_ len start :=
  ⟨p.answer,p.state.reports,fun original w cap => by
    obtain ⟨j,_,_,_,_,_,_,state,_,_,answer⟩ := StopSchedule.join_raw p.generation p.owner p.gates p.mark p.sync
      p.load p.loadSite p.coherent p.stop p.run w cap
    exact ⟨j,by rw [answer,Physical.answer,p.read],congrArg Tickets.State.reports state⟩⟩

inductive Trace (t : Schedule.Task) (done_ : ChildRelation Nat) (len : Nat) (start : BitVec w) where
  | effective (observed : Effective t done_ len start)
  | physical (observed : Physical t done_ len start)

def Trace.answer : Trace t done_ len start → Option Int
  | .effective p => p.answer
  | .physical p => p.answer

def Trace.state : Trace t done_ len (start : BitVec w) → State w
  | .effective p => p.state
  | .physical p => p.state

def Trace.exit (p : Trace t done_ len start) : Exit t done_ len start :=
  ⟨p.answer,p.state.reports,fun orig w cap => by
    cases p with
    | effective q => exact q.exit.observes orig w cap
    | physical q => exact q.exit.observes orig w cap⟩

end NeuralReversi.PvsParallel.Interleaving.ObservedJoin

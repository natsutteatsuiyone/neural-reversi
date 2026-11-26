import NeuralReversi.PvsParallel.Interleaving.ObservedJoin
import NeuralReversi.PvsParallel.SlotFrameMir.RawNative

namespace NeuralReversi.PvsParallel.Interleaving.ScoreReuse

open Threading
open AtomicWord (Event Write Read SequencedBefore)
open SplitSchedulerMir (Clock)
open SplitSchedulerMir.OwnerFields (Initialization phases laterWrites maskClock)
open CutoffVisibility (History HappensBefore)
open SlotFrameMir (RawMask OwnerExit GateHistory WorkerStore)
open ObservedJoin (Physical)

structure Scores where
  previous : List Write
  seed : BitVec 32
  improvements : List Write
  later : List Initialization

/-- Once the first reporting helper raises the best score, a later live
    helper returning the same word executes no best-score store. -/
theorem tied_reports (first second : SplitWorkerMir.Input) (locked : SplitWorkerMir.Locked)
    (nextAlpha score : SplitWorkerMir.Word) (active : locked.stopped = false)
    (better : locked.best.toInt < score.toInt) :
    let stored := (SplitWorkerMir.commit first locked score).best
    stored = some score ∧
      (SplitWorkerMir.commit second ⟨false,nextAlpha,stored.getD locked.best⟩ score).best = none := by
  simp [SplitWorkerMir.commit_eq,active,better]

/-- initialize_split_point writes the seed before any worker can acquire the new task. -/
def Scores.initial (r : Scores) (owner : Clock) : Write :=
  ⟨owner.atStep (SplitSchedulerMir.initPhases.idxOf .seed), r.seed.zeroExtend 64, false, false⟩

def Scores.current (r : Scores) (owner : Clock) := r.initial owner :: r.improvements

/-- Modification order for best_score on this physical slot. Initializers of other
    slots and initializers that have not reached their score store contribute no write. -/
def Scores.words (r : Scores) (owner : Clock) (slot guards : Nat) :=
  r.previous ++ r.current owner ++ laterWrites owner slot .score (phases guards).length r.later

def Scores.last (r : Scores) (owner : Clock) (slot guards : Nat) : Fin (r.words owner slot guards).length :=
  ⟨r.previous.length + r.improvements.length, by simp [words,current]; omega⟩

theorem Scores.beforeReuse (r : Scores) (owner : Clock) (slot guards : Nat)
    (load : Read (r.words owner slot guards)) (site : load.event = (maskClock owner).atStep (guards+5))
    (i : Fin (r.words owner slot guards).length) (next : i.val = (r.last owner slot guards).val+1) :
    SequencedBefore load.event ((r.words owner slot guards).get i).event := by
  rw [site]
  change SequencedBefore (owner.atStep (if guards+5 < 3 then guards+5 else guards+5+2)) _
  rw [ite_eq_right (by omega)]
  apply SplitSchedulerMir.OwnerFields.before_reuse owner slot .score guards r.later r.previous (r.current owner) _
  · simp [SplitSchedulerMir.OwnerFields.phases_eq]
  · simpa [last,current,Nat.add_assoc] using next

private theorem Scores.member (r : Scores) (owner : Clock) (slot guards : Nat) :
    (r.words owner slot guards).get (r.last owner slot guards) ∈ r.current owner := by
  have localIndex : r.improvements.length < (r.current owner).length := by simp [current]
  have selected : (r.words owner slot guards).get (r.last owner slot guards) =
      (r.current owner)[r.improvements.length] := by
    have found : (r.words owner slot guards)[r.previous.length+r.improvements.length]? =
        some ((r.current owner)[r.improvements.length]) := by
      simp only [words,List.append_assoc,List.getElem?_append_right (Nat.le_add_right ..),
        Nat.add_sub_cancel_left,List.getElem?_append_left localIndex,List.getElem?_eq_getElem localIndex]
    exact (List.getElem?_eq_some_iff.mp found).2
  rw [selected]
  exact List.getElem_mem ..

private theorem Scores.window {seed : Int} (r : Scores) (owner : Clock) (slot guards : Nat)
    (events : SplitWindow.BestStores seed replies (r.current owner)) :
    SplitWindow.BestStoreWindow (r.words owner slot guards) (r.last owner slot guards) seed replies :=
  ⟨r.previous,r.current owner,_,rfl,by simp [last,current,Nat.add_assoc],events⟩

private theorem Scores.writer (r : Scores) {g : RawMask mask} (exit : OwnerExit hist g) (gates : GateHistory hist g)
    (slot : Nat) (clock : Event → Nat) (causal : ∀ x y, HappensBefore hist x y → clock x < clock y)
    (publish : ∀ w ∈ r.improvements, Nonempty (WorkerStore hist clock g (g.owner,slot) w.event)) :
    let writes := r.words exit.timeline slot exit.log.length
    let last := r.last exit.timeline slot exit.log.length
    (writes.get last).event = exit.clock.atStep 2 ∨
      ∃ i : Fin (SlotFrameMir.RawProjection.actions g).length,
        ((SlotFrameMir.RawProjection.actions g).get i).release = true ∧
          SequencedBefore (writes.get last).event ((SlotFrameMir.RawProjection.actions g).get i).event := by
  rcases List.mem_cons.mp (r.member exit.timeline slot exit.log.length) with seed | improved
  · left
    rw [seed]
    rfl
  · obtain ⟨worker⟩ := publish _ improved
    obtain ⟨i,word⟩ := SlotFrameMir.RawProjection.normal_credit gates.closed causal worker.toWorkerReset
    change (SlotFrameMir.RawProjection.actions g).get i = .finish worker.worker worker.release at word
    exact .inr ⟨i,by rw [word]; rfl,by rw [word]; exact worker.published⟩

/-- The score records on a returning raw split. Each changed score has its worker
    epilogue on the captured slot. The owner issues the later initializers in program order. -/
structure Frame (t : Schedule.Task) (done_ : PvsMoveLoop.ChildRelation Nat) (len : Nat) (start : BitVec w) where
  memory : NativeStopReturns.Memory
  mask : List Write
  generation : RawMask mask
  owner : OwnerExit memory.history generation
  gates : GateHistory memory.history generation
  mark : (mask.get generation.mark).event = owner.clock.atStep 1
  sync : ∀ pre src rest, owner.log = pre ++ .mask src :: rest → ∀ i, AtomicWord.InReleaseSequence mask i src →
    memory.history.otherSync (mask.get i).event (owner.clock.atStep (4+pre.length))
  stop : SplitReturn.Observation.OwnerCheck memory.history
  bankIndex : Nat
  flagAddress : SlotFrameMir.Slot → Nat
  alphas : List Write
  state : Tickets.State w
  run : StopSchedule.Scored memory stop (owner.clock.atStep (3+owner.log.length)) (flagAddress (generation.owner,bankIndex)) alphas t done_ len start
    generation.owner (SlotFrameMir.RawProjection.actions generation) state
  scores : Scores
  load : Read (scores.words owner.timeline bankIndex owner.log.length)
  site : load.event = owner.clock.atStep (owner.log.length+5)
  stores : SplitWindow.BestStores t.entry.best ((Schedule.reports state).map SplitResult.Report.score) (scores.current owner.timeline)
  time : Event → Nat
  causal : ∀ x y, HappensBefore memory.history x y → time x < time y
  publication : ∀ w ∈ scores.improvements, Nonempty (WorkerStore memory.history time generation (generation.owner,bankIndex) w.event)
  coherent : SplitRebookingMir.Coherence memory.history _ load

def Frame.physical (f : Frame t done_ len start) : Physical t done_ len start :=
  ⟨f.memory,f.mask,f.generation,f.owner,f.gates,f.mark,f.sync,f.stop,f.flagAddress (f.generation.owner,f.bankIndex),f.alphas,f.state,f.run,_,f.scores.last ..,f.load,
    f.scores.window _ _ _ f.stores,f.scores.writer f.owner f.gates f.bankIndex f.time f.causal f.publication,f.site,
    f.scores.beforeReuse _ _ _ f.load f.site,f.coherent⟩

/-- The relaxed result read selects the seed or the last improving store of this
    generation, even if another generation wrote an identical i32. -/
def Contract : Prop := ∀ (t : Schedule.Task) (done_ : PvsMoveLoop.ChildRelation Nat)
    (len w : Nat) (start : BitVec w) (f : Frame t done_ len start),
    f.load.source.val = f.scores.previous.length + f.scores.improvements.length ∧
    AtomicWord.signed32At (f.scores.words f.owner.timeline f.bankIndex f.owner.log.length) f.load.source =
      (t.locked f.state.reports).best

theorem contract_proven : Contract := by
  intro t done_ len w start f
  exact ⟨congrArg Fin.val f.physical.read_source,f.physical.read⟩

/-- Reusing the same bank slot appends to its old modification order. The two
    return-source indices differ even when every generation seeds the same score
    and no worker improves it. Intervening completed generations are allowed. -/
theorem Scores.later_index (old new : Scores) (oldClock newClock : Clock) (slot oldGuards newGuards : Nat)
    (intervening : List Write) (history : new.previous = old.previous ++ old.current oldClock ++ intervening) :
    (old.last oldClock slot oldGuards).val < (new.last newClock slot newGuards).val := by
  simp only [last,history,List.length_append,current,List.length_cons]
  omega

end NeuralReversi.PvsParallel.Interleaving.ScoreReuse

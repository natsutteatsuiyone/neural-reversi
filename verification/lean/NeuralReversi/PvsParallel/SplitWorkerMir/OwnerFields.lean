module
public import NeuralReversi.PvsParallel.SplitWorkerMir.FieldTrace
public import NeuralReversi.PvsParallel.SplitSchedulerMir.Worker

@[expose] public section

namespace NeuralReversi.PvsParallel.SplitWorkerMir.FieldStores

open Threading
open AtomicWord (Write Event Read SequencedBefore signed32At)
open CutoffVisibility (History HappensBefore)
open SplitCompletion (Activation Retirement)
open SplitReturn.Observation (ScoreRead)
open SplitSchedulerMir (Clock Guard Returned OwnerMemory scoreLoad)
open TranspositionTable (Packing.encodeBestMove Packing.decodeBestMove)

private theorem at_last_mem (pre current post : List α) (nonempty : current ≠ [])
    (last : Fin (pre ++ current ++ post).length) (atEnd : last.val+1 = pre.length+current.length) :
    (pre ++ current ++ post).get last ∈ current := by
  have pos := List.length_pos_iff.mpr nonempty
  have lower : pre.length ≤ last.val := by omega
  have upper : last.val < (pre ++ current).length := by simp; omega
  have inside : last.val - pre.length < current.length := by omega
  have full := List.getElem?_eq_getElem last.isLt (l := pre ++ current ++ post)
  rw [List.getElem?_append_left upper,List.getElem?_append_right lower] at full
  exact List.mem_of_getElem? full

private theorem seq_trans {a b c : Event} (ab : SequencedBefore a b) (bc : SequencedBefore b c) :
    SequencedBefore a c := ⟨ab.1.trans bc.1,Nat.lt_trans ab.2 bc.2⟩

/-- The stored fields and the mask release are different atomic objects. The worker's
    unlock and subsequent reset give the program-order edge between them. -/
structure WorkerRelease {mask : List Write} (act : Activation mask) (call : Raw) where
  unlock : Event
  finish : Fin act.actions.length
  released : (act.actions.get finish).release = true
  afterUnlock : SequencedBefore unlock (act.actions.get finish).event
  beforeUnlock : ∀ op ∈ call.ops, ∀ w, op = .writeAlpha w ∨ op = .writeBest w ∨ op = .writeMove w →
    SequencedBefore w.event unlock

private theorem last_source {mask : List Write} (act : Activation mask) (seed : Write) (calls : List Raw)
    (project : Physical → Option Write)
    (field : ∀ op w, project op = some w → op = .writeAlpha w ∨ op = .writeBest w ∨ op = .writeMove w)
    (release : ∀ call ∈ calls, ∀ op ∈ call.ops, ∀ w, project op = some w → Nonempty (WorkerRelease act call))
    (epoch : current = [seed] ++ calls.flatMap (fun call => call.ops.filterMap project))
    (pre post : List Write) (last : Fin (pre ++ current ++ post).length)
    (atEnd : last.val+1 = pre.length+current.length) :
    ((pre ++ current ++ post).get last).event = seed.event ∨
    ∃ i : Fin act.actions.length, (act.actions.get i).release = true ∧
      SequencedBefore ((pre ++ current ++ post).get last).event (act.actions.get i).event := by
  subst current
  have mem := at_last_mem pre _ post (by simp) last atEnd
  rcases List.mem_cons.mp mem with init | stored
  · exact Or.inl (congrArg Write.event init)
  · obtain ⟨call,ran,ops⟩ := List.mem_flatMap.mp stored
    obtain ⟨op,found,wrote⟩ := List.mem_filterMap.mp ops
    obtain ⟨r⟩ := release call ran op found _ wrote
    exact Or.inr ⟨r.finish,r.released,seq_trans (r.beforeUnlock op found _ (field op _ wrote)) r.afterUnlock⟩

def BestOrMove (record : Physical) (w : Write) := record = .writeBest w ∨ record = .writeMove w

structure CoherentRead (history : History) (ws : List Write) (load : Read ws) (last : Fin ws.length) : Prop where
  next : ∀ i : Fin ws.length, i.val = last.val+1 → SequencedBefore load.event (ws.get i).event
  writeRead : ∀ i : Fin ws.length, HappensBefore history (ws.get i).event load.event → i.val ≤ load.source.val
  readWrite : ∀ i : Fin ws.length, HappensBefore history load.event (ws.get i).event → load.source.val < i.val

private theorem after_writers {mask : List Write} {seed : Write} (act : Activation mask)
    (wait : Retirement history mask act) (calls : List Raw) (project : Physical → Option Write)
    (identifies : ∀ op w, project op = some w → BestOrMove op w)
    (release : ∀ call ∈ calls, ∀ op ∈ call.ops, ∀ w, BestOrMove op w → Nonempty (WorkerRelease act call))
    (epoch : current = [seed] ++ calls.flatMap (fun call => call.ops.filterMap project))
    (pre post : List Write) (last : Fin (pre ++ current ++ post).length) (atEnd : last.val+1 = pre.length+current.length)
    (seedBefore : SequencedBefore seed.event wait.read.event) (load : Read (pre ++ current ++ post))
    (afterWait : SequencedBefore wait.read.event load.event) (observed : CoherentRead history _ load last) :
    ScoreRead history wait load last := by
  have origin := last_source act seed calls project (fun op w h => Or.inr (identifies op w h))
    (fun call member op found w h => release call member op found w (identifies op w h))
    epoch pre post last atEnd
  refine ⟨?_,afterWait,observed.next,observed.writeRead,observed.readWrite⟩
  rcases origin with eq | worker
  · exact .inl (eq ▸ seedBefore)
  · exact .inr worker

theorem owner_reads {mask : List Write} (act : Activation mask) (history : History) (clock : Clock)
    (guards : List (Guard mask)) (returned : Returned SplitSchedulerMir.idle mask 0 guards)
    (live : Guard.shutdown true ∉ guards) (a b m : Write) (calls : List Raw)
    (ran : replay pv beta encodeBestMove atIndex (initialLog initial a b m) calls = some final)
    (bPre bPost mPre mPost : List Write)
    (lastB source : Fin (bPre ++ final.best ++ bPost).length)
    (lastM : Fin (mPre ++ final.moves ++ mPost).length) (move : Read (mPre ++ final.moves ++ mPost))
    (endB : lastB.val+1 = bPre.length+final.best.length) (endM : lastM.val+1 = mPre.length+final.moves.length)
    (memory : OwnerMemory history mask _ act clock guards source)
    (seedScore : b.event = clock.atStep 2) (seedMove : SequencedBefore m.event (clock.atStep 3))
    (afterScore : SequencedBefore (scoreLoad _ source clock guards).event move.event)
    (release : ∀ call ∈ calls, ∀ op ∈ call.ops, ∀ w, BestOrMove op w → Nonempty (WorkerRelease act call))
    (later : ∀ i : Fin (bPre ++ final.best ++ bPost).length, i.val = lastB.val+1 →
      SequencedBefore (scoreLoad _ source clock guards).event ((bPre ++ final.best ++ bPost).get i).event)
    (moveMemory : CoherentRead history _ move lastM) :
    ∃ wait : Retirement history mask act,
      ScoreRead history wait (scoreLoad _ source clock guards) lastB ∧ ScoreRead history wait move lastM ∧
      wait.read.event = clock.atStep (3 + guards.length) := by
  obtain ⟨wait,load,assigned,_,stamp⟩ := SplitSchedulerMir.retired_after_assignment history mask _ act clock guards
    source returned live memory
  obtain ⟨_,scores,moves⟩ := modification_orders calls ran
  have seeded : SequencedBefore b.event wait.read.event := by
    rw [seedScore]
    exact seq_trans (b := clock.atStep 3) ⟨rfl,clock.advances 2 3 (by decide)⟩ assigned
  refine ⟨wait,?_,?_,stamp⟩
  · apply after_writers act wait calls Physical.best? _ release scores bPre bPost lastB endB seeded _ load
      ⟨later,memory.scoreWR,memory.scoreRW⟩
    intro op w got
    left
    cases op <;> simp_all [Physical.best?]
  · apply after_writers act wait calls Physical.move? _ release moves mPre mPost lastM endM
      (seq_trans seedMove assigned) move (seq_trans load afterScore) moveMemory
    intro op w got
    right
    cases op <;> simp_all [Physical.move?]

theorem score_from_fields {mask : List Write} (act : Activation mask) (history : History) (clock : Clock)
    (guards : List (Guard mask)) (returned : Returned SplitSchedulerMir.idle mask 0 guards)
    (live : Guard.shutdown true ∉ guards) (a b m : Write) (calls : List Raw)
    (seedAlpha : low32 a = initial.alpha) (seedBest : low32 b = initial.best)
    (seedMove : low8 m = encodeBestMove initial.picked)
    (ran : replay pv beta encodeBestMove atIndex (initialLog initial a b m) calls = some final)
    (pre post : List Write) (last source : Fin (pre ++ final.best ++ post).length)
    (atEnd : last.val+1 = pre.length+final.best.length)
    (memory : OwnerMemory history mask (pre ++ final.best ++ post) act clock guards source)
    (seedTime : b.event = clock.atStep 2)
    (release : ∀ call ∈ calls, ∀ op ∈ call.ops, ∀ w, BestOrMove op w → Nonempty (WorkerRelease act call))
    (noReuse : ∀ i : Fin (pre ++ final.best ++ post).length, i.val = last.val+1 →
      SequencedBefore (scoreLoad _ source clock guards).event ((pre ++ final.best ++ post).get i).event) :
    source = last ∧ signed32At (pre ++ final.best ++ post) source =
      (SplitWindow.scan pv beta initial.state (final.reports.map SplitResult.Report.score)).best := by
  have raw := (modification_orders calls ran).2.1
  have bestField (op : Physical) (w : Write) (ok : op.best? = some w) : op = .writeBest w := by
    cases op <;> simp_all [Physical.best?]
  have origin := last_source act b calls Physical.best?
    (fun op w h => Or.inr (Or.inl (bestField op w h)))
    (fun call member op found w h => release call member op found w (Or.inl (bestField op w h)))
    raw pre post last atEnd
  rw [seedTime] at origin
  have read := SplitSchedulerMir.returned_last_score history mask _ act clock guards source last
    returned live memory origin noReuse
  have checked := replay_preserves calls (seed initial a b m seedAlpha seedBest seedMove) ran
  have window : SplitWindow.BestStoreWindow _ last initial.best.toInt (final.reports.map SplitResult.Report.score) :=
    ⟨pre,final.best,post,rfl,atEnd,checked.bestWrites⟩
  exact ⟨read.1,read.2.trans (SplitWindow.bestStoreContract_proven _ last pv beta initial.state _ window)⟩

end NeuralReversi.PvsParallel.SplitWorkerMir.FieldStores

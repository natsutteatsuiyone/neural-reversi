import NeuralReversi.PvsParallel.SplitSchedulerMir.Correct

namespace NeuralReversi.PvsParallel.SplitSchedulerMir

open Threading
open AtomicWord (Event Write SequencedBefore)
open SplitWorkerMir (Input Searched Locked Step ProgramOrder)

theorem physical_owner_indices {mask : List Write} (pre post : List (Guard mask)) (src : Fin mask.length) :
    let log := pre ++ .mask src :: post
    (ownerSteps log)[1]? = some .mark ∧ (ownerSteps log)[2]? = some .seed ∧
    (ownerSteps log)[4 + pre.length]? = some (.guard (.mask src)) ∧
    (ownerSteps log)[log.length + 5]? = some .best := by
  intro log
  rw [ownerSteps_eq]
  have idx : 4 + pre.length = pre.length + 4 := Nat.add_comm _ _
  simp [log, idx, List.getElem?_append]
  rw [ite_eq_right (by omega), show pre.length + (post.length + 1) + 1 - pre.length = post.length + 2 by omega]
  simp

inductive WorkerStep
  | body (step : Step)
  | release

def workerSteps (steps : List Step) : List WorkerStep := workerPhases.flatMap fun
  | .dispatch => steps.map .body
  | .release => [.release]
  | _ => []

theorem workerSteps_eq (steps : List Step) : workerSteps steps = steps.map .body ++ [.release] := by
  simp [workerSteps, workerPhases]

def releaseIndex (steps : List Step) : Nat := (workerSteps steps).length - 1

theorem releaseIndex_eq (steps : List Step) : releaseIndex steps = steps.length := by
  simp [releaseIndex, workerSteps_eq]

theorem dispatch_eq : dispatchModes = [.NonPV, .PV, .Root] := rfl

private theorem worker_position {a : α} {leading suffix : List α} :
    (leading ++ a :: suffix)[leading.length]? = some a := by simp

/-- The clock indexes all projected worker events in the activation, including every child return,
    poll and unlock in subsequent tickets. The activation identifies the mask RMW by this event. -/
def workerPublication (act : SplitCompletion.Activation mask)
    (i : Input) (body : Searched) (locked : Locked)
    (hit : SplitWorkerMir.search i = some body) (credited : SplitWorkerMir.recordLocked i body locked = some report)
    (earlier later : List Step) (clock : Clock) (worker : SplitCompletion.Worker)
    (workerId : clock.worker = worker.val) (time : Step → Event)
    (observed : ∀ n step, (body.steps ++ SplitWorkerMir.lockedSteps)[n]? = some step →
      time step = clock.atStep (earlier.length + n))
    (finish : Fin act.actions.length)
    (maskRmw : act.actions.get finish = .finish worker
      (clock.atStep (releaseIndex (earlier ++ body.steps ++ SplitWorkerMir.lockedSteps ++ later)))) :
    SplitReturn.Observation.ReportPublication act := by
  let segment := body.steps ++ SplitWorkerMir.lockedSteps
  have eventOrder (n j : Nat) (lt : n < j) :
      SequencedBefore (clock.atStep (earlier.length + n)) (clock.atStep (earlier.length + j)) :=
    ⟨rfl, clock.advances _ _ (by omega)⟩
  have order : ProgramOrder time segment := by
    constructor
    intro a b pre post eq
    have aPos : segment[pre.length]? = some a := eq ▸ worker_position
    have bPos : segment[(pre ++ [a]).length]? = some b := by
      rw [eq]; simpa only [List.append_assoc, List.singleton_append] using
        (worker_position (leading := pre ++ [a]) (a := b) (suffix := post))
    rw [observed _ _ aPos, observed _ _ bPos]
    exact eventOrder _ _ (by simp)
  let unlockIndex := body.steps.length + 1
  have unlockPos : segment[unlockIndex]? = some .unlock := by
    change (body.steps ++ [.poll, .unlock])[body.steps.length + 1]? = _
    simp
  have eq := observed _ _ unlockPos
  have finishEq : (act.actions.get finish).event =
      clock.atStep (earlier ++ body.steps ++ SplitWorkerMir.lockedSteps ++ later).length := by rw [maskRmw]; simp [SplitCompletion.Action.event, releaseIndex_eq]
  apply SplitWorkerMir.publicationFromWorker act i body locked hit credited time order worker finish
  · rw [finishEq]; simpa only [releaseIndex_eq] using maskRmw
  · rw [eq]; exact workerId
  · rw [eq, finishEq]
    exact ⟨rfl, clock.advances _ _ (by simp [unlockIndex, SplitWorkerMir.lockedSteps])⟩

theorem physical_release (steps : List Step) :
    (workerSteps steps)[steps.length]? = some .release := by rw [workerSteps_eq]; simp

theorem returned_scan (history : CutoffVisibility.History) (mask scores : List Write)
    (act : SplitCompletion.Activation mask) (clock : Clock) (log : List (Guard mask))
    (source last : Fin scores.length) (run : Returned idle mask 0 log) (openPool : Guard.shutdown true ∉ log)
    (memory : OwnerMemory history mask scores act clock log source)
    (origin : (scores.get last).event = clock.atStep 2 ∨ ∃ i : Fin act.actions.length,
      (act.actions.get i).release = true ∧ SequencedBefore (scores.get last).event (act.actions.get i).event)
    (noReuse : ∀ i : Fin scores.length, i.val = last.val + 1 →
      SequencedBefore (scoreLoad scores source clock log).event (scores.get i).event)
    (pv : Bool) (beta : Int) (initial : SplitWindow.State) (reports : List Int)
    (stores : SplitWindow.BestStoreWindow scores last initial.best reports) :
    (ownerSteps log)[log.length + 5]? = some .best ∧
      AtomicWord.signed32At scores source = (SplitWindow.scan pv beta initial reports).best := by
  obtain ⟨pre, src, post, eq, _⟩ := idle_acquired_zero run openPool
  refine ⟨eq ▸ (physical_owner_indices pre post src).2.2.2, ?_⟩
  exact (returned_last_score history mask scores act clock log source last run openPool memory origin noReuse).2.trans
    (SplitWindow.bestStoreContract_proven scores last pv beta initial reports stores)

end NeuralReversi.PvsParallel.SplitSchedulerMir

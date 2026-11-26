import NeuralReversi.MoveList.ConcurrentExecution
import NeuralReversi.PvsParallel.SplitWorkerMir.Loop

namespace NeuralReversi.PvsParallel.SplitWorkerMir.Iterator

open MoveList.ConcurrentTrace (Write State forWorker)
open Threading.SplitReturn (Work Departure Loop)
open Threading.SplitResult (Report)
open Threading.SplitCompletion (Worker)

structure LockedCall where
  ticket : Ticket
  order : Nat

abbrev Feed := Nat → Nat → Option LockedCall

def acquired (feed : Feed) (op : Write w) : Option LockedCall := do
  let call ← feed op.worker op.opening
  return {call with ticket.input.index := op.old.toNat}

structure WorkerResult where
  completed : Work
  published : List (Nat × Report)

/-- A normally retired worker consumes its actual successful iterator operations. A cutoff
    or cancellation releases the worker immediately, including the debited current ticket. -/
def retire : List LockedCall → Option WorkerResult
  | [] => some ⟨⟨[],[],emptyExit⟩,[]⟩
  | call :: rest => do
    let t := call.ticket
    let body ← search t.input
    let report := selected t.input body
    match (commit t.input t.locked body.chosen).exit with
    | .stopped =>
      if rest.isEmpty then some ⟨⟨[t.input.index],[],.cancelled⟩,[]⟩ else none
    | .cutoff =>
      if rest.isEmpty then some ⟨⟨[t.input.index],[report],.cutoff⟩,[(call.order,report)]⟩ else none
    | .again =>
      let tail ← retire rest
      return ⟨⟨t.input.index :: tail.completed.tickets,report :: tail.completed.reports,tail.completed.departure⟩,
        (call.order,report) :: tail.published⟩

private theorem retire_spec (calls : List LockedCall) (ok : retire calls = some result) :
    work (calls.map LockedCall.ticket) = some result.completed ∧
    result.completed.tickets = calls.map (fun c => c.ticket.input.index) ∧
    result.published.map Prod.snd = result.completed.reports := by
  induction calls generalizing result with
  | nil => cases ok; exact ⟨rfl,rfl,rfl⟩
  | cons call rest ih =>
    simp only [retire] at ok
    obtain ⟨body,searched,next⟩ := Option.bind_eq_some_iff.mp ok
    simp only [List.map_cons,work,searched]
    dsimp only [Bind.bind,Option.bind]
    cases ended : (commit call.ticket.input call.ticket.locked body.chosen).exit with
    | stopped | cutoff =>
      rw [ended] at next
      cases rest <;> simp_all <;> subst result <;> exact ⟨rfl,rfl,rfl⟩
    | again =>
      rw [ended] at next
      obtain ⟨tail,more,same⟩ := Option.bind_eq_some_iff.mp next
      cases same
      obtain ⟨ran,tickets,reports⟩ := ih more
      simp [ran,tickets,reports]

def worker (feed : Feed) (ops : List (Write w)) (i : Nat) : Option WorkerResult := do
  let calls ← (forWorker ops i).mapM (acquired feed)
  retire calls

private theorem indices (ops : List (Write w)) (ok : ops.mapM (acquired feed) = some calls) :
    calls.map (fun c => c.ticket.input.index) = ops.map (fun op => op.old.toNat) := by
  induction ops generalizing calls with
  | nil => cases ok; rfl
  | cons op rest ih =>
    rw [List.mapM_cons] at ok
    obtain ⟨first,item,next⟩ := Option.bind_eq_some_iff.mp ok
    obtain ⟨tail,more,same⟩ := Option.bind_eq_some_iff.mp next
    cases same
    unfold acquired at item
    obtain ⟨_,_,same⟩ := Option.bind_eq_some_iff.mp item
    cases same
    simp [ih more]

theorem worker_observed {ops : List (Write w)} (ok : worker feed ops i = some result) :
    result.completed.tickets = (forWorker ops i).map (fun op => op.old.toNat) ∧
      result.published.map Prod.snd = result.completed.reports := by
  obtain ⟨_,bound,returned⟩ := Option.bind_eq_some_iff.mp ok
  obtain ⟨_,ts,rs⟩ := retire_spec _ returned
  exact ⟨ts.trans (indices _ bound),rs⟩

abbrev Drained := @MoveList.ConcurrentTrace.Drained

theorem worker_finished (run : MoveList.ConcurrentTrace.Execution len start s) (ok : worker feed s.writes i = some result)
    (ending : result.completed.departure = .exhausted → Nonempty (Drained len start s i)) :
    Loop len s.current result.completed.tickets result.completed.reports result.completed.departure := by
  obtain ⟨calls,_,returned⟩ := Option.bind_eq_some_iff.mp ok
  apply finished (calls.map LockedCall.ticket) (retire_spec _ returned).1
  intro reason
  obtain ⟨observed⟩ := ending reason
  exact ⟨s.current,[],observed.run.prior_end run observed.prior observed.read,.done⟩

def lockReports (results : Worker → WorkerResult) (workers : List Worker) : List Report :=
  let published := workers.flatMap (fun i => (results i).published)
  (published.mergeSort (fun a b => decide (a.1 ≤ b.1))).map Prod.snd

theorem partitions (ops : List (Write w)) (workers : List Worker) (results : Worker → WorkerResult)
    (unique : (workers.map Fin.val).Nodup) (covers : ∀ op ∈ ops, op.worker ∈ workers.map Fin.val)
    (runs : ∀ i ∈ workers, worker feed ops i.val = some (results i)) :
    (ops.map (fun op => op.old.toNat)).Perm (workers.flatMap (fun i => (results i).completed.tickets)) ∧
    (lockReports results workers).Perm (workers.flatMap (fun i => (results i).completed.reports)) := by
  have tickets := (MoveList.ConcurrentTrace.partition ops _ unique covers).map (fun op => op.old.toNat)
  have reports := (List.mergeSort_perm (workers.flatMap (fun i => (results i).published))
    (fun a b => decide (a.1 ≤ b.1))).map Prod.snd
  rw [List.map_flatMap, List.flatMap_map] at tickets
  rw [List.map_flatMap] at reports
  constructor
  · have eq := List.map_congr_left fun i mem => (worker_observed (runs i mem)).1
    rw [List.flatMap_def, eq]
    exact tickets
  · have eq := List.map_congr_left fun i mem => (worker_observed (runs i mem)).2
    rw [List.flatMap_def, ← eq]
    exact reports

/-- Assemble the two partitions and worker loops from one modification order. The scheduler
    supplies releases, the recorded stop observations, and the owner/booked participant list. -/
def dispatches (run : MoveList.ConcurrentTrace.Execution len start s) (feed : Feed)
    (workers : List Worker) (results : Worker → WorkerResult)
    (history : Threading.CutoffVisibility.History) (wait : Threading.AtomicWord.Event)
    (actions : List Threading.SplitCompletion.Action) (owner : Worker)
    (check : Threading.SplitReturn.Observation.OwnerCheck history) (localCutoff : Prop)
    (unique : (workers.map Fin.val).Nodup) (covers : ∀ op ∈ s.writes, op.worker ∈ workers.map Fin.val)
    (participant : ∀ i ∈ workers, i = owner ∨ ∃ e, .book i e ∈ actions)
    (runs : ∀ i ∈ workers, worker feed s.writes i.val = some (results i))
    (released : ∀ i, Threading.SplitReturn.Released actions i → i ∈ workers)
    (ending : ∀ i, Threading.SplitReturn.Released actions i → (results i).completed.departure = .exhausted →
      Nonempty (Drained len start s i.val))
    (cut : ∀ i, Threading.SplitReturn.Released actions i → (results i).completed.departure = .cutoff → localCutoff)
    (cancel : ∀ i, Threading.SplitReturn.Released actions i → (results i).completed.departure = .cancelled →
      localCutoff ∨ Threading.SplitReturn.Observation.ExternalStop history check wait) :
    Threading.SplitReturn.Dispatches history wait actions owner len s.current
      ((s.writes.map Write.old).map BitVec.toNat) (lockReports results workers) check localCutoff := by
  obtain ⟨tickets,reports⟩ := partitions s.writes workers results unique covers runs
  exact ⟨fun i => (results i).completed,workers,participant,
    by simpa [Function.comp_def] using tickets,reports,
    fun i rel => worker_finished run (runs i (released i rel)) (ending i rel),cut,cancel⟩

end NeuralReversi.PvsParallel.SplitWorkerMir.Iterator

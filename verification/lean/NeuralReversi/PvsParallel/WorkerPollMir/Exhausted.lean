module
public import NeuralReversi.PvsParallel.WorkerPollMir.Dispatch
public import NeuralReversi.PvsParallel.NativeSplit

@[expose] public section

namespace NeuralReversi.PvsParallel.WorkerPollMir

open MoveList.ConcurrentTrace (State Write forWorker)
open SplitWorkerMir (search commit)
open SplitWorkerMir.Iterator (LockedCall WorkerResult acquired retire worker Drained)


/-- A cutoff or stopped locked ticket ends the native function immediately. An
    extra physical CAS cannot turn a terminating prefix into an exhausted replay. -/
private theorem retire_prefix (calls extra : List LockedCall) (first : retire calls=some a)
    (whole : retire (calls++extra)=some b) (exhausted : b.completed.departure=.exhausted) :
    a.completed.departure=.exhausted := by
  induction calls generalizing a b with
  | nil => cases first;rfl
  | cons c rest ih =>
    simp only [List.cons_append,retire] at first whole
    obtain ⟨body,searched,small⟩:=Option.bind_eq_some_iff.mp first
    obtain ⟨body2,more,large⟩:=Option.bind_eq_some_iff.mp whole
    rw [searched] at more
    cases more
    cases tag : (commit c.ticket.input c.ticket.locked body.chosen).exit with
    | again =>
      simp only [tag] at small large
      obtain ⟨u,past,eq⟩:=Option.bind_eq_some_iff.mp small
      obtain ⟨v,larger,eq'⟩:=Option.bind_eq_some_iff.mp large
      cases eq;cases eq'
      exact ih (a:=u) (b:=v) past larger exhausted
    | stopped | cutoff =>
      simp only [tag] at large
      split at large <;> cases large
      cases exhausted

theorem drained_of_returned {s : State w} (trace : Trace len start s receipts)
    (run : Returned code 0 marks) (matched : polls marks=receipts.filter (fun r => r.worker==i))
    (fields : Fields feed marks) (replayed : worker feed s.writes i=some result)
    (exhausted : result.completed.departure=.exhausted) : Nonempty (Drained len start s i) := by
  have language : play .poll marks=some .done := run.shape phase control_sites
  obtain ⟨cs,r,items,done,_,_,physical⟩:=summarize marks language fields
  have op:=trace.account i
  change forWorker s.writes i=(receipts.filter (fun r => r.worker==i)).filterMap Receipt.ticket++_ at op
  rw [←matched] at op
  unfold worker at replayed
  rw [op,List.mapM_append,items] at replayed
  obtain ⟨all,next,retired⟩:=Option.bind_eq_some_iff.mp replayed
  change (List.mapM (acquired feed) (pendingTicket (s.localAt i))).bind (fun tail => some (cs++tail))=some all at next
  obtain ⟨tail,_,eq⟩:=Option.bind_eq_some_iff.mp next
  cases eq
  have why:=retire_prefix cs tail done retired exhausted
  obtain ⟨thread,opening,seen,here⟩:=physical why
  rw [matched,List.mem_filter] at here
  have id:thread=i:=beq_iff_eq.mp here.2
  subst thread
  exact trace.drained here.1

end NeuralReversi.PvsParallel.WorkerPollMir

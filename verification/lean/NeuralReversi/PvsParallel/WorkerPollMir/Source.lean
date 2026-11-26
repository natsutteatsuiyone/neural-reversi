import NeuralReversi.PvsParallel.WorkerPollMir.Exhausted

namespace NeuralReversi.PvsParallel.WorkerPollMir

open MoveList.ConcurrentTrace (Write)
open SplitWorkerMir (Input Locked Word Commit search commit)
open SplitWorkerMir.Iterator (Feed LockedCall acquired)


inductive Register (w : Nat)
  | poll
  | claimed (ticket : Write w)
  | underLock (call : LockedCall) (body : SplitWorkerMir.Searched)
  | done

private theorem chosen_exit (i : Input) (words : Locked) (v : Word) :
    (commit i words v).exit = if words.stopped then .stopped else
      if (commit i words v).cut then .cutoff else .again := by
  rw [SplitWorkerMir.commit_eq]
  cases words.stopped <;> rfl

/-- The mutable cutoff tested in the final MIR block is the same local used by
    the extracted commit code. The earlier stop branch uses the actual locked
    read. Rank comes directly from the journal lookup at the lock instruction. -/
inductive Native (feed : Feed) : Block → Register w → List (Mark w) → Prop
  | done : code pc=.done → Native feed pc .done []
  | jump : code pc=.jump dest → next∈dest → Native feed next reg xs → Native feed pc reg xs
  | issued : code pc=.next hit empty → Native feed hit (.claimed op) rest →
      Native feed pc .poll (.poll (.some op)::rest)
  | empty : code pc=.next hit empty → Native feed empty .done rest →
      Native feed pc .poll (.poll (.none i opening seen)::rest)
  | lock : code pc=.lock next → acquired feed op=some call → search call.ticket.input=some body →
      Native feed next (.underLock call body) rest → Native feed pc (.claimed op) (.lock call.order::rest)
  | running : code pc=.stop yes no → call.ticket.locked.stopped=false →
      Native feed yes (.underLock call body) rest → Native feed pc (.underLock call body) rest
  | stopped : code pc=.stop yes no → call.ticket.locked.stopped=true → Native feed no .done rest →
      Native feed pc (.underLock call body) (.leave .stopped::rest)
  | again : code pc=.cutoff yes no → call.ticket.locked.stopped=false →
      (commit call.ticket.input call.ticket.locked body.chosen).cut=false → Native feed yes .poll rest →
      Native feed pc (.underLock call body) (.leave .again::rest)
  | cutoff : code pc=.cutoff yes no → call.ticket.locked.stopped=false →
      (commit call.ticket.input call.ticket.locked body.chosen).cut=true → Native feed no .done rest →
      Native feed pc (.underLock call body) (.leave .cutoff::rest)

theorem Native.control (run : Native feed pc reg marks) : Returned code pc marks := by
  induction run with
  | done h => exact .done h
  | jump eq member _ ih => exact .jump eq member ih
  | issued eq _ ih => exact .issued eq ih
  | empty eq _ ih => exact .empty eq ih
  | lock eq _ _ _ ih => exact .lock eq ih
  | running eq _ _ ih => exact .running eq ih
  | stopped eq _ _ ih => exact .stopped eq ih
  | again eq _ _ _ ih => exact .again eq ih
  | cutoff eq _ _ _ ih => exact .cutoff eq ih

private inductive Operand (feed : Feed) : Register w → Mark w → Register w → Prop
  | issued : Operand feed .poll (.poll (.some op)) (.claimed op)
  | empty : Operand feed .poll (.poll (.none i op seen)) .done
  | lock : acquired feed op=some call → search call.ticket.input=some body →
      Operand feed (.claimed op) (.lock call.order) (.underLock call body)
  | leave : (commit call.ticket.input call.ticket.locked body.chosen).exit=route →
      Operand feed (.underLock call body) (.leave route) next

private inductive Operands (feed : Feed) : Register w → List (Mark w) → Prop
  | nil : Operands feed reg []
  | cons : Operand feed reg m after → Operands feed after xs → Operands feed reg (m::xs)

private theorem Native.operands (run : Native feed pc reg marks) : Operands feed reg marks := by
  induction run with
  | done => exact .nil
  | jump _ _ _ ih | running _ _ _ ih => exact ih
  | issued _ _ ih => exact .cons .issued ih
  | empty _ _ ih => exact .cons .empty ih
  | lock _ op body _ ih => exact .cons (.lock op body) ih
  | stopped _ stop _ ih => exact .cons (.leave (by rw [chosen_exit,stop];rfl)) ih
  | again _ stop flag _ ih | cutoff _ stop flag _ ih =>
    exact .cons (.leave (by rw [chosen_exit,stop,flag];rfl)) ih

private theorem Operands.after (xs : List (Mark w)) {ys : List (Mark w)}
    (run : Operands feed reg (xs++ys)) : ∃ next, Operands feed next ys := by
  induction xs generalizing reg with
  | nil => exact ⟨reg,run⟩
  | cons a rest ih => cases run with | cons _ h => exact ih h

theorem Native.fields (run : Native feed pc reg marks) : Fields feed marks := by
  intro earlier op rank endOfMove tail same
  obtain ⟨next,observed⟩:=Operands.after earlier (show Operands feed reg (earlier++(.poll (.some op):: .lock rank:: .leave endOfMove::tail)) from by simpa [same,List.append_assoc] using run.operands)
  cases observed with
  | cons first rest =>
    cases first
    cases rest with
    | cons second following =>
      cases second with
      | lock claim result =>
        cases following with
        | cons last _ =>
          cases last with | leave eq => exact ⟨_,_,claim,rfl,result,eq⟩

/-- Uses the actual normal worker path. Passing the concrete journal feed prevents
    a separately supplied report, exit classification or drained ticket. -/
theorem exhausted_ticket {t : NativeSplit.Tickets rc pv beta} {receipts : List (Receipt t.width)}
    (iterator : Trace rc.length t.start t.state receipts) (i : Threading.SplitCompletion.Worker) (here : i∈t.workers)
    {marks : List (Mark t.width)} (ran : Native (IteratorFrames.source t.journal t.stream) 0 .poll marks)
    (reads : polls marks=receipts.filter (fun r => r.worker==i.val))
    (exhausted : (t.result i).completed.departure=.exhausted) :
    Nonempty (SplitWorkerMir.Iterator.Drained rc.length t.start t.state i.val) :=
  drained_of_returned iterator ran.control reads ran.fields (t.ran i here) exhausted

end NeuralReversi.PvsParallel.WorkerPollMir

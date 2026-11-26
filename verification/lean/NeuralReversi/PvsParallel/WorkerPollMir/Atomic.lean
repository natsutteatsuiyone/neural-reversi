import NeuralReversi.MoveList.ConcurrentExecution

namespace NeuralReversi.PvsParallel.WorkerPollMir

open MoveList.ConcurrentTrace
open MoveList.Concurrent (advance)

/-- A completed call of ConcurrentMoveIterator::next, indexed by its opening
    event and by the physical word returned by the atomic operation. -/
inductive Receipt (w : Nat)
  | some (op : Write w)
  | none (worker opening : Nat) (seen : BitVec w)
  deriving DecidableEq

def Receipt.worker : Receipt w → Nat
  | .some op => op.worker
  | .none thread .. => thread

def Receipt.ticket : Receipt w → Option (Write w)
  | .some op => Option.some op
  | .none .. => Option.none

/-- Individual primitive steps of ConcurrentExecution, with the return value
    recorded only when the native call returns it to the worker. -/
inductive Edge (len : Nat) (start : BitVec w) : State w → Option (Receipt w) → State w → Prop
  | load (s : State w) (i source opening : Nat) (seen : BitVec w) (free : s.localAt i = .idle)
      (read : (words start s.writes)[source]? = some seen) :
      Edge len start s none {s with localAt := replace s.localAt i (.testing opening seen)}
  | failed (s : State w) (i opening source : Nat) (prev wanted seen : BitVec w)
      (active : s.localAt i = .testing opening prev) (proposed : advance len prev = some wanted)
      (read : (words start s.writes)[source]? = some seen) :
      Edge len start s none {s with localAt := replace s.localAt i (.testing opening seen)}
  | success (s : State w) (i opening : Nat) (wanted : BitVec w)
      (active : s.localAt i = .testing opening s.current) (proposed : advance len s.current = some wanted) :
      Edge len start s none ⟨wanted, s.writes ++ [⟨i, opening, s.current, wanted⟩],
        replace s.localAt i (.held ⟨i, opening, s.current, wanted⟩)⟩
  | decline (s : State w) (i opening : Nat) (prev : BitVec w)
      (active : s.localAt i = .testing opening prev) (endOfList : advance len prev = none) :
      Edge len start s none {s with localAt := replace s.localAt i (.drained opening prev)}
  | some (s : State w) (i : Nat) (op : Write w) (held : s.localAt i = .held op) :
      Edge len start s (.some (.some op)) {s with localAt := replace s.localAt i .idle}
  | none (s : State w) (i opening : Nat) (seen : BitVec w) (drained : s.localAt i = .drained opening seen) :
      Edge len start s (.some (.none i opening seen)) {s with localAt := replace s.localAt i .idle}

theorem Edge.execution (e : Edge len start s obs z) (run : MoveList.ConcurrentTrace.Execution len start s) :
    MoveList.ConcurrentTrace.Execution len start z := by
  cases e with
  | load _ _ opening _ free rd => exact .load run free rd opening
  | failed _ _ _ _ _ _ active proposed rd => exact .failed run active proposed rd
  | success _ _ _ active proposed => exact .success run active proposed
  | decline _ _ _ active stop => exact .decline run active stop
  | some _ _ held => exact .returnedSome run held
  | none _ _ _ drained => exact .returnedNone run drained

inductive Trace (len : Nat) (start : BitVec w) : State w → List (Receipt w) → Prop
  | init : Trace len start ⟨start, [], fun _ => .idle⟩ []
  | step : Trace len start s receipts → Edge len start s emitted z →
      Trace len start z (receipts ++ emitted.toList)

theorem Trace.execution (h : Trace len start s rs) : MoveList.ConcurrentTrace.Execution len start s := by
  induction h with
  | init => exact .init
  | step _ edge ih => exact edge.execution ih

theorem recorded {s : State w} (run : MoveList.ConcurrentTrace.Execution len start s) :
    ∃ rs, Trace len start s rs := by
  induction run with
  | init => exact ⟨[], .init⟩
  | load _ free read opening ih =>
    obtain ⟨_, prev⟩ := ih
    exact ⟨_, prev.step (.load _ _ _ opening _ free read)⟩
  | failed _ active proposed read ih =>
    obtain ⟨_, prev⟩ := ih
    exact ⟨_, prev.step (.failed _ _ _ _ _ _ _ active proposed read)⟩
  | success _ active proposed ih =>
    obtain ⟨_, prev⟩ := ih
    exact ⟨_, prev.step (.success _ _ _ _ active proposed)⟩
  | decline _ active declined ih =>
    obtain ⟨_, prev⟩ := ih
    exact ⟨_, prev.step (.decline _ _ _ _ active declined)⟩
  | returnedSome _ active ih =>
    obtain ⟨_, prev⟩ := ih
    exact ⟨_, prev.step (.some _ _ _ active)⟩
  | returnedNone _ ended ih =>
    obtain ⟨_, prev⟩ := ih
    exact ⟨_, prev.step (.none _ _ _ _ ended)⟩

theorem Edge.prefix (e : Edge len start s obs z) : s.writes <+: z.writes := by
  cases e <;> simp

theorem Trace.drained (h : Trace len start s rs) (seen : Receipt.none i opening value ∈ rs) :
    Nonempty (Drained len start s i) := by
  induction h with
  | init => cases seen
  | @step prev before op after trace edge ih =>
    rcases List.mem_append.mp seen with earlier | atReturn
    · obtain ⟨witness⟩ := ih earlier
      exact ⟨{witness with prior := witness.prior.trans edge.prefix}⟩
    · cases edge <;> simp only [Option.toList_none, List.not_mem_nil, Option.toList_some, List.mem_singleton] at atReturn
      · cases atReturn
      · cases atReturn
        exact ⟨⟨_, trace.execution, List.prefix_refl _, _, _, by assumption⟩⟩

def pendingTicket : Local w → List (Write w)
  | .held op => [op]
  | _ => []

def returnedTickets (i : Nat) (rs : List (Receipt w)) : List (Write w) :=
  (rs.filter (fun r => r.worker == i)).filterMap Receipt.ticket

theorem Edge.account {start : BitVec w} {s z : State w} (edge : Edge len start s event z)
    (run : Execution len start s)
    (i : Nat) (rs : List (Receipt w))
    (valid : forWorker s.writes i = returnedTickets i rs ++ pendingTicket (s.localAt i)) :
    forWorker z.writes i = returnedTickets i (rs ++ event.toList) ++ pendingTicket (z.localAt i) := by
  cases edge with
  | load actor _ _ _ free _ | failed actor _ _ _ _ _ free _ _ | decline actor _ _ free _ =>
    by_cases eq : i = actor
    · subst i
      simpa [replace, returnedTickets, free, pendingTicket] using valid
    · simpa [replace, returnedTickets, eq] using valid
  | success actor opening wanted active _ =>
    by_cases eq : i = actor
    · subst i
      simp [returnedTickets, pendingTicket, active] at valid
      simpa [forWorker, replace, returnedTickets, pendingTicket, active, valid] using congrArg (· ++ [⟨actor, opening, s.current, wanted⟩]) valid
    · simpa [forWorker, replace, returnedTickets, eq, Ne.symm eq] using valid
  | some actor op slot =>
    have hit := run.local_valid actor
    rw [slot] at hit
    have id : op.worker = actor := hit.1
    by_cases eq : i = actor
    · subst i
      simpa [returnedTickets, replace, pendingTicket, slot, Receipt.worker, Receipt.ticket, id] using valid
    · simpa [returnedTickets, replace, pendingTicket, eq, Ne.symm eq, Receipt.worker, Receipt.ticket, id] using valid
  | none actor opening value slot =>
    by_cases eq : i = actor
    · subst i
      have empty : [Receipt.none actor opening value].filterMap Receipt.ticket = [] := rfl
      simpa [returnedTickets, replace, pendingTicket, slot, Receipt.worker, empty] using valid
    · simpa [returnedTickets, replace, pendingTicket, eq, Ne.symm eq, Receipt.worker, Receipt.ticket] using valid

theorem Trace.account (trace : Trace len start s rs) (i : Nat) :
    forWorker s.writes i = returnedTickets i rs ++ pendingTicket (s.localAt i) := by
  induction trace with
  | init => rfl
  | step run edge ih => exact edge.account run.execution i _ ih

end NeuralReversi.PvsParallel.WorkerPollMir

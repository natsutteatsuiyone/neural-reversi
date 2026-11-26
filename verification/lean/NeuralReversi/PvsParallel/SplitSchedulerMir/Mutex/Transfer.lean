import NeuralReversi.PvsParallel.SplitSchedulerMir.Mutex

namespace NeuralReversi.PvsParallel.SplitSchedulerMir.Mutex

open Threading
open AtomicWord (Event SequencedBefore)
open SpinLock (Operation)
open CutoffVisibility (HappensBefore)

private def releasedAfter (start : Event) : Operation → Bool
  | .release u => start.worker == u.worker && start.step < u.step
  | _ => false

private def acquiredBefore (stop : Event) : Operation → Bool
  | .acquire a => a.worker == stop.worker && a.step < stop.step
  | _ => false

/-- Past the first eligible release, a later acquire may be separated from it
    by any number of critical sections on this mutex. -/
def crosses (start stop : Event) : List Operation → Bool
  | [] => false
  | op :: tail => if releasedAfter start op then tail.any (acquiredBefore stop) else crosses start stop tail

private theorem eligible (ok : xs.any (acquiredBefore stop)) :
    ∃ gap a rest, xs = gap ++ .acquire a :: rest ∧ SequencedBefore a stop := by
  obtain ⟨op, member, selected⟩ := List.any_eq_true.mp ok
  cases op with
  | release => contradiction
  | acquire a =>
    obtain ⟨pre,post,split⟩ := List.append_of_mem member
    exact ⟨pre,a,post,split,by simpa [acquiredBefore,SequencedBefore] using selected⟩

theorem crosses_iff (start stop : Event) (ops : List Operation) : crosses start stop ops = true ↔
    ∃ pre u gap a rest, ops = pre ++ .release u :: (gap ++ .acquire a :: rest) ∧
      SequencedBefore start u ∧ SequencedBefore a stop := by
  induction ops with
  | nil => simp [crosses]
  | cons op ops ih =>
    constructor
    · intro ok
      unfold crosses at ok
      split at ok
      · rename_i chosen
        cases op with
        | acquire => contradiction
        | release u =>
          obtain ⟨gap,a,rest,eq,last⟩ := eligible ok
          exact ⟨[],u,gap,a,rest,by simp [eq],by simpa [releasedAfter,SequencedBefore] using chosen,last⟩
      · obtain ⟨pre,u,gap,a,rest,eq,first,last⟩ := ih.mp ok
        exact ⟨op::pre,u,gap,a,rest,by simp [eq],first,last⟩
    · rintro ⟨pre,u,gap,a,rest,eq,first,last⟩
      cases pre with
      | nil =>
        cases eq
        have enters : acquiredBefore stop (.acquire a) := by simpa [acquiredBefore,SequencedBefore] using last
        have leaves : releasedAfter start (.release u) := by simpa [releasedAfter,SequencedBefore] using first
        simp [crosses,leaves,enters]
      | cons h pre =>
        cases eq
        unfold crosses
        split
        · apply List.any_eq_true.mpr
          exact ⟨.acquire a,by simp,by simpa [acquiredBefore,SequencedBefore] using last⟩
        · exact ih.mpr ⟨pre,u,gap,a,rest,rfl,first,last⟩

def carries (ops : List Operation) (start stop : Event) : Bool :=
  (guards ops).isSome && crosses start stop ops

theorem Capture.transfer (c : Capture history) (ok : carries c.ops start stop) :
    Nonempty (SpinLock.OrderedAccess history start stop) := by
  obtain ⟨formed,path⟩ := Bool.and_eq_true_iff.mp ok
  obtain ⟨pairs,good⟩ := Option.isSome_iff_exists.mp formed
  obtain ⟨pre,u,gap,a,rest,split,first,last⟩ := (crosses_iff ..).mp path
  exact ⟨⟨u,a,first,⟨c.trace good,pre,gap,rest,split⟩,last⟩⟩

end NeuralReversi.PvsParallel.SplitSchedulerMir.Mutex

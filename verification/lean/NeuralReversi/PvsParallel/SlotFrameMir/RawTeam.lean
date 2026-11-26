import NeuralReversi.PvsParallel.SlotFrameMir.RawNative

namespace NeuralReversi.PvsParallel.SlotFrameMir.RawProjection

open Threading
open AtomicWord (Write Event)
open SplitCompletion (Action MaskState Allowed Legal step)

def target : Action → Option Worker
  | .book w _ => some w
  | .finish .. => none

def roster (g : RawMask ws) := (g.owner::g.actions.filterMap target).eraseDups

theorem roster_mem (g : RawMask ws) (w : Worker) :
    w∈roster g ↔ w=g.owner ∨ ∃ e,.book w e ∈ g.actions := by
  simp only [roster,List.mem_eraseDups,List.mem_cons,List.mem_filterMap]
  constructor
  · rintro (eq | ⟨op,here,found⟩)
    · exact Or.inl eq
    · cases op with
      | finish => cases found
      | book _ e => cases found;exact Or.inr ⟨e,here⟩
  · rintro (eq | ⟨e,here⟩)
    · exact Or.inl eq
    · exact Or.inr ⟨.book w e,here,rfl⟩

private theorem origin (s : MaskState) (ops : List Action) (w : Worker) (e : Event)
    (legal : Legal s ops) (finished : .finish w e ∈ ops) :
    s.has w=true ∨ ∃ ev,.book w ev ∈ ops := by
  induction ops generalizing s with
  | nil => cases finished
  | cons a rest ih =>
    rcases List.mem_cons.mp finished with eq | later
    · subst a;exact Or.inl legal.1
    · rcases ih (step s a) legal.2 later with active | ⟨ev,here⟩
      · cases a with
        | finish f stamp =>
          rw [SplitCompletion.finish_has,Bool.and_eq_true] at active
          exact Or.inl active.1
        | book f stamp =>
          rw [SplitCompletion.book_has,Bool.or_eq_true] at active
          rcases active with was | yes
          · exact Or.inl was
          · have eq:=of_decide_eq_true yes
            subst f;exact Or.inr ⟨stamp,List.mem_cons_self ..⟩
      · exact Or.inr ⟨ev,List.mem_cons_of_mem _ here⟩

theorem effective_roster (g : RawMask ws) (closed : ClosedAfterFirst g) (w : Worker) (effective : Effective g w) :
    w∈roster g := by
  obtain ⟨e,mem⟩:=(released_iff g closed w).2 effective
  rcases origin _ _ w e (activation g closed).legal mem with owned | ⟨book,here⟩
  · change (Bitboard.squareMask g.owner).getLsbD w.val=true at owned
    rw [Bitboard.squareMask_bit] at owned
    exact (roster_mem g w).2 (.inl (Fin.ext (of_decide_eq_true owned)))
  · obtain ⟨i,_,eq⟩:=List.mem_map.mp here
    exact (roster_mem g w).2 (.inr ⟨book,eq ▸ List.get_mem g.actions i⟩)

/-- A roster equality with the recorded physical setup suffices for both
    directions of participation; no worker-release classification is required. -/
theorem team_of_roster {g : RawMask ws} {t : NativeSplit.Tickets rc pv beta}
    (p : PhysicalOwner g hist time slot t) (same : t.workers.Perm (roster g))
    (exhausted : ∀ i ∈ t.workers, (t.result i).completed.departure=.exhausted →
      Nonempty (SplitWorkerMir.Iterator.Drained rc.length t.start t.state i.val))
    (stopped : let calls:=IteratorFields.chronological t.state.writes t.journal t.stream t.workers
      let event:=p.exit.clock.atStep (3+p.exit.log.length)
      ∀ c∈calls, c.ticket.locked.stopped=true → Nonempty (SplitWorkerMir.Stop.Poll p.check event calls) ∨
        (∀ obs, p.check.shared=some obs → SplitReturn.Observation.AbortCause hist obs event)) : Team p where
  participant i mem:=(roster_mem g i).1 (same.mem_iff.mp mem)
  released i effective:=same.mem_iff.mpr (effective_roster g p.gates.closed i effective)
  exhausted i effective:=exhausted i (same.mem_iff.mpr (effective_roster g p.gates.closed i effective))
  stopped:=stopped

end NeuralReversi.PvsParallel.SlotFrameMir.RawProjection

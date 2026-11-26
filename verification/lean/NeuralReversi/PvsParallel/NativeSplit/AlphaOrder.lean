import NeuralReversi.PvsParallel.NativeSplit

namespace NeuralReversi.PvsParallel.NativeSplit.Tickets

open Threading
open AtomicWord (Write SequencedBefore)
open SplitWorkerMir.FieldStores (Physical)
open SplitWorkerMir.Iterator (LockedCall)

variable (t : Tickets rc pv beta)

/-- Each physical alpha store belongs to the guard of the ticket at that journal rank. -/
private theorem guarded_alpha {w : Write} (inside : k < t.journal.length) (stored : Physical.writeAlpha w ∈ t.traffic k) :
    ∃ obs, t.journal[k]? = some obs ∧ SequencedBefore obs.acquired w.event := by
  let calls := IteratorFields.chronological t.state.writes t.journal t.stream t.workers
  have ranks : calls.map LockedCall.order = List.range t.journal.length := IteratorFields.ranks t.replay
  obtain ⟨c, member, rank⟩ := List.mem_map.mp (ranks.symm ▸ List.mem_range.mpr inside)
  obtain ⟨op, issued, got⟩ := IteratorFields.acquired_at t.workers t.result t.ran member
  obtain ⟨obs, _, lock, _, _, _⟩ := IteratorFrames.acquire_control got
  have row := (IteratorFrames.lock_at lock).1
  rw [rank] at row
  refine ⟨obs, row, ?_⟩
  have checked := List.all_eq_true.mp t.traced op issued
  cases found : SplitSchedulerMir.WorkerFields.locate t.dispatches obs with
  | none => simp [SplitSchedulerMir.WorkerFields.matchesTicket, lock, found] at checked
  | some a =>
    cases body : SplitWorkerMir.search c.ticket.input with
    | none => simp [SplitSchedulerMir.WorkerFields.matchesTicket, lock, got, body] at checked
    | some b =>
      have aligned : a.aligned obs b.steps := by
        simpa [SplitSchedulerMir.WorkerFields.matchesTicket, lock, found, got, body] using checked
      apply a.acquire_before_store body aligned (op := .writeAlpha w) _ (Or.inl rfl)
      simpa [traffic, SplitSchedulerMir.WorkerFields.traffic, row, found] using stored

private theorem next_cell {original pre current suffix post : List α}
    (parts : original = pre ++ current ++ suffix ++ post) (atCell : original[n]? = some w)
    (position : n = pre.length + current.length) :
    w ∈ suffix ∨ n = pre.length + current.length + suffix.length := by
  cases suffix with
  | nil => exact Or.inr (by simpa using position)
  | cons a rest =>
    have atHead : (pre ++ current ++ (a :: rest) ++ post)[(pre ++ current).length]? = some a := by
      simp [List.append_assoc]
    rw [parts, position, ← List.length_append] at atCell
    exact Or.inl (List.mem_cons.mpr (Or.inl (Option.some.inj (atCell.symm.trans atHead))))

/-- The alpha modification just beyond the scout's window is in this or a subsequent
    critical section, or is the first modification made by a later use of the slot. -/
theorem next_alpha (part : t.reports = earlier ++ r :: later) (pre post : List Write)
    (i : Fin (pre ++ t.final.alpha ++ post).length)
    (pos : i.val = ((t.interval part).last pre post).val + 1) :
    (∃ k obs, (t.post part).rank ≤ k ∧ t.journal[k]? = some obs ∧
      SequencedBefore obs.acquired ((pre ++ t.final.alpha ++ post).get i).event) ∨
      i.val = pre.length + t.final.alpha.length := by
  let x := t.post part
  let slice := t.interval part
  let stores := fun k => (t.traffic k).filterMap Physical.alpha?
  let suffix := ((List.range t.journal.length).drop x.rank).flatMap stores
  have facts := (IteratorReports.prefixLog (IteratorFields.scout_stores rc t.workers t.result t.ran
    t.seedAlpha t.seedBest t.seedMove t.replay x part)).property
  have rest : t.final.alpha = slice.current ++ suffix := by
    change t.final.alpha = slice.current ++ ((List.range t.journal.length).drop x.rank).flatMap stores
    rw [← IteratorFields.ranks t.replay, ← List.map_drop, List.flatMap_map]
    exact facts.2.1
  have positive := slice.seeded.length_le
  change 1 ≤ slice.current.length at positive
  change i.val = pre.length + slice.current.length - 1 + 1 at pos
  have atCell : (pre ++ t.final.alpha ++ post)[i.val]? = some ((pre ++ t.final.alpha ++ post).get i) :=
    List.getElem?_eq_getElem i.isLt
  have parts : pre ++ t.final.alpha ++ post = pre ++ slice.current ++ suffix ++ post := by
    simpa only [List.append_assoc] using congrArg (fun words => pre ++ words ++ post) rest
  rcases next_cell parts atCell (by omega) with hit | endAt
  · obtain ⟨k, remaining, store⟩ := List.mem_flatMap.mp hit
    obtain ⟨j, found⟩ := List.mem_iff_getElem?.mp remaining
    rw [List.getElem?_drop] at found
    have bound := (List.getElem?_eq_some_iff.mp found).1
    simp only [List.length_range] at bound
    rw [List.getElem?_range bound] at found
    cases found
    obtain ⟨op, ran, word⟩ := List.mem_filterMap.mp store
    have kind : op = .writeAlpha ((pre ++ t.final.alpha ++ post).get i) := by
      cases op <;> simp only [Physical.alpha?, Option.some.injEq, reduceCtorEq] at word <;> try contradiction
      exact congrArg Physical.writeAlpha word
    obtain ⟨obs, row, held⟩ := t.guarded_alpha bound (kind ▸ ran)
    exact Or.inl ⟨_, obs, Nat.le_add_right x.rank j, row, held⟩
  · have len := congrArg List.length rest
    simp only [List.length_append] at len
    exact Or.inr (by omega)

end NeuralReversi.PvsParallel.NativeSplit.Tickets

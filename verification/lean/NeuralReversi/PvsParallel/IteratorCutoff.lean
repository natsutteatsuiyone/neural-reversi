import NeuralReversi.PvsParallel.IteratorFields

namespace NeuralReversi.PvsParallel.IteratorFields

open SplitWorkerMir
open Iterator (LockedCall)
open Threading.SplitResult (Report)
open Threading.SplitReturn (LocalCutoff)

private theorem retired_cut (cs : List LockedCall) (ran : Iterator.retire cs = some result)
    (cut : result.completed.departure = .cutoff) :
    ∃ c ∈ cs, ∃ b, search c.ticket.input = some b ∧ (commit c.ticket.input c.ticket.locked b.chosen).exit = .cutoff := by
  induction cs generalizing result with
  | nil => cases ran; contradiction
  | cons c rest ih =>
    unfold Iterator.retire at ran
    obtain ⟨b,searched,more⟩ := Option.bind_eq_some_iff.mp ran
    dsimp only at more
    cases h : (commit c.ticket.input c.ticket.locked b.chosen).exit with
    | cutoff => exact ⟨c,by simp,b,searched,h⟩
    | stopped =>
      cases rest <;> simp [h] at more
      subst result
      contradiction
    | again =>
      rw [h] at more
      obtain ⟨_,done,same⟩ := Option.bind_eq_some_iff.mp more
      cases same
      obtain ⟨actual,inside,b,search,exit⟩ := ih done cut
      exact ⟨actual,List.mem_cons_of_mem _ inside,b,search,exit⟩

private theorem split_at (pre post : List FieldStores.Raw)
    (done : FieldStores.replay pv beta encode cells initial (pre ++ post) = some final) :
    ∃ mid, FieldStores.replay pv beta encode cells initial pre = some mid ∧
      FieldStores.replay pv beta encode cells mid post = some final := by
  induction pre generalizing initial with
  | nil => exact ⟨_,rfl,done⟩
  | cons hd tl ih =>
    obtain ⟨r,read,tail⟩ := Option.bind_eq_some_iff.mp done
    obtain ⟨mid,earlier,later⟩ := ih tail
    exact ⟨mid,by simp [FieldStores.replay,read,earlier],later⟩

private theorem extends_reports (cs : List FieldStores.Raw)
    (done : FieldStores.replay pv beta encode cells initial cs = some final) : initial.reports <+: final.reports := by
  induction cs generalizing initial with
  | nil => cases done; exact List.prefix_refl _
  | cons hd tl ih =>
    obtain ⟨r,_,tail⟩ := Option.bind_eq_some_iff.mp done
    exact (List.prefix_append _ r.report.toList).trans (ih tail)

theorem call_cutoff (a : FieldStores.low32 aStore = initial.alpha) (b : FieldStores.low32 bStore = initial.best)
    (m : FieldStores.low8 mStore = encode initial.picked)
    (ok : replay rc ops journal stream workers traffic pv beta encode
      (FieldStores.initialLog initial aStore bStore mStore) = some final)
    (member : c ∈ chronological ops journal stream workers) (searched : search c.ticket.input = some body)
    (exit : (commit c.ticket.input c.ticket.locked body.chosen).exit = .cutoff) :
    LocalCutoff pv beta initial.state (Iterator.lockReports (results ops journal stream) workers) := by
  have returned := worker_returns ok
  have all : final.reports = Iterator.lockReports (results ops journal stream) workers :=
    reports rc ops workers _ returned ok
  rw [← all]
  have operands := traffic_at ok member searched
  obtain ⟨pre,post,whereAt⟩ := List.append_of_mem member
  have ran := field_replay ok
  rw [whereAt,List.map_append,List.map_cons] at ran
  obtain ⟨mid,prefixRun,suffixRun⟩ := split_at _ _ ran
  obtain ⟨r,read,tail⟩ := Option.bind_eq_some_iff.mp suffixRun
  obtain ⟨published,marks⟩ := FieldStores.Raw.cut_report mid.fields c.ticket.locked (traffic c.order)
    body searched exit operands read
  have checked := FieldStores.replay_preserves _ (FieldStores.seed initial aStore bStore mStore a b m) prefixRun
  obtain ⟨later,finish⟩ := extends_reports _ tail
  exact ⟨mid.reports,selected c.ticket.input body,later,
    by simpa [FieldStores.Log.append,published,List.append_assoc] using finish.symm,
    by unfold Threading.SplitReturn.stateAfter; rw [← checked.state]; exact marks⟩

theorem worker_cutoff (a : FieldStores.low32 aStore = initial.alpha) (b : FieldStores.low32 bStore = initial.best)
    (m : FieldStores.low8 mStore = encode initial.picked)
    (ok : replay rc ops journal stream workers traffic pv beta encode
      (FieldStores.initialLog initial aStore bStore mStore) = some final)
    (i : Threading.SplitCompletion.Worker) (here : i ∈ workers)
    (cut : (results ops journal stream i).completed.departure = .cutoff) :
    LocalCutoff pv beta initial.state (Iterator.lockReports (results ops journal stream) workers) := by
  obtain ⟨cs,done,every⟩ := worker_calls (worker_returns ok i here) here
  obtain ⟨c,member,body,searched,exit⟩ := retired_cut cs done cut
  exact call_cutoff a b m ok (every c member) searched exit

end NeuralReversi.PvsParallel.IteratorFields

module
public import NeuralReversi.EndgameSearch.CacheHistory.Payloads.Ownership

@[expose] public section

namespace NeuralReversi.EndgameSearch.CacheHistory.Payloads

open NativeTrace (Worker Cursor Event Raw Capture readWorker workerAt run workerEvents)

private theorem in_pool (p : Pool) (stream : List (Event S)) {c : Capture p S} (parsed : NativeTrace.decode p stream = some c)
    (h : Event.worker n raw ∈ stream) : n < (CacheOwner.poolPlan p.size).length := by
  unfold NativeTrace.decode at parsed
  dsimp only at parsed
  split at parsed <;> try contradiction
  rename_i accepted
  simp only [Bool.not_eq_true',Bool.not_eq_false] at accepted
  simpa using List.all_eq_true.mp accepted (.worker n raw) h

private theorem enclosing (p : Pool) (stream : List (Event S)) (c : Capture p S) (parsed : NativeTrace.decode p stream = some c)
    (event : Event.worker n (.leave t o node score) ∈ stream) (prim : target node = some (table,kind)) :
    let site := Site.mk n ⟨o,t,table,kind⟩
    o ≤ t ∧ ((kind == .store || table == .shallow) = true → ∃ i ∈ intervals c, i.contains site) := by
  obtain ⟨_,completed,allWorkers⟩ := NativeTrace.decode_keeps p stream parsed
  obtain ⟨⟨w,s⟩,member,read⟩ := mapped allWorkers (List.mem_range.mpr (in_pool p stream parsed event))
  unfold readWorker at read
  obtain ⟨actual,boot,read⟩ := Option.bind_eq_some_iff.mp read
  obtain ⟨cursor,ran,ident⟩ := Option.bind_eq_some_iff.mp read
  cases ident
  have id : w.ordinal = n := by
    unfold workerAt at boot
    split at boot <;> cases boot
    rfl
  have raw : Raw.leave t o node score ∈ workerEvents w.ordinal stream := by
    apply List.mem_filterMap.mpr
    exact ⟨_,event,by simp [id]⟩
  obtain ⟨time,h⟩ := worker_site w stream s ran raw (by simp [prim])
  refine ⟨time,fun mustCover => ?_⟩
  have notDeepRead : node ≠ .deepSlot ∧ node ≠ .deepProbe := by
    cases node <;> simp [target] at prim
    all_goals rcases prim with ⟨rfl,rfl⟩; simp at mustCover ⊢
  rcases h notDeepRead with ⟨v,closed,worker,lo,hi⟩ | ⟨f,found,lo,hi⟩
  · have present : v ∈ c.timeline.visits := completed ▸ List.mem_flatMap.mpr ⟨_,member,closed⟩
    exact ⟨⟨v.worker,v.first,v.last⟩,List.mem_append_left _ (List.mem_map.mpr ⟨_,present,rfl⟩),by simp [Interval.contains,worker,id,lo,hi]⟩
  · refine ⟨⟨w.ordinal,f.first,s.clock⟩,List.mem_append_right _ (List.mem_filterMap.mpr ⟨_,member,?_⟩),?_⟩
    · simp [found]
    · simp [Interval.contains,id,lo,hi]

/-- Parsing the compiler-checked call edges and their returns implies complete,
    unique attribution of all store and shallow cache sites over all completed
    and outstanding workers, and at-most-once attribution of deep read-only sites. -/
theorem covered_of_parse (p : Pool) (stream : List (Event S)) (c : Capture p S)
    (parsed : NativeTrace.decode p stream = some c) : covered c stream = true := by
  apply List.all_eq_true.mpr
  intro s represented
  obtain ⟨event,present,site⟩ := List.mem_filterMap.mp represented
  cases event with
  | shared => contradiction
  | worker n ev =>
    cases ev with
    | enter => contradiction
    | leave t o node score =>
      cases found : target node with
      | none => simp [site?,found] at site
      | some pair =>
        obtain ⟨table,kind⟩ := pair
        simp [site?,found] at site
        subst s
        obtain ⟨time,must⟩ := enclosing p stream c parsed present found
        have upper := intervals_unique p stream c parsed ⟨n,o,t,table,kind⟩ time
        dsimp only
        split
        next boundaryOnly =>
          obtain ⟨i,member,hit⟩ := must boundaryOnly
          have included : i ∈ (intervals c).filter (·.contains ⟨n,o,t,table,kind⟩) := List.mem_filter.mpr ⟨member,hit⟩
          have lower := List.length_pos_of_mem included
          exact beq_iff_eq.mpr (by omega)
        next => exact decide_eq_true upper

private theorem parsed_workers (p : Pool) (stream : List (Event S)) (ids : List Nat)
    (ws : List ((w : Worker p) × Cursor w)) (ran : ids.mapM (readWorker p stream) = some ws) :
    ∀ item ∈ ws, NativeTrace.run item.1 stream = some item.2 := by
  induction ids generalizing ws with
  | nil => cases ran; simp
  | cons n ns ih =>
    rw [List.mapM_cons] at ran
    obtain ⟨first,read,tail⟩ := Option.bind_eq_some_iff.mp ran
    obtain ⟨rest,more,eq⟩ := Option.bind_eq_some_iff.mp tail
    cases eq
    intro item mem
    rcases List.mem_cons.mp mem with rfl | later
    · obtain ⟨w,_,read⟩ := Option.bind_eq_some_iff.mp read
      obtain ⟨s,parsed,eq⟩ := Option.bind_eq_some_iff.mp read
      cases eq
      exact parsed
    · exact ih _ more item later

theorem capture_workers (p : Pool) (stream : List (Event S)) (c : Capture p S) (parsed : NativeTrace.decode p stream = some c) :
    ∀ item ∈ c.workers, NativeTrace.run item.1 stream = some item.2 :=
  parsed_workers p stream _ _ (NativeTrace.decode_keeps _ _ parsed).2.2

/-- The native call parser supplies coverage itself. Successful per-visit and
    per-worker array replays on the same register file suffice for the decoder. -/
theorem decode_of_visits (p : Pool) (stream : List (Event S)) (c : Capture p S) (values : Values)
    (parsed : NativeTrace.decode p stream = some c)
    (visits : ∀ v ∈ c.timeline.visits, ∃ a, completeVisit p c.timeline mc fc values stream v = some a)
    (workers : ∀ item ∈ c.workers, ∃ a, pendingWorker p c.timeline mc fc values stream item = some a) :
    ∃ out, decode p mc fc values stream = some out ∧ out.native = c := by
  have total {A B : Type} (xs : List A) (f : A → Option B) (each : ∀ x ∈ xs, ∃ y, f x = some y) :
      ∃ result, xs.mapM f = some result := by
    induction xs with
    | nil => exact ⟨[],rfl⟩
    | cons x rest ih =>
      obtain ⟨y,head⟩ := each x List.mem_cons_self
      obtain ⟨ys,tail⟩ := ih (fun a ha => each a (List.mem_cons_of_mem _ ha))
      exact ⟨y::ys,by simp [List.mapM_cons,head,tail]⟩
  obtain ⟨closed,done_⟩ := total _ _ visits
  obtain ⟨pending,open_⟩ := total _ _ workers
  exact ⟨⟨c,closed,pending⟩,by simp [decode,parsed,covered_of_parse p stream c parsed,done_,open_],rfl⟩

end NeuralReversi.EndgameSearch.CacheHistory.Payloads

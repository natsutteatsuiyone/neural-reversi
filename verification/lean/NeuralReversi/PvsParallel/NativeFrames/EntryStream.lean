import NeuralReversi.PvsParallel.NativeFrames.Returns
import NeuralReversi.EndgameSearch.CacheHistory.NativeTrace.Order

namespace NeuralReversi.PvsParallel.NativeFrames.EntryStream

open EndgameSearch.CacheHistory (Pool)
open EndgameSearch.CacheHistory.NativeTrace (Worker Cursor advance_clock)
open Returns (localEvents)

abbrev Entry := (Nat × Nat) × Args

/-- A physical frame event together with the PVS arguments sampled at entry. -/
structure Sample where
  record : Record
  args : Option Args

def Sample.entry (s : Sample) : Option Entry := match s.record with
  | .localCall worker (.entry opening _ .pvs ..) => s.args.map ((worker,opening),·)
  | _ => none

def inputs (rs : List Sample) (w t : Nat) := (rs.filterMap Sample.entry).lookup (w,t)

def capture (p : Pool) (rs : List Sample) := decode p (inputs rs) (rs.map Sample.record)

private theorem step_time {w : Worker p} (s out : State w)
    (ok : s.advance args e = some out) : s.cursor.clock ≤ e.step ∧ out.cursor.clock = e.step+1 := by
  cases e with
  | entry t site node rec arg =>
    simp only [State.advance] at ok
    split at ok <;> try contradiction
    obtain ⟨n,ran,rest⟩ := Option.bind_eq_some_iff.mp ok
    have h := advance_clock _ _ ran
    split at rest
    · obtain ⟨_,_,rest⟩ := Option.bind_eq_some_iff.mp rest
      split at rest <;> cases rest
      exact h
    · cases rest; exact h
  | scored t word =>
    simp only [State.advance] at ok
    split at ok <;> try contradiction
    split at ok <;> try contradiction
    rename_i cond
    have time : s.cursor.clock ≤ t := by simpa using (Bool.or_eq_false_iff.mp (Bool.eq_false_iff.mpr cond)).1
    obtain ⟨a,_,rest⟩ := Option.bind_eq_some_iff.mp ok
    split at rest
    · cases rest; exact ⟨time,rfl⟩
    · split at rest <;> try contradiction
      obtain ⟨_,_,rest⟩ := Option.bind_eq_some_iff.mp rest
      cases rest; exact ⟨time,rfl⟩
  | leave t opening node boundary normal =>
    simp only [State.advance] at ok
    obtain ⟨n,ran,rest⟩ := Option.bind_eq_some_iff.mp ok
    have h := advance_clock _ _ ran
    split at rest
    · obtain ⟨a,_,rest⟩ := Option.bind_eq_some_iff.mp rest
      cases normal with
      | none => cases rest; exact h
      | some word =>
        dsimp only at rest
        split at rest
        · obtain ⟨_,_,rest⟩ := Option.bind_eq_some_iff.mp rest
          cases rest; exact h
        · cases rest; exact h
    · split at rest <;> cases rest
      exact h
  | discarded =>
    simp only [State.advance] at ok
    obtain ⟨_,_,rest⟩ := Option.bind_eq_some_iff.mp ok
    split at rest <;> try contradiction
    obtain ⟨n,ran,end_⟩ := Option.bind_eq_some_iff.mp rest
    cases end_
    exact advance_clock _ _ ran

private theorem local_order {w : Worker p} (s out : State w) (es : List Event)
    (ok : es.foldlM (fun st e => State.advance st args e) s = some out) :
    es.Pairwise (fun a b => a.step < b.step) ∧ ∀ e ∈ es, s.cursor.clock ≤ e.step := by
  induction es generalizing s with
  | nil => simp
  | cons e es ih =>
    rw [List.foldlM_cons] at ok
    obtain ⟨next,one,tail⟩ := Option.bind_eq_some_iff.mp ok
    obtain ⟨atStep,after⟩ := step_time s next one
    obtain ⟨order,bounds⟩ := ih next tail
    refine ⟨List.pairwise_cons.mpr ⟨fun a mem => by have h := bounds a mem; omega,order⟩,?_⟩
    intro a member
    rcases List.mem_cons.mp member with rfl | mem
    · exact atStep
    · have hi := bounds a mem; omega

theorem worker_clock {p : Pool} {cap : Capture p} (parsed : decode p args rs = some cap) (w : Nat) :
    (localEvents w rs).Pairwise (fun a b => a.step < b.step) := by
  obtain ⟨_,ws,ran,_⟩ := decode_keeps parsed
  by_cases bound : w < (EndgameSearch.CacheOwner.poolPlan p.size).length
  · have get {ns : List Nat} {states : List ((who : Worker p) × State who)} (run : ns.mapM (workerRun p args rs) = some states) (here : w ∈ ns) :
        ∃ result, workerRun p args rs w = some result := by
      induction ns generalizing states with
      | nil => contradiction
      | cons n ns ih =>
        rw [List.mapM_cons] at run
        obtain ⟨obj,one,tail⟩ := Option.bind_eq_some_iff.mp run
        obtain ⟨more,done_,_⟩ := Option.bind_eq_some_iff.mp tail
        rcases List.mem_cons.mp here with rfl | later
        · exact ⟨obj,one⟩
        · exact ih done_ later
    obtain ⟨result,did⟩ := get ran (List.mem_range.mpr bound)
    unfold workerRun at did
    obtain ⟨who,_,rest⟩ := Option.bind_eq_some_iff.mp did
    obtain ⟨out,local_,_⟩ := Option.bind_eq_some_iff.mp rest
    exact (local_order _ out (localEvents w rs) local_).1
  · have empty : localEvents w rs = [] := by
      apply List.eq_nil_iff_forall_not_mem.mpr
      intro event mem
      obtain ⟨record,member,tag⟩ := List.mem_filterMap.mp mem
      cases record with
      | table => contradiction
      | localCall other ev =>
        change (if w = other then some ev else none) = some event at tag
        split at tag <;> try contradiction
        rename_i same
        subst other
        unfold decode at parsed
        dsimp only at parsed
        split at parsed <;> try contradiction
        rename_i valid
        simp only [Bool.not_eq_true',Bool.not_eq_false] at valid
        exact bound (of_decide_eq_true (List.all_eq_true.mp valid _ member))
    simp [empty]

private theorem at_event {s : Sample} (found : s.entry = some e) :
    ∃ site resource op, s.record = .localCall e.1.1 (.entry e.1.2 site .pvs resource op) := by
  unfold Sample.entry at found
  split at found <;> try contradiction
  obtain ⟨_,_,eq⟩ := Option.map_eq_some_iff.mp found
  cases eq
  exact ⟨_,_,_,by assumption⟩

private theorem unique (rs : List Sample)
    (time : ∀ w, (localEvents w (rs.map Sample.record)).Pairwise (fun a b => a.step < b.step)) :
    ((rs.filterMap Sample.entry).map Prod.fst).Nodup := by
  induction rs with
  | nil => simp
  | cons r rest ih =>
    have subseq (w : Nat) : (localEvents w (rest.map Sample.record)).Sublist
        (localEvents w ((r::rest).map Sample.record)) := by
      exact (List.sublist_cons_self _ _).filterMap _
    have tail := ih (fun w => (time w).sublist (subseq w))
    cases found : r.entry with
    | none => simpa [found] using tail
    | some e =>
      obtain ⟨site,resource,op,actual⟩ := at_event found
      simp only [List.filterMap_cons_some found,List.map_cons,List.nodup_cons]
      refine ⟨?_,tail⟩
      intro repeated
      obtain ⟨later,mem,eq⟩ := List.mem_map.mp repeated
      obtain ⟨s,member,valid⟩ := List.mem_filterMap.mp mem
      obtain ⟨site',resource',op',other⟩ := at_event valid
      rw [eq] at other
      have t := time e.1.1
      simp only [List.map_cons,localEvents,actual,List.filterMap_cons,ite_true,List.pairwise_cons] at t
      have laterTime := t.1 (.entry e.1.2 site' .pvs resource' op') (List.mem_filterMap.mpr
        ⟨_,List.mem_map.mpr ⟨s,member,other⟩,by simp⟩)
      exact Nat.lt_irrefl _ laterTime

/-- A successful decode supplies the unique worker/opening keys. A register
    sample can never be shadowed by the entry of a nested or concurrent call. -/
theorem at_entry {p : Pool} {cap : Capture p} (rs : List Sample) (parsed : capture p rs = some cap)
    (e : Entry) (mem : e ∈ rs.filterMap Sample.entry) : cap.inputs e.1.1 e.1.2 = some e.2 := by
  rw [(source_of_decode parsed).1]
  change (rs.filterMap Sample.entry).lookup e.1 = some e.2
  have keys := unique rs (worker_clock parsed)
  generalize rs.filterMap Sample.entry = es at keys mem ⊢
  induction es with
  | nil => contradiction
  | cons first more ih =>
    rcases List.nodup_cons.mp keys with ⟨fresh,tail⟩
    rcases List.mem_cons.mp mem with rfl | later
    · exact List.lookup_cons_self
    · have ne : e.1 ≠ first.1 := fun eq => fresh (List.mem_map.mpr ⟨e,later,eq⟩)
      simpa [List.lookup,beq_eq_false_iff_ne.mpr ne] using ih tail later

end NeuralReversi.PvsParallel.NativeFrames.EntryStream

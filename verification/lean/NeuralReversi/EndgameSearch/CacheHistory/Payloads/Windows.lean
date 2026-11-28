module
public import NeuralReversi.EndgameSearch.CacheHistory.Payloads.Workers
public import NeuralReversi.EndgameSearch.CacheHistory.NativeTrace.Order

@[expose] public section

namespace NeuralReversi.EndgameSearch.CacheHistory.Payloads

open NativeTrace (Worker Cursor Event Capture decode_keeps)

def Interval.earlier (a b : Interval) := a.worker = b.worker → a.last < b.first

private def finishedWindow (v : Visit p) : Interval := ⟨v.worker,v.first,v.last⟩

private def openWindow (s : (w : Worker p) × Cursor w) : Option Interval := do
  let f ← s.2.stack.find? (fun a => a.node == .boundary)
  return ⟨s.1.ordinal,f.first,s.2.clock⟩

private def windows (s : (w : Worker p) × Cursor w) :=
  s.2.completed.map finishedWindow ++ (openWindow s).toList

private theorem worker_windows (w : Worker p) (stream : List (Event S)) (s : Cursor w) (ok : NativeTrace.run w stream = some s) :
    (windows ⟨w,s⟩).Pairwise Interval.earlier ∧ ∀ a ∈ windows ⟨w,s⟩, a.worker = w.ordinal := by
  obtain ⟨ordered,worker⟩ := NativeTrace.run_ordered w stream ok
  have prior : (s.completed.map finishedWindow).Pairwise Interval.earlier := List.pairwise_map.mpr ordered
  unfold windows openWindow
  cases found : s.stack.find? (fun a => a.node == .boundary) with
  | none => simpa [finishedWindow] using And.intro prior worker
  | some f =>
    have atStack := List.mem_of_find?_eq_some found
    have boundary : f.node = .boundary := by simpa using List.find?_some found
    let current : Interval := ⟨w.ordinal,f.first,s.clock⟩
    let allWindows := s.completed.map finishedWindow ++ [current]
    change allWindows.Pairwise Interval.earlier ∧ ∀ a ∈ allWindows, a.worker = w.ordinal
    simp only [allWindows,List.pairwise_append,prior,List.pairwise_singleton,
      true_and,List.mem_map,List.mem_singleton,List.mem_append]
    exact ⟨fun a ⟨v,hv,h⟩ b hb _ => by
      subst a b
      exact NativeTrace.pending_after w stream ok f atStack boundary v hv,
      fun a ha => by
        rcases ha with ⟨v,hv,rfl⟩ | rfl
        · exact worker _ hv
        · rfl⟩

private theorem count_group (ws : List ((w : Worker p) × Cursor w)) (pred : Interval → Bool) :
    (((ws.flatMap (fun w => w.2.completed)).map finishedWindow ++ ws.filterMap openWindow).filter pred).length =
      ((ws.flatMap windows).filter pred).length := by
  induction ws with
  | nil => rfl
  | cons w rest ih =>
    cases cur : openWindow w <;>
      simp only [List.flatMap_cons,List.map_append,List.filterMap_cons,cur,List.filter_append,List.length_append,
        List.filter_cons,windows,Option.toList_none,Option.toList_some,List.append_nil] at *
    all_goals first | omega | (split <;> simp_all <;> omega)

theorem Interval.atMostOne (xs : List Interval) (ordered : xs.Pairwise Interval.earlier)
    (s : Site) (time : s.request.opening ≤ s.request.returned) : (xs.filter (·.contains s)).length ≤ 1 := by
  induction xs with
  | nil => exact Nat.zero_le _
  | cons a rest ih =>
    obtain ⟨first,more⟩ := List.pairwise_cons.mp ordered
    by_cases hit : a.contains s = true
    · have earlier : a.worker = s.worker ∧ a.first ≤ s.request.opening ∧ s.request.returned ≤ a.last := by
        simpa [contains] using hit
      have later : rest.filter (·.contains s) = [] := by
        apply List.filter_eq_nil_iff.mpr
        intro b hb overlap
        have after : b.worker = s.worker ∧ b.first ≤ s.request.opening ∧ s.request.returned ≤ b.last := by
          simpa [contains] using overlap
        have := first b hb (earlier.1.trans after.1.symm)
        omega
      simp [hit,later]
    · simpa [hit] using ih more

/-- In every successfully parsed stream, each well-timed primitive can be in
    at most one completed or still-open visit across the entire worker pool. -/
theorem intervals_unique (p : Pool) (stream : List (Event S)) (c : Capture p S) (parsed : NativeTrace.decode p stream = some c)
    (site : Site) (time : site.request.opening ≤ site.request.returned) :
    ((intervals c).filter (·.contains site)).length ≤ 1 := by
  obtain ⟨_,completed,ran⟩ := decode_keeps p stream parsed
  have ordered := (NativeTrace.group_workers p stream windows Interval.worker Interval.earlier
    (fun _ _ ne eq => (ne eq).elim) (fun w s => worker_windows w stream s) _ List.nodup_range ran).1
  unfold intervals
  rw [completed]
  exact (count_group c.workers _).symm ▸ Interval.atMostOne _ ordered site time

end NeuralReversi.EndgameSearch.CacheHistory.Payloads

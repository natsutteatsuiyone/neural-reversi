module
public import NeuralReversi.EndgameSearch.CacheHistory.NativeTrace

@[expose] public section

namespace NeuralReversi.EndgameSearch.CacheHistory.NativeTrace

variable {p : Pool} {w : Worker p}

/-- The stack can hold at most one cache boundary. Its entry follows every completed
    visit on this worker even while other calls are active above it. -/
structure Ordered (s : Cursor w) : Prop where
  visits : s.completed.Pairwise Earlier
  past : ∀ v ∈ s.finished, v.worker = w.ordinal ∧ v.last < s.clock
  boundary : ∀ top ∈ s.stack, top.node = .boundary → ∀ v ∈ s.finished, v.last < top.first
  unique : s.stack.Pairwise fun a b => a.node = .boundary → b.node ≠ .boundary
  tagged : ∀ top ∈ s.stack, top.operands.isSome = (top.node == .boundary)

theorem Ordered.initial : Ordered (.initial w) := by
  constructor <;> simp [Cursor.initial,Cursor.completed]

private theorem enter_clock (s : Cursor w) (ok : s.enter t site node resource args = some out) :
    out.clock = t+1 := by
  unfold Cursor.enter at ok
  dsimp only at ok
  repeat' first | split at ok | contradiction
  all_goals cases ok <;> rfl

private theorem leave_clock (s : Cursor w) (ok : s.leave t opening node score = some out) :
    out.clock = t+1 := by
  unfold Cursor.leave at ok
  repeat' first | split at ok | contradiction
  all_goals cases ok <;> rfl

theorem advance_clock (s : Cursor w) (ev : Raw) (ok : s.advance ev = some out) :
    s.clock ≤ ev.step ∧ out.clock = ev.step+1 := by
  unfold Cursor.advance at ok
  split at ok <;> try contradiction
  refine ⟨by assumption,?_⟩
  cases ev with
  | enter => exact enter_clock s ok
  | leave => exact leave_clock s ok

private theorem entered (s : Cursor w) (prior : Ordered s) (time : s.clock ≤ t)
    (ok : s.enter t site node resource args = some out) : Ordered out := by
  obtain ⟨top,stack,finished,first,kind,_,payload,tag,reserved⟩ := checked_entry s out ok
  have tick := enter_clock s ok
  refine ⟨by simpa only [Cursor.completed,finished] using prior.visits,?_,?_,?_,?_⟩
  · intro v mem
    obtain ⟨worker,earlier⟩ := prior.past v (finished ▸ mem)
    exact ⟨worker,by omega⟩
  · intro a mem boundary v back
    rw [stack] at mem
    rcases List.mem_cons.mp mem with rfl | below
    · have earlier := (prior.past v (finished ▸ back)).2
      omega
    · exact prior.boundary _ below boundary _ (finished ▸ back)
  · rw [stack,List.pairwise_cons]
    refine ⟨?_,prior.unique⟩
    intro a mem topBoundary another
    have empty := reserved (kind ▸ topBoundary)
    have busy : s.busy = true := List.any_eq_true.mpr ⟨a,mem,by simp [another]⟩
    simp [empty] at busy
  · intro a mem
    rw [stack] at mem
    rcases List.mem_cons.mp mem with rfl | inside
    · simpa only [payload,kind] using tag
    · exact prior.tagged _ inside

private theorem left (s : Cursor w) (prior : Ordered s) (time : s.clock ≤ t)
    (ok : s.leave t opening node score = some out) : Ordered out := by
  cases stack : s.stack with
  | nil => simp [Cursor.leave,stack] at ok
  | cons top rest =>
    have tick := leave_clock s ok
    obtain ⟨noOther,unique⟩ := List.pairwise_cons.mp (stack ▸ prior.unique)
    have below : ∀ a ∈ rest, a ∈ s.stack := fun a mem => stack ▸ List.mem_cons_of_mem top mem
    have atTop : top ∈ s.stack := stack ▸ List.mem_cons_self
    cases payload : top.operands with
    | none =>
      obtain ⟨finished,popped⟩ := propagation s out top stack payload ok
      refine ⟨by simpa only [Cursor.completed,finished] using prior.visits,?_,?_,
        popped ▸ unique,fun a h => prior.tagged a (below _ (popped ▸ h))⟩
      · intro v mem
        obtain ⟨worker,earlier⟩ := prior.past v (finished ▸ mem)
        exact ⟨worker,by omega⟩
      · intro a mem eq v hit
        exact prior.boundary a (below _ (popped ▸ mem)) eq v (finished ▸ hit)
    | some op =>
      obtain ⟨v,finished,popped,worker,first,last,_,_,_,_⟩ := matched_boundary s out top stack payload ok
      have tag := prior.tagged top atTop
      have boundary : top.node = .boundary := by simpa [payload] using tag.symm
      have start : top.first = opening := by
        by_cases h : top.first = opening
        · exact h
        · simp [Cursor.leave,stack,h] at ok
      refine ⟨?_,?_,?_,popped ▸ unique,fun a h => prior.tagged a (below _ (popped ▸ h))⟩
      · simp only [Cursor.completed,finished,List.reverse_cons,List.pairwise_append,List.pairwise_singleton,
          true_and]
        refine ⟨prior.visits,?_⟩
        intro previous mem b one _
        obtain rfl := List.mem_singleton.mp one
        have earlier := prior.boundary _ atTop boundary previous (List.mem_reverse.mp mem)
        omega
      · intro a mem
        rw [finished] at mem
        rcases List.mem_cons.mp mem with rfl | old
        · exact ⟨worker,by omega⟩
        · obtain ⟨same,earlier⟩ := prior.past a old
          exact ⟨same,by omega⟩
      · intro a mem another
        exact (noOther a (popped ▸ mem) boundary another).elim

theorem Ordered.advance (s : Cursor w) (prior : Ordered s) (ev : Raw)
    (ok : s.advance ev = some out) : Ordered out := by
  unfold Cursor.advance at ok
  split at ok <;> try contradiction
  rename_i time
  cases ev with
  | enter => exact entered s prior time ok
  | leave => exact left s prior time ok

private theorem run_invariant (w : Worker p) (stream : List (Event S)) {out : Cursor w} (ok : run w stream = some out) :
    Ordered out := by
  have loop (events : List Raw) (s : Cursor w) (prior : Ordered s)
      (ran : events.foldlM Cursor.advance s = some out) : Ordered out := by
    induction events generalizing s with
    | nil => exact (show s = out from Option.some.inj ran) ▸ prior
    | cons e rest ih =>
      rw [List.foldlM_cons] at ran
      obtain ⟨next,act,more⟩ := Option.bind_eq_some_iff.mp ran
      exact ih next (Ordered.advance s prior e act) more
  exact loop _ _ Ordered.initial ok

theorem pending_after (w : Worker p) (stream : List (Event S)) {out : Cursor w} (ok : run w stream = some out)
    (top : Activation w) (mem : top ∈ out.stack) (boundary : top.node = .boundary) (v : Visit p) (done_ : v ∈ out.completed) :
    v.last < top.first :=
  (run_invariant w stream ok).boundary top mem boundary v (List.mem_reverse.mp done_)

/-- Successful worker parsing constructs a chronological list of disjoint cache visits
    even when the observed execution prefix ends inside a call. -/
theorem run_ordered (w : Worker p) (stream : List (Event S)) {out : Cursor w} (ok : run w stream = some out) :
    out.completed.Pairwise Earlier ∧ ∀ v ∈ out.completed, v.worker = w.ordinal := by
  have done_ := run_invariant w stream ok
  exact ⟨done_.visits, fun v mem => (done_.past v (List.mem_reverse.mp mem)).1⟩

/-- Combine each parsed worker's ordered records. The parser starts one
    worker per ordinal, so records from distinct workers need only the foreign case. -/
theorem group_workers (p : Pool) (stream : List (Event S))
    (records : ((w : Worker p) × Cursor w) → List α) (owner : α → Nat) (rel : α → α → Prop)
    (foreign : ∀ a b, owner a ≠ owner b → rel a b)
    (localOrder : ∀ (w : Worker p) s, run w stream = some s →
      (records ⟨w,s⟩).Pairwise rel ∧ ∀ a ∈ records ⟨w,s⟩, owner a = w.ordinal)
    (ids : List Nat) {ws : List ((w : Worker p) × Cursor w)} (distinct : ids.Nodup)
    (ok : ids.mapM (readWorker p stream) = some ws) :
    (ws.flatMap records).Pairwise rel ∧ ∀ a ∈ ws.flatMap records, owner a ∈ ids := by
  induction ids generalizing ws with
  | nil => cases ok; simp
  | cons n ids ih =>
    obtain ⟨absent,unique⟩ := List.nodup_cons.mp distinct
    rw [List.mapM_cons] at ok
    obtain ⟨item,read,next⟩ := Option.bind_eq_some_iff.mp ok
    obtain ⟨tail,more,eq⟩ := Option.bind_eq_some_iff.mp next
    cases eq
    obtain ⟨w,boot,read⟩ := Option.bind_eq_some_iff.mp read
    obtain ⟨s,ran,eq⟩ := Option.bind_eq_some_iff.mp read
    cases eq
    have id : w.ordinal = n := by unfold workerAt at boot; split at boot <;> cases boot; rfl
    obtain ⟨head,here⟩ := localOrder w s ran
    obtain ⟨rest,later⟩ := ih unique more
    simp only [List.flatMap_cons,List.pairwise_append,List.mem_append]
    refine ⟨⟨head,rest,?_⟩,?_⟩
    · intro a ha b hb
      exact foreign a b (fun same => absent (id ▸ here a ha ▸ same.symm ▸ later b hb))
    · intro a mem
      rcases mem with first | last
      · exact List.mem_cons.mpr (.inl ((here a first).trans id))
      · exact List.mem_cons_of_mem _ (later a last)

/-- Combining distinct workers keeps the order of visits to each private cache. -/
theorem workers_ordered (p : Pool) (stream : List (Event S)) (ids : List Nat)
    {ws : List ((w : Worker p) × Cursor w)} (distinct : ids.Nodup)
    (ok : ids.mapM (readWorker p stream) = some ws) :
    (ws.flatMap (fun w => w.2.completed)).Pairwise Earlier ∧
      ∀ v ∈ ws.flatMap (fun w => w.2.completed), v.worker ∈ ids :=
  group_workers p stream (fun w => w.2.completed) Visit.worker Earlier (fun _ _ ne eq => (ne eq).elim)
    (fun w _ => run_ordered w stream) ids distinct ok

end NeuralReversi.EndgameSearch.CacheHistory.NativeTrace

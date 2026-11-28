module
public import NeuralReversi.PvsParallel.NativeFrames
public import NeuralReversi.EndgameSearch.CacheHistory.Payloads.Coverage

@[expose] public section

namespace NeuralReversi.PvsParallel.NativeFrameCache

open EndgameSearch.CacheHistory (Pool)
open EndgameSearch.CacheHistory.NativeTrace (Worker Cursor Raw)
open NativeFrames (State Args Capture Record Event)

abbrev EdgeEvent := EndgameSearch.CacheHistory.NativeTrace.Event ConcurrentReplay.Event

def project (stream : List Record) : List EdgeEvent := stream.filterMap fun
  | .localCall worker event => event.edge?.map (.worker worker)
  | .table e => some (.shared e)

def localCalls (n : Nat) (stream : List Record) : List Event := stream.filterMap fun
  | .localCall worker e => if n = worker then some e else none
  | .table _ => none

theorem same_table (stream : List Record) :
    EndgameSearch.CacheHistory.NativeTrace.sharedEvents (project stream) = NativeFrames.tableEvents stream := by
  induction stream with
  | nil => rfl
  | cons r rs ih =>
    cases r with
    | table entry => exact congrArg (entry :: ·) ih
    | localCall n e =>
      cases e <;> exact ih

theorem projected_local (stream : List Record) (n : Nat) :
    EndgameSearch.CacheHistory.NativeTrace.workerEvents n (project stream) = (localCalls n stream).filterMap Event.edge? := by
  simp only [EndgameSearch.CacheHistory.NativeTrace.workerEvents,project,localCalls,List.filterMap_filterMap]
  apply congrArg (fun fn => stream.filterMap fn)
  funext r
  cases r with
  | table => rfl
  | localCall m e =>
    by_cases h : n = m
    · subst m; cases e <;> simp [Event.edge?]
    · cases e <;> simp [Event.edge?,h,Ne.symm h]

variable {p : Pool} {w : Worker p}

/-- `edge` ignores scored events; `full` includes them. All fields except their clocks coincide. -/
structure Agrees (full edge : Cursor w) : Prop where
  clock : edge.clock ≤ full.clock
  stack : edge.stack = full.stack
  finished : edge.finished = full.finished
  sites : edge.boundarySites = full.boundarySites

theorem Agrees.refl (c : Cursor w) : Agrees c c := ⟨Nat.le_refl _,rfl,rfl,rfl⟩

private theorem accept_edge (same : Agrees full edge) (raw : Raw) (ok : full.advance raw = some after) :
    edge.advance raw = some after := by
  have clock : full.clock ≤ raw.step := by
    by_cases le : full.clock ≤ raw.step
    · exact le
    · simp [Cursor.advance,le] at ok
  have small := Nat.le_trans same.clock clock
  have shape : edge = {full with clock := edge.clock} := by
    cases edge
    simp only [Cursor.mk.injEq,true_and]
    exact ⟨same.stack,same.finished,same.sites⟩
  cases raw with
  | enter t i n v args =>
    simp only [Cursor.advance,clock,small,ite_true] at ok ⊢
    rw [shape]
    exact ok
  | leave t i n stamp =>
    simp only [Cursor.advance,clock,small,ite_true] at ok ⊢
    rw [shape]
    exact ok

private theorem entry_cursor (s out : State w)
    (ok : s.advance inputs (.entry t site node r args) = some out) :
    s.cursor.advance (.enter t site node r args) = some out.cursor := by
  simp only [State.advance] at ok
  split at ok <;> try contradiction
  obtain ⟨next,read,tail⟩ := Option.bind_eq_some_iff.mp ok
  split at tail
  · obtain ⟨_,_,tail⟩ := Option.bind_eq_some_iff.mp tail
    split at tail <;> try contradiction
    cases tail
    exact read
  · cases tail
    exact read

private theorem leave_cursor (s out : State w)
    (ok : s.advance inputs (.leave t i node stamp normal) = some out) :
    s.cursor.advance (.leave t i node stamp) = some out.cursor := by
  simp only [State.advance] at ok
  obtain ⟨next,read,tail⟩ := Option.bind_eq_some_iff.mp ok
  split at tail
  · obtain ⟨_,_,tail⟩ := Option.bind_eq_some_iff.mp tail
    split at tail
    · split at tail
      · obtain ⟨_,_,ret⟩ := Option.bind_eq_some_iff.mp tail
        cases ret
        exact read
      · cases tail
        exact read
    · cases tail
      exact read
  · split at tail <;> try contradiction
    cases tail
    exact read

private theorem score_cursor (s out : State w) (ok : s.advance inputs (.scored t score) = some out) :
    out.cursor = {s.cursor with clock := t+1} ∧ s.cursor.clock ≤ t := by
  simp only [State.advance] at ok
  split at ok
  · split at ok <;> try contradiction
    rename_i time
    obtain ⟨_,_,tail⟩ := Option.bind_eq_some_iff.mp ok
    have clock : s.cursor.clock ≤ t := by
      by_cases h : t < s.cursor.clock
      · simp [h] at time
      · omega
    split at tail
    · cases tail; exact ⟨rfl,clock⟩
    · split at tail <;> try contradiction
      obtain ⟨_,_,tail⟩ := Option.bind_eq_some_iff.mp tail
      cases tail
      exact ⟨rfl,clock⟩
  · contradiction

private theorem discarded_cursor (s out : State w) (ok : s.advance inputs (.discarded t i score) = some out) :
    s.cursor.advance (.leave t i .pvs none) = some out.cursor := by
  unfold State.advance at ok
  obtain ⟨_,_,next⟩ := Option.bind_eq_some_iff.mp ok
  split at next <;> try contradiction
  obtain ⟨cursor,moved,eq⟩ := Option.bind_eq_some_iff.mp next
  cases eq
  exact moved

private theorem step_clock (s out : State w) (ev : Event) (ok : s.advance inputs ev = some out) :
    s.cursor.clock ≤ ev.step ∧ out.cursor.clock = ev.step+1 := by
  cases ev with
  | entry => exact EndgameSearch.CacheHistory.NativeTrace.advance_clock _ _ (entry_cursor s out ok)
  | leave => exact EndgameSearch.CacheHistory.NativeTrace.advance_clock _ _ (leave_cursor s out ok)
  | discarded => exact EndgameSearch.CacheHistory.NativeTrace.advance_clock _ _ (discarded_cursor s out ok)
  | scored =>
    obtain ⟨same,earlier⟩ := score_cursor s out ok
    exact ⟨earlier,congrArg Cursor.clock same⟩

theorem worker_chronology (p : Pool) {cap : Capture p} (inputs : Nat → Nat → Option Args) (stream : List Record)
    (n : Nat) (inside : n < (EndgameSearch.CacheOwner.poolPlan p.size).length)
    (parsed : NativeFrames.decode p inputs stream = some cap) :
    (localCalls n stream).Pairwise (fun a b => a.step < b.step) := by
  obtain ⟨_,ws,workers,_⟩ := NativeFrames.decode_keeps parsed
  have existsRun (ids : List Nat) (states) (ran : ids.mapM (NativeFrames.workerRun p inputs stream) = some states)
      (member : n ∈ ids) : ∃ out, NativeFrames.workerRun p inputs stream n = some out := by
    induction ids generalizing states with
    | nil => cases member
    | cons x rest ih =>
      rw [List.mapM_cons] at ran
      obtain ⟨state,one,tail⟩ := Option.bind_eq_some_iff.mp ran
      obtain ⟨ss,more,_⟩ := Option.bind_eq_some_iff.mp tail
      rcases List.mem_cons.mp member with rfl | later
      · exact ⟨_,one⟩
      · exact ih _ more later
  obtain ⟨item,ran⟩ := existsRun _ ws workers (List.mem_range.mpr inside)
  obtain ⟨w,_,ran⟩ := Option.bind_eq_some_iff.mp ran
  obtain ⟨s,done_,_⟩ := Option.bind_eq_some_iff.mp ran
  have ordered (initial out : State w) (evs : List Event)
      (ok : evs.foldlM (fun state ev => state.advance (inputs n) ev) initial = some out) :
      evs.Pairwise (fun a b => a.step < b.step) ∧ ∀ ev ∈ evs, initial.cursor.clock ≤ ev.step := by
    induction evs generalizing initial with
    | nil => simp
    | cons e es ih =>
      rw [List.foldlM_cons] at ok
      obtain ⟨next,took,more⟩ := Option.bind_eq_some_iff.mp ok
      obtain ⟨ordered,rest⟩ := ih _ more
      obtain ⟨earlier,tick⟩ := step_clock _ _ _ took
      refine ⟨List.pairwise_cons.mpr ⟨fun ev mem => by have := rest ev mem; omega,ordered⟩,?_⟩
      intro ev mem
      rcases List.mem_cons.mp mem with rfl | later
      · exact earlier
      · have := rest ev later; omega
  exact (ordered _ s _ done_).1

theorem run_edges (w : Worker p) (inputs : Nat → Option Args) (events : List Event) {output : State w}
    (ok : NativeFrames.run w inputs events = some output) :
    ∃ edge, (events.filterMap Event.edge?).foldlM Cursor.advance (.initial w) = some edge ∧ Agrees output.cursor edge := by
  have sim (s : State w) (c : Cursor w) (prior : Agrees s.cursor c)
      (execution : events.foldlM (fun s e => s.advance inputs e) s = some output) :
      ∃ edge, (events.filterMap Event.edge?).foldlM Cursor.advance c = some edge ∧ Agrees output.cursor edge := by
    clear ok
    induction events generalizing s c with
    | nil =>
      simp at execution
      cases execution
      exact ⟨c,rfl,prior⟩
    | cons e es ih =>
      rw [List.foldlM_cons] at execution
      obtain ⟨next,step,remaining⟩ := Option.bind_eq_some_iff.mp execution
      cases e with
      | entry t site node r args =>
        have applied := accept_edge prior _ (entry_cursor s next step)
        obtain ⟨edge,rest,final⟩ := ih next next.cursor (.refl _) remaining
        exact ⟨edge,by simpa [Event.edge?,List.foldlM_cons,applied] using rest,final⟩
      | leave t i node stamp normal =>
        have applied := accept_edge prior _ (leave_cursor s next step)
        obtain ⟨edge,rest,final⟩ := ih next next.cursor (.refl _) remaining
        exact ⟨edge,by simpa [Event.edge?,List.foldlM_cons,applied] using rest,final⟩
      | discarded t i score =>
        have applied := accept_edge prior _ (discarded_cursor s next step)
        obtain ⟨edge,rest,final⟩ := ih next next.cursor (.refl _) remaining
        exact ⟨edge,by simpa [Event.edge?,List.foldlM_cons,applied] using rest,final⟩
      | scored t score =>
        obtain ⟨updated,tick⟩ := score_cursor s next step
        have before := prior.clock
        have agree : Agrees next.cursor c := by
          rw [updated]
          exact ⟨Nat.le_trans before (Nat.le_succ_of_le tick),prior.stack,prior.finished,prior.sites⟩
        exact ih next c agree remaining
  exact sim _ _ (.refl _) ok

private theorem worker_id (found : EndgameSearch.CacheHistory.NativeTrace.workerAt p n = some w) :
    w.ordinal = n := by
  unfold EndgameSearch.CacheHistory.NativeTrace.workerAt at found
  split at found <;> cases found
  rfl

theorem worker_edges (p : Pool) (inputs : Nat → Nat → Option Args) (stream : List Record)
    {output : (w : Worker p) × State w}
    (ok : NativeFrames.workerRun p inputs stream n = some output) :
    ∃ (w : Worker p) (full : State w) (edge : Cursor w), output = ⟨w,full⟩ ∧
      EndgameSearch.CacheHistory.NativeTrace.readWorker p (project stream) n = some ⟨w,edge⟩ ∧ Agrees full.cursor edge := by
  unfold NativeFrames.workerRun at ok
  obtain ⟨w,boot,next⟩ := Option.bind_eq_some_iff.mp ok
  change (NativeFrames.run w (inputs n) (localCalls n stream)).bind _ = _ at next
  obtain ⟨state,execution,out⟩ := Option.bind_eq_some_iff.mp next
  cases out
  obtain ⟨edge,projected,agree⟩ := run_edges w (inputs n) _ execution
  refine ⟨w,state,edge,rfl,?_,agree⟩
  have raw : EndgameSearch.CacheHistory.NativeTrace.run w (project stream) = some edge := by
    simpa only [EndgameSearch.CacheHistory.NativeTrace.run,worker_id boot,projected_local] using projected
  simp [EndgameSearch.CacheHistory.NativeTrace.readWorker,boot,raw]

private theorem all_workers (p : Pool) (inputs : Nat → Nat → Option Args) (stream : List Record) (ids : List Nat)
    {full : List ((w : Worker p) × State w)}
    (finished : ids.mapM (NativeFrames.workerRun p inputs stream) = some full) :
    ∃ edges, ids.mapM (EndgameSearch.CacheHistory.NativeTrace.readWorker p (project stream)) = some edges ∧
      edges.flatMap (fun w => w.2.completed) = full.flatMap (fun w => w.2.cursor.completed) := by
  induction ids generalizing full with
  | nil => simp at finished; cases finished; exact ⟨[],rfl,rfl⟩
  | cons n ns ih =>
    rw [List.mapM_cons] at finished
    obtain ⟨item,one,more⟩ := Option.bind_eq_some_iff.mp finished
    obtain ⟨tail,rest,eq⟩ := Option.bind_eq_some_iff.mp more
    change some (item :: tail) = some full at eq
    cases eq
    obtain ⟨w,state,edge,rfl,first,same⟩ := worker_edges p inputs stream one
    obtain ⟨edges,remaining,visits⟩ := ih rest
    refine ⟨⟨w,edge⟩ :: edges,by simp [List.mapM_cons,first,remaining],?_⟩
    simp only [List.flatMap_cons]
    rw [visits]
    change edge.finished.reverse ++ _ = state.cursor.finished.reverse ++ _
    rw [same.finished]

/-- Per-worker parsing and the pool bounds suffice to construct the combined PVS
    frame capture. The global cache-visit ordering check follows from the stacks. -/
theorem capture_of_workers (p : Pool) (inputs : Nat → Nat → Option Args) (stream : List Record)
    {states : List ((w : Worker p) × State w)}
    (inRange : stream.all (fun
      | .localCall n _ => decide (n < (EndgameSearch.CacheOwner.poolPlan p.size).length)
      | .table _ => true) = true)
    (parsed : (List.range (EndgameSearch.CacheOwner.poolPlan p.size).length).mapM
      (NativeFrames.workerRun p inputs stream) = some states) :
    ∃ c, NativeFrames.decode p inputs stream = some c := by
  obtain ⟨edges,read,same⟩ := all_workers p inputs stream _ parsed
  have ordered := (EndgameSearch.CacheHistory.NativeTrace.workers_ordered p (project stream) _
    List.nodup_range read).1
  rw [same] at ordered
  simp [NativeFrames.decode,parsed,ordered]
  intro record mem
  have valid := List.all_eq_true.mp inRange record mem
  cases record <;> exact valid

private def inPool (size : Nat) : Record → Bool
  | .localCall worker _ => decide (worker < size)
  | .table _ => true

private def edgeInPool (size : Nat) : EdgeEvent → Bool
  | .worker worker _ => decide (worker < size)
  | .shared _ => true

private theorem worker_range {stream : List Record} (inside : stream.all (inPool size) = true) :
    (project stream).all (edgeInPool size) = true := by
  apply List.all_eq_true.mpr
  intro edge mem
  obtain ⟨record,present,got⟩ := List.mem_filterMap.mp mem
  have number := List.all_eq_true.mp inside record present
  cases record with
  | table => cases got; rfl
  | localCall w e => cases e <;> cases got <;> exact number

private theorem decode_range {p : Pool} {c : Capture p} (ok : NativeFrames.decode p inputs stream = some c) :
    stream.all (inPool (EndgameSearch.CacheOwner.poolPlan p.size).length) = true := by
  unfold NativeFrames.decode at ok
  dsimp only at ok
  split at ok
  · contradiction
  · rename_i range
    apply List.all_eq_true.mpr
    intro r mem
    simp at range
    have each := range r mem
    cases r <;> exact each

theorem projected_capture {p : Pool} {c : Capture p} (ok : NativeFrames.decode p inputs stream = some c) :
    ∃ localCache, EndgameSearch.CacheHistory.NativeTrace.decode p (project stream) = some localCache ∧
      localCache.timeline = c.timeline ∧ localCache.shared = c.table := by
  obtain ⟨shared,states,done,_,completed⟩ := NativeFrames.decode_keeps ok
  obtain ⟨edges,built,agree⟩ := all_workers p inputs stream _ done
  have visits := agree.trans completed.symm
  have valid := worker_range (decode_range ok)
  have sorted : (edges.flatMap (fun w => w.2.completed)).Pairwise EndgameSearch.CacheHistory.Earlier := visits ▸ c.timeline.ordered
  let result : EndgameSearch.CacheHistory.NativeTrace.Capture p ConcurrentReplay.Event :=
    ⟨edges,⟨_,sorted⟩,EndgameSearch.CacheHistory.NativeTrace.sharedEvents (project stream)⟩
  have eqTimeline {x y : EndgameSearch.CacheHistory.Timeline p} (v : x.visits = y.visits) : x = y := by
    cases x; cases y; congr
  refine ⟨result,?_,eqTimeline visits,(same_table stream).trans shared.symm⟩
  unfold EndgameSearch.CacheHistory.NativeTrace.decode
  dsimp only
  split
  · rename_i bad
    simp at bad
    obtain ⟨e,mem,failure⟩ := bad
    have each := List.all_eq_true.mp valid e mem
    cases e <;> simp_all [edgeInPool] <;> omega
  · simp [built,sorted,result]

structure Checked (p : Pool) where
  frames : Capture p
  cache : EndgameSearch.CacheHistory.Payloads.Checked p ConcurrentReplay.Event

def decode (p : Pool) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (inputs : Nat → Nat → Option Args) (values : EndgameSearch.CacheHistory.Payloads.Values) (stream : List Record) :
    Option (Checked p) := do
  let frames ← NativeFrames.decode p inputs stream
  let cache ← EndgameSearch.CacheHistory.Payloads.decode p mc fc values (project stream)
  return ⟨frames,cache⟩

theorem same_run {out : Checked p} (ok : decode p mc fc inputs values stream = some out) :
    NativeFrames.decode p inputs stream = some out.frames ∧
    EndgameSearch.CacheHistory.Payloads.decode p mc fc values (project stream) = some out.cache ∧
    out.cache.native.timeline = out.frames.timeline ∧ out.cache.native.shared = out.frames.table := by
  unfold decode at ok
  obtain ⟨f,frames,rest⟩ := Option.bind_eq_some_iff.mp ok
  obtain ⟨c,cache,equal⟩ := Option.bind_eq_some_iff.mp rest
  cases equal
  have raw := (EndgameSearch.CacheHistory.Payloads.observed cache).1
  obtain ⟨projected,equality,visits,table⟩ := projected_capture frames
  rw [raw] at equality
  cases equality
  exact ⟨frames,cache,visits,table⟩

/-- PVS rows use the timeline and table records whose cache operands were checked. -/
def Checked.physical (out : Checked p) : Capture p :=
  {out.frames with table := out.cache.native.shared, timeline := out.cache.native.timeline}

theorem decode_physical {out : Checked p} (ok : decode p mc fc inputs values stream = some out) :
    NativeFrames.decode p inputs stream = some out.physical := by
  obtain ⟨frames,_,timeline,shared⟩ := same_run ok
  simpa [Checked.physical,timeline,shared] using frames

/-- Combine a native search-frame execution with each completed visit and
    outstanding worker's physical array replay. Cache-site coverage and agreement
    with the frame timeline follow from the same call stream. -/
theorem capture_of_replays {p : Pool} {c : Capture p} (parsed : NativeFrames.decode p inputs stream = some c)
    (values : EndgameSearch.CacheHistory.Payloads.Values)
    (visits : ∀ v ∈ c.timeline.visits, ∃ a, EndgameSearch.CacheHistory.Payloads.completeVisit
      p c.timeline mc fc values (project stream) v = some a)
    (workers : ∀ (w : Worker p) s, EndgameSearch.CacheHistory.NativeTrace.run w (project stream) = some s →
      ∃ a, EndgameSearch.CacheHistory.Payloads.pendingWorker p c.timeline mc fc values (project stream) ⟨w,s⟩ = some a) :
    ∃ out, decode p mc fc inputs values stream = some out ∧ out.physical = c := by
  obtain ⟨native,read,timeline,_⟩ := projected_capture parsed
  have states := EndgameSearch.CacheHistory.Payloads.capture_workers p (project stream) native read
  obtain ⟨cache,encoded,_⟩ := EndgameSearch.CacheHistory.Payloads.decode_of_visits
    p (project stream) native values read (timeline ▸ visits) (fun item member => timeline ▸ workers item.1 item.2 (states item member))
  have assembled : decode p mc fc inputs values stream = some ⟨c,cache⟩ := by simp [decode,parsed,encoded]
  have same := (decode_physical assembled).symm.trans parsed
  exact ⟨_,assembled,Option.some.inj same⟩

theorem closed_arrays {out : Checked p} (ok : decode p mc fc inputs values stream = some out)
    (v : EndgameSearch.CacheHistory.Visit p) (mem : v ∈ out.physical.timeline.visits) :
    let prior := out.physical.timeline.readAt mc fc v.worker v.first
    ∃ effect ∈ out.cache.finished,
      EndgameSearch.CacheHistory.Payloads.collect values (project stream) v.worker v.first v.last = some effect.ops ∧
      effect.after = v.input.after mc fc p.address prior ∧
      EndgameSearch.CacheHistory.Good mc fc p.width p.sizes p.key effect.after :=
  EndgameSearch.CacheHistory.Payloads.closed_at (same_run ok).2.1 v mem

end NeuralReversi.PvsParallel.NativeFrameCache

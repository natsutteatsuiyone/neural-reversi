module
public import NeuralReversi.PvsParallel.NativeFrames.OwnerSites.Admission

@[expose] public section

namespace NeuralReversi.PvsParallel.NativeFrames.OwnerSites

open EndgameSearch.CacheHistory.NativeTrace (Worker Activation Cursor Raw)
open EndgameSearch.CacheOwner (Node Resource)
open FrameReturns.OwnerCalls (ChildReturn)

variable {p : EndgameSearch.CacheHistory.Pool} {w : Worker p}
abbrev advance (inputs : Nat → Option Args) (s : State w) (events : List Event) :=
  events.foldlM (fun state e => state.advance inputs e) s

/-- The getter and all deep/shallow cache routines execute below the boundary.
    The extracted graph has no edge from them back into PVS. -/
def inCache : Node → Bool
  | .boundary | .access | .deep | .deepChild | .shallow | .shallowMoves | .shallowChild |
    .deepSlot | .deepProbe | .deepStore | .shallowSlot | .shallowProbe | .shallowStore => true
  | _ => false

theorem cache_edge {e : EndgameSearch.CacheOwner.Edge} (member : e ∈ EndgameSearch.CacheOwner.transfers)
    (ordinary : e.callee ≠ .boundary) : inCache e.caller = inCache e.callee := by
  have all : EndgameSearch.CacheOwner.transfers.all (fun e => e.callee == .boundary || inCache e.caller == inCache e.callee) = true := by decide
  simpa [ordinary] using List.all_eq_true.mp all e member

structure Caller (s : Cursor w) (node : Node) (receiver : Resource) (opening : Nat) : Prop where
  kind : s.parent.node = node
  resource : s.parent.received = receiver
  first : s.parent.first = opening
  free : inCache node = false → s.busy = false

theorem Caller.same (prior : Caller s node receiver start) (stack : t.stack = s.stack) : Caller t node receiver start := by
  obtain ⟨kind,rec,first,free⟩ := prior
  constructor <;> simp only [Cursor.parent,Cursor.busy,stack] at * <;> assumption

private theorem Caller.pushed {s t : Cursor w} (prior : Caller s caller receiver parentFirst)
    (e : EndgameSearch.CacheOwner.Edge) (member : e ∈ EndgameSearch.CacheOwner.transfers)
    (parent : e.caller = caller) (plain : e.callee ≠ .boundary)
    (top : Activation w) (stack : t.stack = top::s.stack) (kind : top.node = e.callee)
    (rec : top.received = child) (opened : top.first = opening) : Caller t e.callee child opening := by
  have head : t.parent = top := by simp [Cursor.parent,stack]
  refine ⟨head ▸ kind,head ▸ rec,head ▸ opened,?_⟩
  intro outside
  have free := prior.free ((parent ▸ cache_edge member plain).trans outside)
  simpa [Cursor.busy,stack,kind,plain] using free

/-- A worker's sequence of complete calls can be run on any earlier history.
    New rows are prepended and their opening times lie within this interval.
    The caller stack and every older row are restored, including pending rows. -/
def Segment (w : Worker p) (inputs : Nat → Option Args) (caller : Node) (receiver : Resource)
    (parentOpening lo hi : Nat) (events : List Event) : Prop :=
  ∀ (s : State w), Caller s.cursor caller receiver parentOpening →
    s.cursor.clock ≤ lo → (∀ r ∈ s.rows, r.opening < lo) →
    ∃ (out : State w) (added : List Row), advance inputs s events = some out ∧
      out.cursor.stack = s.cursor.stack ∧ out.cursor.clock ≤ hi ∧ out.rows = added ++ s.rows ∧
      ∀ r ∈ added, lo ≤ r.opening ∧ r.opening < hi

theorem Segment.nil (le : lo ≤ hi) : Segment w inputs caller receiver owner lo hi [] := by
  intro s _ tick _
  exact ⟨s,[],rfl,rfl,Nat.le_trans tick le,rfl,by simp⟩

private theorem prior_rows {old newRows : List Row} (older : ∀ r ∈ old, r.opening < lo) (window : lo ≤ hi)
    (added : ∀ r ∈ newRows, lo ≤ r.opening ∧ r.opening < hi) :
    ∀ r ∈ newRows ++ old, r.opening < hi := by
  intro r mem
  rcases List.mem_append.mp mem with recent | previous
  · exact (added r recent).2
  · exact Nat.lt_of_lt_of_le (older _ previous) window

theorem Segment.append (first : Segment w inputs caller receiver owner lo mid xs)
    (next : Segment w inputs caller receiver owner mid hi ys) (left : lo ≤ mid) (right : mid ≤ hi) :
    Segment w inputs caller receiver owner lo hi (xs ++ ys) := by
  intro s callerAt tick old
  obtain ⟨t,added,ran,stack,time,rows,range⟩ := first s callerAt tick old
  obtain ⟨u,more,ran',stack',time',rows',range'⟩ := next t (callerAt.same stack) time
    (rows ▸ prior_rows old left range)
  refine ⟨u,more++added,?_,stack'.trans stack,time',by rw [rows',rows,List.append_assoc],?_⟩
  · simp [advance,List.foldlM_append] at ran ran' ⊢
    rw [ran]; exact ran'
  · intro r mem
    rcases List.mem_append.mp mem with young | before
    · obtain ⟨a,b⟩ := range' r young; exact ⟨Nat.le_trans left a,b⟩
    · obtain ⟨a,b⟩ := range r before; exact ⟨a,Nat.lt_of_lt_of_le b right⟩

/-- Compose both sides of the physical score instruction. Either side can call
    further children, which may return with identical register contents. -/
theorem Segment.normal (edge : EndgameSearch.CacheOwner.Edge.mk caller site .pvs .forward ∈ EndgameSearch.CacheOwner.transfers)
    (a : Args) (word : ScoreRefresh.Score) (admissible : Admissible (Returns.argsFields a word.raw.toInt))
    (exact_ : a.exact = true) (input : inputs opening = some a)
    (before : Segment w inputs .pvs receiver opening (opening+1) stamp pre)
    (after : Segment w inputs .pvs receiver opening (stamp+1) finish post)
    (toScore : opening < stamp) (toReturn : stamp < finish) :
    Segment w inputs caller receiver owner opening (finish+1)
      ([.entry opening site .pvs receiver none] ++ pre ++ [.scored stamp word] ++
        post ++ [.leave finish opening .pvs none (some word)]) := by
  intro s callerAt clock older
  obtain ⟨entry,push,entryClock,entryRows,top,pushed,opened,kind,who,payload,_⟩ :=
    enter_call (inputs:=inputs) s ⟨caller,site,.pvs,.forward⟩ edge (by intro h; cases h) callerAt.kind
      (congrArg some callerAt.resource) clock (by simp [input])
  change s.advance inputs (.entry opening site .pvs receiver none) = some entry at push
  have current := callerAt.pushed ⟨caller,site,.pvs,.forward⟩ edge rfl (by intro h; cases h) top pushed kind who opened
  obtain ⟨atScore,preRows,ranPre,stackPre,preTime,rowsPre,preBound⟩ :=
    before entry current (Nat.le_of_eq entryClock)
      (fun r mem => by have := older r (entryRows ▸ mem); omega)
  rw [entryRows] at rowsPre
  have unused (r : Row) (mem : r ∈ atScore.rows) : r.opening ≠ opening := by
    rcases List.mem_append.mp (rowsPre ▸ mem) with recent | prior
    · have := (preBound _ recent).1; omega
    · have := older _ prior; omega
  obtain ⟨scored,r,mark,setClock,setRows,rKey,rStamp,pending,raw,_⟩ :=
    score_child atScore top _ (stackPre.trans pushed) kind opened a word admissible exact_ input preTime unused
  have stackScored : scored.cursor.stack = entry.cursor.stack := by simpa only [setClock] using stackPre
  have beforePost : ∀ q ∈ scored.rows, q.opening < stamp+1 := by
    rw [setRows,rowsPre]
    intro q mem
    rcases List.mem_cons.mp mem with rfl | past
    · omega
    · have := prior_rows older (by omega : opening ≤ stamp+1)
        (fun a h => by obtain ⟨l,u⟩ := preBound a h; exact ⟨by omega,by omega⟩) q past
      exact this
  obtain ⟨atReturn,postRows,ranPost,stackPost,postTime,rowsPost,postBound⟩ :=
    after scored (current.same stackScored) (by simp [setClock]) beforePost
  obtain ⟨out,q,leave,popped,now,finalRows,qKey,_⟩ :=
    return_child atReturn top _ (stackPost.trans (stackScored.trans pushed)) kind opened payload postTime exact_ input
      postRows _ r (rowsPost.trans (by rw [setRows])) rKey pending raw (by omega)
      (fun a mem => by have := (postBound _ mem).1; omega)
  refine ⟨out,postRows ++ q :: preRows,?_,popped,Nat.le_of_eq now,?_,?_⟩
  · simp only [advance,List.foldlM_append,List.foldlM_cons,List.foldlM_nil,push,
      pure,bind,Option.bind] at ranPre ranPost ⊢
    rw [ranPre]; simp only [mark]; rw [ranPost]; simp [leave]
  · rw [finalRows,rowsPre,List.append_assoc,List.cons_append]
  · intro x mem
    rcases List.mem_append.mp mem with recent | earlier
    · obtain ⟨a,b⟩ := postBound x recent; omega
    · rcases List.mem_cons.mp earlier with rfl | beforeScore
      · omega
      · obtain ⟨a,b⟩ := preBound x beforeScore; omega

inductive PlainExit (node : Node) where
  | leave
  | discarded (pvs : node = .pvs) (word : ScoreRefresh.Score)

def PlainExit.event {node : Node} (exit : PlainExit node) (finish opening : Nat) : Event :=
  match exit with
  | .leave => .leave finish opening node none none
  | .discarded _ word => .discarded finish opening word

private theorem plain_pop (s : State w) (exit : PlainExit node) (top : Activation w)
    (stack : s.cursor.stack = top::rest) (kind : top.node = node) (key : top.first = opening)
    (payload : top.operands = none) (ordered : s.cursor.clock ≤ finish)
    (registers : (inputs opening).isSome = (node == .pvs))
    (unused : ∀ r ∈ s.rows, r.opening ≠ opening) :
    s.advance inputs (exit.event finish opening) =
      some ⟨{s.cursor with stack := rest,clock := finish+1},s.rows⟩ := by
  have free : (s.rows.any (fun r => decide (r.opening = opening))) = false := by simpa using unused
  have raw : ∀ n, node = n → s.cursor.advance (.leave finish opening n none) =
      some {s.cursor with stack := rest,clock := finish+1} := by
    rintro _ rfl
    simp [Cursor.advance,Raw.step,ordered,Cursor.leave,stack,kind,key,payload]
  cases exit with
  | leave =>
    by_cases pv : node = .pvs
    · obtain ⟨a,atEntry⟩ := Option.isSome_iff_exists.mp (by simpa [pv] using registers)
      simp [PlainExit.event,State.advance,pv,raw _ pv,atEntry]
    · simp [PlainExit.event,State.advance,raw node rfl,pv]
  | discarded pv value =>
    obtain ⟨a,atEntry⟩ := Option.isSome_iff_exists.mp (by simpa [pv] using registers)
    simp [PlainExit.event,State.advance,raw _ pv,atEntry,free]

/-- Follow a non-boundary MIR transfer, including pool dispatch, split entry,
    ordinary helper calls and stopped PVS activations. A discarded score never
    creates a row, including when children have completed inside that call. -/
theorem Segment.call (e : EndgameSearch.CacheOwner.Edge) (member : e ∈ EndgameSearch.CacheOwner.transfers)
    (plain : e.callee ≠ .boundary) (route : e.flow.apply p.outside receiver = some child)
    (registers : (inputs opening).isSome = (e.callee == .pvs))
    (exit : PlainExit e.callee) (body : Segment w inputs e.callee child opening (opening+1) finish events)
    : Segment w inputs e.caller receiver owner opening (finish+1)
      ([.entry opening e.site e.callee child none] ++ events ++ [exit.event finish opening]) := by
  intro s callerAt clock older
  obtain ⟨entry,push,tick,rows,top,pushed,first,kind,who,payload,_⟩ :=
    enter_call s e member plain callerAt.kind (callerAt.resource ▸ route) clock registers
  have current := callerAt.pushed e member rfl plain top pushed kind who first
  obtain ⟨last,added,ran,stack,time,formed,bound⟩ := body entry current
    (Nat.le_of_eq tick) (fun a mem => by have := older a (rows ▸ mem); omega)
  rw [rows] at formed
  have free (r : Row) (mem : r ∈ last.rows) : r.opening ≠ opening := by
    rcases List.mem_append.mp (formed ▸ mem) with newRow | old
    · have := (bound _ newRow).1; omega
    · have := older _ old; omega
  have pop := plain_pop last exit top (stack.trans pushed) kind first payload time registers free
  refine ⟨⟨{last.cursor with stack := s.cursor.stack,clock := finish+1},last.rows⟩,added,?_,rfl,Nat.le_refl _,formed,?_⟩
  · simp only [advance,List.foldlM_append,List.foldlM_cons,List.foldlM_nil,push,pure,bind,Option.bind] at ran ⊢
    rw [ran]; simp [pop]
  · intro r mem
    obtain ⟨a,b⟩ := bound r mem
    omega

/-- Worker events can end inside any registered call, even before its score.
    A finite extension certifies the prefix; only events in the prefix are run. -/
theorem worker_prefix (w : Worker p) (inputs : Nat → Option Args) (events suffix : List Event)
    (source : Segment w inputs (EndgameSearch.CacheOwner.entry w.kind) w.root 0 0 finish (events ++ suffix)) :
    ∃ s, run w inputs events = some s := by
  obtain ⟨_,_,ran,_⟩ := source ⟨Cursor.initial w,[]⟩ ⟨rfl,rfl,rfl,by simp [Cursor.initial,Cursor.busy]⟩ (Nat.le_refl _) (by simp)
  simp only [advance,List.foldlM_append] at ran
  obtain ⟨s,ok,_⟩ := Option.bind_eq_some_iff.mp ran
  exact ⟨s,ok⟩

/-- A pool may freely interleave these worker prefixes with shared-table records.
    Construct the successful frame decode from the local source segments. -/
theorem capture (p : EndgameSearch.CacheHistory.Pool) (inputs : Nat → Nat → Option Args) (stream : List Record)
    (ids : ∀ n ev, Record.localCall n ev ∈ stream → n < (EndgameSearch.CacheOwner.poolPlan p.size).length)
    (source : ∀ w : Worker p, ∃ suffix finish, Segment w (inputs w.ordinal) (EndgameSearch.CacheOwner.entry w.kind)
      w.root 0 0 finish (NativeFrameCache.localCalls w.ordinal stream ++ suffix)) :
    ∃ cap, decode p inputs stream = some cap := by
  let plan := EndgameSearch.CacheOwner.poolPlan p.size
  have parse (n : Nat) (bound : n < plan.length) : ∃ item, workerRun p inputs stream n = some item := by
    let w : Worker p := ⟨n,plan[n],by simp [plan,bound]⟩
    have boot : EndgameSearch.CacheHistory.NativeTrace.workerAt p n = some w := by
      unfold EndgameSearch.CacheHistory.NativeTrace.workerAt
      split
      · rename_i kind got
        have h := Option.some.inj (got.symm.trans w.started)
        subst kind; rfl
      · rename_i got
        have h := got.symm.trans w.started; contradiction
    obtain ⟨suffix,_,complete⟩ := source w
    obtain ⟨s,done_⟩ := worker_prefix w _ _ suffix complete
    refine ⟨⟨w,s⟩,?_⟩
    unfold workerRun
    rw [boot]
    change (run w (inputs n) (NativeFrameCache.localCalls n stream)).bind _ = _
    rw [done_]; rfl
  have collect (ns : List Nat) (valid : ∀ n ∈ ns, n < plan.length) :
      ∃ states, ns.mapM (workerRun p inputs stream) = some states := by
    induction ns with
    | nil => exact ⟨[],rfl⟩
    | cons n ns ih =>
      obtain ⟨s,ok⟩ := parse n (valid _ List.mem_cons_self)
      obtain ⟨ss,rest⟩ := ih (fun x h => valid _ (List.mem_cons_of_mem _ h))
      exact ⟨s::ss,by simp [List.mapM_cons,ok,rest]⟩
  obtain ⟨_,workers⟩ := collect (List.range plan.length) (fun _ mem => List.mem_range.mp mem)
  apply NativeFrameCache.capture_of_workers p inputs stream _ workers
  apply List.all_eq_true.mpr
  intro record mem
  cases record with
  | table => rfl
  | localCall n e => exact decide_eq_true (ids n e mem)

end NeuralReversi.PvsParallel.NativeFrames.OwnerSites

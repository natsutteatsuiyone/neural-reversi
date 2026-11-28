module
public import NeuralReversi.PvsParallel.NativeFrames.CacheCalls.Boundary

@[expose] public section

namespace NeuralReversi.PvsParallel.NativeFrames.CacheCalls

open EndgameSearch.CacheHistory (Pool Input Local Good)
open EndgameSearch.CacheHistory.NativeTrace (Worker workerEvents)
open EndgameSearch.CacheHistory.Payloads
open NativeFrameCache (EdgeEvent)

private theorem window_prefix (w firstTick through : Nat) (clock : Nat → Nat)
    (time : ∀ {a b}, a < b → clock a < clock b) (xs : List Issued)
    (ordered : xs.Pairwise (fun a b => a.index < b.index)) (starts : ∀ x ∈ xs, firstTick ≤ clock x.index) :
    ((xs.map (fun x => Site.mk w (x.request clock))).filter (Interval.mk w firstTick through).contains).map Site.request <+:
      xs.map (Issued.request clock) := by
  induction xs with
  | nil => exact List.prefix_refl _
  | cons x rest ih =>
    obtain ⟨first,more⟩ := List.pairwise_cons.mp ordered
    have start := starts x List.mem_cons_self
    have tail := ih more (fun y hy => starts y (List.mem_cons_of_mem _ hy))
    by_cases current : clock (x.index+1) ≤ through
    · obtain ⟨extra,eq⟩ := tail
      exact ⟨extra,by simpa [Interval.contains,Issued.request,start,current] using congrArg (x.request clock :: ·) eq⟩
    · have empty : ((rest.map (fun y => Site.mk w (y.request clock))).filter (Interval.mk w firstTick through).contains) = [] := by
        apply List.filter_eq_nil_iff.mpr
        intro s mem
        obtain ⟨y,hy,rfl⟩ := List.mem_map.mp mem
        have order : clock (x.index+1) < clock (y.index+1) := time (Nat.succ_lt_succ (first y hy))
        have tooLate : ¬ clock (y.index+1) ≤ through := by omega
        simp [Interval.contains,Issued.request,tooLate]
      have skip : (Interval.mk w firstTick through).contains ⟨w,x.request clock⟩ = false := by
        simp [Interval.contains,Issued.request,current]
      simp only [List.map_cons,List.filter_cons,skip,Bool.false_eq_true,ite_false,empty,List.map_nil]
      exact List.nil_prefix

private theorem mapM_prefix {f : α → Option β} (xs : List α) (ok : xs.mapM f = some ys) (k : Nat) :
    (xs.take k).mapM f = some (ys.take k) := by
  induction xs generalizing k ys with
  | nil => simp_all
  | cons x rest ih =>
    rw [List.mapM_cons] at ok
    obtain ⟨v,first,ok⟩ := Option.bind_eq_some_iff.mp ok
    obtain ⟨tail,more,equal⟩ := Option.bind_eq_some_iff.mp ok
    cases equal
    cases k with
    | zero => rfl
    | succ n => simp [List.mapM_cons,first,ih more n]

private theorem observe_take (p : Pool) (ops : List Instruction) (s : Local) (shape : ∀ t, (s.cells t).size = p.sizes t) (k : Nat) :
    (observeRun p s shape ops).take k = observeRun p s shape (ops.take k) := by
  induction ops generalizing s k with
  | nil => simp [observeRun]
  | cons op rest ih =>
    cases k with
    | zero => rfl
    | succ n => simp [observeRun,ih]

/-- A raw prefix may stop in a recursive child or between a primitive's entry
    and return. The requests visible by any clock cutoff are exactly an initial
    segment of the issued access list and audit against the full program. -/
theorem Routine.partial_run (r : Routine) (w : Worker p) (site owner : Nat) (i : Input) (branch : i.arm = r.arm)
    (stamp : Option Nat) (clock : Nat → Nat) (time : ∀ {a b}, a < b → clock a < clock b) (n : Nat)
    (s : Local) (g : Good mc fc p.width p.sizes p.key s) (other : Values) (limit through : Nat)
    (blocks : List (Block p)) (ordered : blocks.Pairwise Block.earlier) (before after_ : List EdgeEvent)
    (stream : List (EndgameSearch.CacheHistory.NativeTrace.Event S)) :
    let code := r.program p i mc fc s branch
    let values := Block.registers blocks other
    let raw := (r.stream w site owner i stamp code.forest clock n).take limit
    let k := (requests raw w.ordinal (clock n) through).length
    let ops := (code.accesses.map Access.instruction).take k
    let samples := observeRun p s g.shape ops
    code.block w clock time n s g.shape ∈ blocks →
    workerEvents w.ordinal stream = workerEvents w.ordinal (before ++ raw ++ after_) →
    requests before w.ordinal (clock n) through = [] → requests after_ w.ordinal (clock n) through = [] →
    collect values stream w.ordinal (clock n) through = some samples ∧
      check p false s (i.run mc fc p.address s.history).history samples = some ⟨samples,ops.foldl (fun st op => op.step p st) s⟩ := by
  dsimp only
  intro member interleaved earlier later
  let code := r.program p i mc fc s branch
  let xs := code.schedule (thread w) (n+4)
  let full := r.stream w site owner i stamp code.forest clock n
  let ask := requests (full.take limit) w.ordinal (clock n) through
  have prefixReq : ask <+: xs.map (Issued.request clock) := by
    have selected := ((((List.take_prefix limit full).filterMap site?).filter (Interval.mk w.ordinal (clock n) through).contains).map Site.request)
    apply selected.trans
    rw [show full.filterMap site? = _ from (r.sites w site owner i branch stamp code.forest clock n).trans
      (code.schedule_sites w.ordinal (thread w) clock (n+4))]
    have bounds := code.schedule_order (thread w) (n+4)
    exact window_prefix _ _ _ _ time xs bounds.1 (fun x hx => Nat.le_of_lt (time (by have := (bounds.2 x hx).1; omega)))
  let values := Block.registers blocks other
  have read := code.read_packets (thread w) (n+4) p w.ordinal clock time values s g.shape
  have installed := Block.install_registers blocks ordered other _ member
  change Issued.install p w.ordinal clock values xs s g.shape = values at installed
  dsimp only at read
  rw [installed] at read
  have truncated := mapM_prefix _ read ask.length
  rw [observe_take] at truncated
  have agreed : ask.mapM _ = _ := (congrArg (List.mapM _) (List.prefix_iff_eq_take.mp prefixReq)).trans truncated
  have sound := (EndgameSearch.CacheHistory.after_call i g (EndgameSearch.CacheMir.budget_fits p.size p.width p.word)).1.valid
  refine ⟨?_,code.audit_prefix sound p g.shape ask.length⟩
  rw [collect_interleaved values stream _ _ _ _ interleaved]
  unfold collect
  rw [requests_append,requests_append,earlier,later,List.nil_append,List.append_nil]
  exact agreed

/-- Replay a still-open native boundary against the worker cursor's own
    clock. The caller may cut its stream after any nested entry or return. -/
theorem Routine.pending_run (r : Routine) (w : Worker p) (site owner : Nat) (i : Input) (branch : i.arm = r.arm)
    (stamp : Option Nat) (clock : Nat → Nat) (time : ∀ {a b}, a < b → clock a < clock b) (n : Nat)
    (timeline : EndgameSearch.CacheHistory.Timeline p) (cursor : EndgameSearch.CacheHistory.NativeTrace.Cursor w)
    (frame : EndgameSearch.CacheHistory.NativeTrace.Activation w)
    (found : cursor.stack.find? (fun f => f.node == .boundary) = some frame)
    (args : frame.operands = some (boundaryArgs w owner i)) (opened : frame.first = clock n)
    (limit : Nat) (before after_ : List EdgeEvent) (stream : List (EndgameSearch.CacheHistory.NativeTrace.Event S)) (other : Values) :
    let s := timeline.readAt mc fc w.ordinal (clock n)
    let g := EndgameSearch.CacheHistory.replay_good mc fc p
      (EndgameSearch.CacheHistory.beforeStep timeline.visits w.ordinal (clock n))
    let code := r.program p i mc fc s branch
    let block_ := code.block w clock time n s g.shape
    let raw := (r.stream w site owner i stamp code.forest clock n).take limit
    let k := (requests raw w.ordinal (clock n) cursor.clock).length
    let ops := (code.accesses.map Access.instruction).take k
    workerEvents w.ordinal stream = workerEvents w.ordinal (before ++ raw ++ after_) →
    requests before w.ordinal (clock n) cursor.clock = [] → requests after_ w.ordinal (clock n) cursor.clock = [] →
    pendingWorker p timeline mc fc (block_.install other) stream ⟨w,cursor⟩ =
      some (some ⟨observeRun p s g.shape ops,ops.foldl (fun st op => op.step p st) s⟩) := by
  dsimp only
  intro interleaved earlier later
  let s := timeline.readAt mc fc w.ordinal (clock n)
  have g : Good mc fc p.width p.sizes p.key s := EndgameSearch.CacheHistory.replay_good mc fc p
    (EndgameSearch.CacheHistory.beforeStep timeline.visits w.ordinal (clock n))
  let code := r.program p i mc fc s branch
  let b := code.block w clock time n s g.shape
  obtain ⟨read,audit⟩ := r.partial_run w site owner i branch stamp clock time n s g other limit cursor.clock
    [b] (List.pairwise_singleton ..) before after_ stream List.mem_cons_self interleaved earlier later
  unfold pendingWorker
  simp only [found,args,opened,boundaryArgs]
  change (do
    let ops ← collect (Block.registers [b] other) stream w.ordinal (clock n) cursor.clock
    let partialRun ← check p false s (i.run mc fc p.address s.history).history ops
    return some partialRun) = _
  rw [read]
  simp [audit]
  exact ⟨rfl,rfl⟩

end NeuralReversi.PvsParallel.NativeFrames.CacheCalls

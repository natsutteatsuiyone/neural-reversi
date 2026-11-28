module
public import NeuralReversi.EndgameSearch.CacheHistory.NativeTrace

@[expose] public section

namespace NeuralReversi.EndgameSearch.CacheHistory.Payloads

open EndgameTreeCache (Store Table)
open EndgameCache (Entry Write)
open CacheState (Cells)
open CacheOwner (Node)
open NativeTrace (Event Raw)

structure Words where
  player : BitVec 64
  opponent : BitVec 64
  lower : BitVec 8
  upper : BitVec 8
  deriving DecidableEq

def words (e : Entry) : Words := ⟨e.board.player,e.board.opponent,e.lower.toBitVec,e.upper.toBitVec⟩

inductive Data where
  | slot (hash : BitVec 64) (returned : Nat)
  | probe (index : Nat) (board : Board.Board) (alpha : BitVec 32) (loaded : Words) (returned : Option (BitVec 32))
  | store (index : Nat) (board : Board.Board) (alpha score : BitVec 32) (loaded written : Words)

structure Op where
  table : Table
  data : Data

def Op.write? (o : Op) : Option Store :=
  match o.data with
  | .store _ b a s _ _ => some ⟨o.table,b,a.toInt,s.toInt⟩
  | _ => none

/-- Both 64-bit bitboards and both signed bound bytes are compared to the recorded memory words. -/
def execute (p : Pool) (s : Local) (o : Op) : Option Local := do
  let t := o.table
  let cells := s.cells t
  match o.data with
  | .slot key returned =>
    if returned = CacheMir.slot p.width cells.size key then some s else none
  | .probe idx board alpha loaded result =>
    let e ← cells[idx]?
    if idx = CacheMir.slot p.width cells.size (p.key board) ∧ loaded = words e ∧
       CacheMir.probe cells idx board alpha.toInt = some (result.map BitVec.toInt) then some s else none
  | .store idx board alpha score loaded written =>
    let old ← cells[idx]?
    if idx = CacheMir.slot p.width cells.size (p.key board) ∧ loaded = words old ∧
       written = words (CacheMir.storeCell old board alpha.toInt score.toInt) then
      let w : Store := ⟨t,board,alpha.toInt,score.toInt⟩
      let next ← CacheMir.writeCells p.width p.key s.cells w
      return ⟨s.history ++ [w],next⟩
    else none

def applyWrite (p : Pool) (s : Local) (w : Store) : Local :=
  ⟨s.history ++ [w],CacheState.write p.address s.cells w⟩

theorem step_writes (p : Pool) (o : Op) (shape : ∀ t, (s.cells t).size = p.sizes t)
    (ok : execute p s o = some after) : after = o.write?.elim s (applyWrite p s) := by
  cases o with | mk t data =>
    cases data with
    | slot key observed =>
      simp [execute] at ok
      exact ok.2.symm
    | probe idx b a e r =>
      unfold execute at ok
      cases read : (s.cells t)[idx]? with
      | none => simp [read] at ok
      | some old =>
        simp [read] at ok
        exact ok.2.symm
    | store idx b a v old written =>
      let w : Store := ⟨t,b,a.toInt,v.toInt⟩
      have yes := CacheMir.write_eq (hash:=p.key) w shape (CacheMir.budget_fits p.size p.width p.word)
      unfold execute at ok
      cases read : (s.cells t)[idx]? with
      | none => simp [read] at ok
      | some cell =>
        change CacheMir.writeCells p.width p.key s.cells ⟨t,b,a.toInt,v.toInt⟩ = _ at yes
        simp [read,yes] at ok
        exact ok.2.symm

theorem execute_probe (p : Pool) (s : Local) (t : Table)
    (ok : execute p s ⟨t,.probe idx board alpha loaded result⟩ = some after) :
    ∃ entry, (s.cells t)[idx]? = some entry ∧
      idx = CacheMir.slot p.width (s.cells t).size (p.key board) ∧ loaded = words entry ∧
      CacheMir.probeCell entry board alpha.toInt = result.map BitVec.toInt ∧ after = s := by
  unfold execute at ok
  cases read : (s.cells t)[idx]? with
  | none => simp [read] at ok
  | some entry =>
    simp [CacheMir.probe,read] at ok
    exact ⟨entry,rfl,ok.1.1,ok.1.2.1,ok.1.2.2,ok.2.symm⟩

def writes (ops : List Op) : List Store := ops.filterMap Op.write?

theorem trace_writes (p : Pool) (ops : List Op) (shape : ∀ t, (s.cells t).size = p.sizes t)
    (ok : ops.foldlM (execute p) s = some after) :
    after.history = s.history ++ writes ops ∧ after.cells = (writes ops).foldl (CacheState.write p.address) s.cells := by
  induction ops generalizing s with
  | nil => simp at ok; cases ok; simp [writes]
  | cons o rest ih =>
    cases first : execute p s o with
    | none => simp [List.foldlM_cons,first] at ok
    | some next =>
      simp [List.foldlM_cons,first] at ok
      have update := step_writes p o shape first
      cases w : o.write? with
      | none =>
        simp [w] at update
        subst next
        simpa [writes,w] using ih shape ok
      | some item =>
        simp [w] at update
        subst next
        have size (t) : ((applyWrite p s item).cells t).size = p.sizes t := by
          by_cases same : t = item.table <;> simp [applyWrite,CacheState.write,same,shape]
        simpa [writes,w,applyWrite] using ih size ok

inductive Kind where
  | slot | probe | store
  deriving DecidableEq

def Data.kind : Data → Kind | .slot .. => .slot | .probe .. => .probe | .store .. => .store

def target : Node → Option (Table × Kind)
  | .deepSlot => some (.deep,.slot) | .shallowSlot => some (.shallow,.slot)
  | .deepProbe => some (.deep,.probe) | .shallowProbe => some (.shallow,.probe)
  | .deepStore => some (.deep,.store) | .shallowStore => some (.shallow,.store)
  | _ => none

structure Request where
  opening : Nat
  returned : Nat
  table : Table
  kind : Kind

structure Packet where
  returned : Nat
  data : Data

abbrev Values := Nat → Nat → Option Packet

structure Site where
  worker : Nat
  request : Request

def site? : Event S → Option Site
  | .worker n (.leave stamp opening callee _) => do
    let (t,k) ← target callee
    return ⟨n,opening,stamp,t,k⟩
  | _ => none

structure Interval where
  worker : Nat
  first : Nat
  last : Nat

def Interval.contains (a : Interval) (s : Site) : Bool :=
  a.worker == s.worker && decide (a.first ≤ s.request.opening ∧ s.request.returned ≤ a.last)

def requests (stream : List (Event S)) (worker first last : Nat) : List Request :=
  let scope : Interval := ⟨worker,first,last⟩
  ((stream.filterMap site?).filter scope.contains).map Site.request

def intervals (c : NativeTrace.Capture p S) : List Interval :=
  c.timeline.visits.map (fun v => ⟨v.worker,v.first,v.last⟩) ++ c.workers.filterMap fun w => do
    let f ← w.2.stack.find? (fun frame => frame.node == .boundary)
    return ⟨w.1.ordinal,f.first,w.2.clock⟩

def covered (c : NativeTrace.Capture p S) (stream : List (Event S)) : Bool :=
  (stream.filterMap site?).all fun s =>
    let count := ((intervals c).filter (fun window => window.contains s)).length
    if s.request.kind == .store || s.request.table == .shallow then count == 1 else count ≤ 1

def packet (values : Values) (worker : Nat) (r : Request) : Option Op := do
  let v ← values worker r.opening
  if v.returned = r.returned ∧ v.data.kind = r.kind then some ⟨r.table,v.data⟩ else none

theorem matched_operand (values : Values) (worker : Nat) (r : Request) (accepted : packet values worker r = some op) :
    ∃ observed, values worker r.opening = some observed ∧ observed.returned = r.returned ∧
      observed.data.kind = r.kind ∧ op = ⟨r.table,observed.data⟩ := by
  unfold packet at accepted
  cases found : values worker r.opening with
  | none => simp [found] at accepted
  | some v =>
    simp [found] at accepted
    exact ⟨v,rfl,accepted.1.1,accepted.1.2,accepted.2.symm⟩

def collect (values : Values) (stream : List (Event S)) (worker first last : Nat) : Option (List Op) :=
  (requests stream worker first last).mapM (packet values worker)

instance : DecidableEq Write := fun x y =>
  decidable_of_iff (x.board=y.board ∧ x.alpha=y.alpha ∧ x.score=y.score) (by cases x; cases y; simp)

instance : DecidableEq Store := fun x y =>
  decidable_of_iff (x.table=y.table ∧ x.write=y.write) (by cases x; cases y; simp)

structure Audit where
  ops : List Op
  after : Local

/-- A complete physical trace is checked for equality with the recursive program's full write
    sequence even when the caller discarded its return. A finite outstanding call checks a prefix. -/
def check (p : Pool) (complete : Bool) (s : Local) (expected : List Store) (ops : List Op) : Option Audit := do
  let after ← ops.foldlM (execute p) s
  let wanted := if complete then expected else expected.take after.history.length
  if after.history = wanted then some ⟨ops,after⟩ else none

theorem check_result (p : Pool) (s : Local) (ops : List Op)
    (accepted : check p complete s expected ops = some audit) :
    audit.ops = ops ∧ ops.foldlM (execute p) s = some audit.after ∧
      audit.after.history = (if complete then expected else expected.take audit.after.history.length) := by
  unfold check at accepted
  cases saved : ops.foldlM (execute p) s with
  | none => simp [saved] at accepted
  | some after =>
    change (ops.foldlM (execute p) s).bind (fun st =>
      if st.history = (if complete then expected else expected.take st.history.length) then some ⟨ops,st⟩ else none)
      = some audit at accepted
    rw [saved] at accepted
    dsimp only [Option.bind] at accepted
    by_cases history : after.history = (if complete then expected else expected.take after.history.length)
    · rw [ite_eq_left history] at accepted
      cases accepted
      exact ⟨rfl,rfl,history⟩
    · rw [ite_eq_right history] at accepted
      contradiction

theorem complete_state (i : Input) (g : Good mc fc p.width p.sizes p.key s) (ops : List Op)
    (accepted : check p true s (i.run mc fc p.address s.history).history ops = some audit) :
    audit.after = i.after mc fc p.address s ∧ Good mc fc p.width p.sizes p.key audit.after := by
  obtain ⟨_,physical,history⟩ := check_result p s ops accepted
  have grew := trace_writes p ops g.shape physical
  change audit.after.history = (i.run mc fc p.address s.history).history at history
  have dropped : (i.run mc fc p.address s.history).history.drop s.history.length = writes ops := by
    rw [← history,grew.1]; simp
  have arrays : audit.after.cells = (i.after mc fc p.address s).cells := by
    simpa [Input.after,dropped] using grew.2
  have equal : audit.after = i.after mc fc p.address s := by
    change Local.mk audit.after.history audit.after.cells = Local.mk (i.run mc fc p.address s.history).history _
    rw [history,arrays]
    rfl
  exact ⟨equal,equal ▸ (after_call i g (CacheMir.budget_fits p.size p.width p.word)).1⟩

theorem aligned_ops (p : Pool) (ops : List Op)
    (shape : ∀ t, (s.cells t).size = p.sizes t)
    (aligned : CacheState.represents p.address s.history s.cells)
    (ran : ops.foldlM (execute p) s = some out) :
    (∀ t, (out.cells t).size = p.sizes t) ∧ CacheState.represents p.address out.history out.cells := by
  obtain ⟨history,arrays⟩ := trace_writes p ops shape ran
  refine ⟨?_,?_⟩
  · rw [arrays]
    have preserve (ws : List Store) (start : Cells) (t : Table) :
        (ws.foldl (CacheState.write p.address) start t).size = (start t).size := by
      induction ws generalizing start with
      | nil => rfl
      | cons w rest ih =>
        simp only [List.foldl_cons,ih]
        by_cases same : t = w.table <;> simp [CacheState.write,same]
    exact fun t => (preserve _ _ t).trans (shape t)
  · rw [history,arrays]
    exact CacheState.replay_represents aligned _

theorem prefix_valid (i : Input) (g : Good mc fc p.width p.sizes p.key s) (ops : List Op)
    (accepted : check p complete s (i.run mc fc p.address s.history).history ops = some audit) :
    Good mc fc p.width p.sizes p.key audit.after := by
  obtain ⟨_,physical,history⟩ := check_result p s ops accepted
  obtain ⟨shape,arrays⟩ := aligned_ops p ops g.shape g.arrays physical
  have final := (after_call i g (CacheMir.budget_fits p.size p.width p.word)).1.valid
  refine ⟨?_,shape,arrays⟩
  · intro event mem
    apply final event
    cases complete with
    | false =>
      simp only [Bool.false_eq_true,ite_false] at history
      rw [history] at mem
      exact List.mem_of_mem_take mem
    | true =>
      simp only [ite_true] at history
      rw [history] at mem
      exact mem

theorem interior_read (i : Input) (g : Good mc fc p.width p.sizes p.key s) (t : Table) (b : PvsBoardTree.Position)
    (accepted : check p complete s (i.run mc fc p.address s.history).history
        (before ++ ⟨t,.probe idx b.val alpha loaded (some raw)⟩ :: rest) = some audit) :
    EndgameCache.CacheRange raw.toInt ∧ Solve2.NWS alpha.toInt (EndgameTreeCache.discValue mc fc b.val) raw.toInt := by
  have final := prefix_valid i g _ accepted
  obtain ⟨_,trace,_⟩ := check_result p s _ accepted
  rw [List.foldlM_append] at trace
  obtain ⟨atRead,prior,remaining⟩ := Option.bind_eq_some_iff.mp trace
  obtain ⟨size,represents⟩ := aligned_ops p before g.shape g.arrays prior
  have extendsHistory := (trace_writes p _ size remaining).1
  have sound : EndgameTreeCache.soundHistory mc fc atRead.history := fun ev member =>
    final.valid ev (extendsHistory.symm ▸ List.mem_append_left _ member)
  rw [List.foldlM_cons] at remaining
  obtain ⟨_,probed,_⟩ := Option.bind_eq_some_iff.mp remaining
  obtain ⟨entry,read,index,_,response,_⟩ := execute_probe p atRead t probed
  apply CacheMir.probe_bound t b (size t) represents sound
  rw [←index]
  simp [CacheMir.probe,read,response]

def completeVisit (p : Pool) (timeline : Timeline p) (mc : Bitboard.Moves.X86Config) (fc : FlipConfig)
    (values : Values) (stream : List (Event S)) (v : Visit p) : Option Audit := do
  let ops ← collect values stream v.worker v.first v.last
  let prior := timeline.readAt mc fc v.worker v.first
  check p true prior (v.input.run mc fc p.address prior.history).history ops

theorem completeVisit_state (p : Pool) (timeline : Timeline p) (mc : Bitboard.Moves.X86Config) (fc : FlipConfig)
    (values : Values) (stream : List (Event S)) (v : Visit p)
    (ok : completeVisit p timeline mc fc values stream v = some out) :
    let prior := timeline.readAt mc fc v.worker v.first
    collect values stream v.worker v.first v.last = some out.ops ∧
      out.after = v.input.after mc fc p.address prior ∧ Good mc fc p.width p.sizes p.key out.after := by
  unfold completeVisit at ok
  cases ops : collect values stream v.worker v.first v.last with
  | none => simp [ops] at ok
  | some actual =>
    simp [ops] at ok
    have before := replay_good mc fc p (beforeStep timeline.visits v.worker v.first)
    have reply := (check_result p _ actual ok).1
    exact ⟨congrArg some reply.symm,complete_state v.input before actual ok⟩

def pendingWorker (p : Pool) (timeline : Timeline p) (mc : Bitboard.Moves.X86Config) (fc : FlipConfig)
    (values : Values) (stream : List (Event S)) (w : (owner : NativeTrace.Worker p) × NativeTrace.Cursor owner) :
    Option (Option Audit) := do
  match w.2.stack.find? (fun frame => frame.node == .boundary) with
  | none => return none
  | some frame =>
    let args ← frame.operands
    let ops ← collect values stream w.1.ordinal frame.first w.2.clock
    let prior := timeline.readAt mc fc w.1.ordinal frame.first
    let partialRun ← check p false prior (args.input.run mc fc p.address prior.history).history ops
    return some partialRun

structure Checked (p : Pool) (S : Type) where
  native : NativeTrace.Capture p S
  finished : List Audit
  pending : List (Option Audit)

def decode (p : Pool) (mc : Bitboard.Moves.X86Config) (fc : FlipConfig) (values : Values)
    (stream : List (Event S)) : Option (Checked p S) := do
  let c ← NativeTrace.decode p stream
  if !covered c stream then none else
    let finished ← c.timeline.visits.mapM (completeVisit p c.timeline mc fc values stream)
    let pending ← c.workers.mapM (pendingWorker p c.timeline mc fc values stream)
    return ⟨c,finished,pending⟩

theorem observed {out : Checked p S} (ok : decode p mc fc values stream = some out) :
    NativeTrace.decode p stream = some out.native ∧ out.native.shared = NativeTrace.sharedEvents stream ∧
      out.native.timeline.visits.mapM (completeVisit p out.native.timeline mc fc values stream) = some out.finished ∧
      out.native.workers.mapM (pendingWorker p out.native.timeline mc fc values stream) = some out.pending ∧
      ∀ site ∈ stream.filterMap site?,
        let count := ((intervals out.native).filter (fun window => window.contains site)).length
        if site.request.kind == .store || site.request.table == .shallow then count = 1 else count ≤ 1 := by
  unfold decode at ok
  obtain ⟨c,caps,body⟩ := Option.bind_eq_some_iff.mp ok
  split at body
  · contradiction
  · rename_i coverage
    obtain ⟨closed,ends,body⟩ := Option.bind_eq_some_iff.mp body
    obtain ⟨openCalls,more,returned⟩ := Option.bind_eq_some_iff.mp body
    cases returned
    refine ⟨caps,(NativeTrace.decode_keeps _ _ caps).1,ends,more,?_⟩
    intro site inside
    have all : covered c stream = true := by simpa using coverage
    have item := List.all_eq_true.mp all site inside
    dsimp only at item ⊢
    by_cases h : (site.request.kind == .store || site.request.table == .shallow) = true
    · simp only [h, ite_true] at item ⊢
      exact beq_iff_eq.mp item
    · simp only [h, Bool.false_eq_true, ite_false] at item ⊢
      exact of_decide_eq_true item

theorem mapped {f : α → Option β} {xs : List α} {ys : List β} (success : xs.mapM f = some ys) {value : α} (inside : value ∈ xs) :
    ∃ result ∈ ys, f value = some result := by
  induction xs generalizing ys with
  | nil => simp at inside
  | cons x rest ih =>
    rw [List.mapM_cons] at success
    obtain ⟨first,read,success⟩ := Option.bind_eq_some_iff.mp success
    obtain ⟨tail,more,same⟩ := Option.bind_eq_some_iff.mp success
    change some (first :: tail) = some ys at same
    cases same
    rcases List.mem_cons.mp inside with rfl | old
    · exact ⟨first,by simp,read⟩
    · obtain ⟨r,member,hit⟩ := ih more old
      exact ⟨r,List.mem_cons_of_mem _ member,hit⟩

theorem closed_at {out : Checked p S} (accepted : decode p mc fc values stream = some out)
    (v : Visit p) (present : v ∈ out.native.timeline.visits) :
    let prior := out.native.timeline.readAt mc fc v.worker v.first
    ∃ effect ∈ out.finished, collect values stream v.worker v.first v.last = some effect.ops ∧
      effect.after = v.input.after mc fc p.address prior ∧ Good mc fc p.width p.sizes p.key effect.after := by
  obtain ⟨effect,mem,observed⟩ := mapped (observed accepted).2.2.1 present
  exact ⟨effect,mem,completeVisit_state _ _ _ _ _ _ _ observed⟩

theorem hit_in_closed_call {out : Checked p S} (accepted : decode p mc fc values stream = some out)
    (v : Visit p) (present : v ∈ out.native.timeline.visits) (ops : List Op)
    (read : collect values stream v.worker v.first v.last = some ops)
    (t : Table) (b : PvsBoardTree.Position)
    (occurred : Op.mk t (.probe idx b.val alpha loaded (some raw)) ∈ ops) :
    EndgameCache.CacheRange raw.toInt ∧ Solve2.NWS alpha.toInt (EndgameTreeCache.discValue mc fc b.val) raw.toInt := by
  obtain ⟨effect,_,observed⟩ := mapped (observed accepted).2.2.1 present
  unfold completeVisit at observed
  simp [read] at observed
  obtain ⟨before,rest,cut⟩ := List.append_of_mem occurred
  rw [cut] at observed
  exact interior_read v.input (replay_good mc fc p (beforeStep out.native.timeline.visits v.worker v.first)) t b observed

theorem pending_state (timeline : Timeline p) (w : (worker : NativeTrace.Worker p) × NativeTrace.Cursor worker)
    (finished : pendingWorker p timeline mc fc values stream w = some (some audit)) :
    ∃ frame operands, w.2.stack.find? (fun f => f.node == .boundary) = some frame ∧ frame.operands = some operands ∧
      collect values stream w.1.ordinal frame.first w.2.clock = some audit.ops ∧
      Good mc fc p.width p.sizes p.key audit.after := by
  unfold pendingWorker at finished
  split at finished
  · cases finished
  · rename_i frame hasFrame
    obtain ⟨args,argsFound,more⟩ := Option.bind_eq_some_iff.mp finished
    obtain ⟨ops,read,more⟩ := Option.bind_eq_some_iff.mp more
    obtain ⟨result,matched,same⟩ := Option.bind_eq_some_iff.mp more
    cases same
    have g := replay_good mc fc p (beforeStep timeline.visits w.1.ordinal frame.first)
    have valid := prefix_valid args.input g ops matched
    have seq := (check_result p _ ops matched).1
    exact ⟨frame,args,hasFrame,argsFound,seq ▸ read,valid⟩

theorem hit_in_pending_call {out : Checked p S} (accepted : decode p mc fc values stream = some out)
    (w : (owner : NativeTrace.Worker p) × NativeTrace.Cursor owner) (inside : w ∈ out.native.workers)
    (frame : NativeTrace.Activation w.1) (openLeaf : w.2.stack.find? (fun f => f.node == .boundary) = some frame)
    (args : NativeTrace.Operands) (captured : frame.operands = some args) (ops : List Op)
    (read : collect values stream w.1.ordinal frame.first w.2.clock = some ops)
    (t : Table) (b : PvsBoardTree.Position)
    (occurred : Op.mk t (.probe idx b.val alpha loaded (some raw)) ∈ ops) :
    EndgameCache.CacheRange raw.toInt ∧ Solve2.NWS alpha.toInt (EndgameTreeCache.discValue mc fc b.val) raw.toInt := by
  obtain ⟨result,_,audit⟩ := mapped (observed accepted).2.2.2.1 inside
  unfold pendingWorker at audit
  rw [openLeaf] at audit
  dsimp only at audit
  rw [captured] at audit
  simp only [read] at audit
  obtain ⟨_,matched,_⟩ := Option.bind_eq_some_iff.mp (show (check p false _ _ ops).bind _ = _ from audit)
  obtain ⟨before,rest,cut⟩ := List.append_of_mem occurred
  rw [cut] at matched
  exact interior_read args.input (replay_good mc fc p (beforeStep out.native.timeline.visits w.1.ordinal frame.first)) t b matched

end NeuralReversi.EndgameSearch.CacheHistory.Payloads

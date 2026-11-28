module
public import NeuralReversi.PvsParallel.NativeFrames
public import NeuralReversi.PvsParallel.FrameReturns.OwnerCalls

@[expose] public section

namespace NeuralReversi.PvsParallel.NativeFrames.Returns

open EndgameSearch.CacheHistory (Pool)
open EndgameSearch.CacheOwner (poolPlan)
open EndgameSearch.CacheHistory.NativeTrace (Worker Cursor Activation)
open FrameReturns.OwnerCalls (ChildReturn fields)

/-- The actual return instruction identifies the call by its worker and opening.
    The entry supplies the window, board and node flags. -/
structure Returned where
  worker : Nat
  opening : Nat
  id : Nat
  step : Nat
  parent : Nat
  rootLine : Option Nat
  call : ChildReturn
  deriving DecidableEq

def argsFields (a : Args) (value : Int) : ChildReturn :=
  ⟨a.board,a.depth.toNat,a.pv,a.cut,a.alpha.raw.toInt,a.beta.raw.toInt,value⟩

def fromRow (r : Row) : Option Returned :=
  r.frame.returnStep.map fun t => ⟨r.frame.worker,r.opening,r.frame.id,t,r.frame.parent,r.rootLine,fields r.frame⟩

abbrev Stack := List (Nat × EndgameSearch.CacheOwner.Node)

def parentKey (size worker : Nat) : Stack → Nat
  | (opening,.pvs) :: _ => frameKey size worker opening
  | _ => 0

def project (s : Cursor w) : Stack := s.stack.map fun a => (a.first,a.node)

/-- Only entry/leave control flow and the parent at the scoring instruction are
    needed to recover a direct caller. No decoded score row is consulted. -/
structure Walk where
  stack : Stack := []
  parents : List (Nat × Nat) := []

def Walk.advance (s : Walk) (size w : Nat) (inputs : Nat → Option Args) : Event → Walk
  | .entry t _ node .. => {s with stack := (t,node) :: s.stack}
  | .leave .. | .discarded .. => {s with stack := s.stack.tail}
  | .scored .. =>
    match s.stack with
    | (t,.pvs) :: rest => if (inputs t).any Args.exact then
          {s with parents := (t,parentKey size w rest) :: s.parents} else s
    | _ => s

def fromEvent (size worker : Nat) (inputs : Nat → Option Args) (walk : Walk) : Event → Option Returned
  | .leave t opening .pvs _ (some value) => do
    let a ← inputs opening
    if a.exact then some ⟨worker,opening,frameKey size worker opening,t,(walk.parents.lookup opening).getD 0,a.rootLine,argsFields a value.raw.toInt⟩ else none
  | _ => none

private def FromEntry (size worker : Nat) (inputs : Nat → Option Args) (parents : List (Nat × Nat)) (r : Row) : Prop :=
  parents.lookup r.opening = some r.frame.parent ∧ r.frame.id = frameKey size worker r.opening ∧ ∃ a, inputs r.opening = some a ∧ a.exact = true ∧ r.frame.worker = worker ∧ r.rootLine = a.rootLine ∧
    fields r.frame = argsFields a r.frame.score

private theorem created {a : Args} (args : inputs opening = some a) (exact_ : a.exact = true)
    (got : a.row size worker opening parent step value = some r) :
    FromEntry size worker inputs ((opening,parent) :: parents) r ∧ fromRow r = none := by
  unfold Args.row at got
  dsimp only at got
  split at got <;> try contradiction
  split at got <;> try contradiction
  cases got
  refine ⟨⟨List.lookup_cons_self,rfl,a,args,exact_,rfl,rfl,?_⟩,rfl⟩
  simp [fields,argsFields,TraceCheck.FrameReturn.board]

private theorem finished {r r' : Row} (prior : FromEntry size w inputs parents r) (args : inputs r.opening = some a)
    (ok : r.finish t value = some r') :
    FromEntry size w inputs parents r' ∧ fromRow r = none ∧
      fromRow r' = some ⟨w,r.opening,frameKey size w r.opening,t,(parents.lookup r.opening).getD 0,a.rootLine,argsFields a value.raw.toInt⟩ := by
  obtain ⟨parent,id,op,source,exact_,worker,line,data⟩ := prior
  obtain rfl := Option.some.inj (source.symm.trans args)
  unfold Row.finish at ok
  split at ok <;> try contradiction
  split at ok <;> try contradiction
  rename_i valid
  obtain ⟨empty,score⟩ := Bool.and_eq_true_iff.mp valid
  have unreturned := Option.isNone_iff_eq_none.mp empty
  have same := of_decide_eq_true score
  cases ok
  refine ⟨⟨parent,id,_,source,exact_,worker,line,data⟩,by simp [fromRow,unreturned],?_⟩
  change some (⟨r.frame.worker,r.opening,r.frame.id,t,r.frame.parent,r.rootLine,fields r.frame⟩ : Returned) = _
  rw [worker,line,data,same,parent,id]
  rfl

private theorem finish_rows (rows : List Row) (known : ∀ r ∈ rows, FromEntry size w inputs parents r)
    (args : inputs opening = some a) (ok : finishRow opening t value rows = some out) :
    (∀ r ∈ out, FromEntry size w inputs parents r) ∧
    (out.filterMap fromRow).Perm (⟨w,opening,frameKey size w opening,t,(parents.lookup opening).getD 0,a.rootLine,argsFields a value.raw.toInt⟩ :: rows.filterMap fromRow) := by
  induction rows generalizing out with
  | nil => simp [finishRow] at ok
  | cons r rest ih =>
    rw [finishRow] at ok
    split at ok
    next same =>
      obtain ⟨r',done_,eq⟩ := Option.map_eq_some_iff.mp ok
      subst out
      obtain ⟨valid,old,new⟩ := finished (known r List.mem_cons_self) (same ▸ args) done_
      refine ⟨?_,by simp [old,new,same]⟩
      intro q member
      rcases List.mem_cons.mp member with rfl | tail
      · exact valid
      · exact known _ (List.mem_cons_of_mem _ tail)
    next =>
      obtain ⟨tail,more,eq⟩ := Option.map_eq_some_iff.mp ok
      subst out
      obtain ⟨valid,perm⟩ := ih (fun q inTail => known q (List.mem_cons_of_mem _ inTail)) more
      refine ⟨?_,?_⟩
      · intro q member
        rcases List.mem_cons.mp member with rfl | rest
        · exact known _ List.mem_cons_self
        · exact valid _ rest
      · cases h : fromRow r with
        | none => simpa [h] using perm
        | some v =>
          simp only [List.filterMap_cons_some h]
          exact (perm.cons v).trans (.swap _ _ _)

private theorem pushed (s out : Cursor w) (ok : s.advance (.enter t site node rec args) = some out) :
    project out = (t,node) :: project s := by
  unfold Cursor.advance at ok
  split at ok <;> try contradiction
  obtain ⟨top,stack,_,opening,kind,_,_,_⟩ := EndgameSearch.CacheHistory.NativeTrace.checked_entry s out ok
  simp [project,stack,opening,kind]

private theorem popped (s out : Cursor w) (ok : s.advance (.leave t opening node stamp) = some out) :
    project out = (project s).tail := by
  unfold Cursor.advance at ok
  split at ok <;> try contradiction
  unfold Cursor.leave at ok
  cases stack : s.stack <;> simp only [stack] at ok
  · contradiction
  · repeat' first | split at ok | contradiction
    all_goals cases ok; simp [project,stack]

private theorem key_matches {p : Pool} {w : Worker p} (rest : List (Activation w)) :
    (match rest with
      | caller :: _ => if caller.node == .pvs then frameKey (poolPlan p.size).length w.ordinal caller.first else 0
      | [] => 0) = parentKey (poolPlan p.size).length w.ordinal (rest.map (fun a => (a.first,a.node))) := by
  cases rest with
  | nil => rfl
  | cons a rest => cases eq : a.node <;> simp [parentKey,eq]

private theorem prior_entry (known : FromEntry size w inputs parents r) (other : opening ≠ r.opening) :
    FromEntry size w inputs ((opening,parent) :: parents) r := by
  exact ⟨by simpa only [List.lookup_cons,beq_eq_false_iff_ne.mpr (Ne.symm other)] using known.1,known.2⟩

private theorem next_rows {p : Pool} {w : Worker p} (s out : State w) (walk : Walk) (event : Event)
    (stack : walk.stack = project s.cursor)
    (known : ∀ r ∈ s.rows, FromEntry (poolPlan p.size).length w.ordinal inputs walk.parents r) (ok : s.advance inputs event = some out) :
    let next := walk.advance (poolPlan p.size).length w.ordinal inputs event
    next.stack = project out.cursor ∧ (∀ r ∈ out.rows, FromEntry (poolPlan p.size).length w.ordinal inputs next.parents r) ∧
    (out.rows.filterMap fromRow).Perm ((fromEvent (poolPlan p.size).length w.ordinal inputs walk event).toList ++ s.rows.filterMap fromRow) := by
  cases event with
  | entry t site node rec args =>
    simp only [State.advance] at ok
    split at ok <;> try contradiction
    obtain ⟨c,edge,tail⟩ := Option.bind_eq_some_iff.mp ok
    have agrees := (pushed s.cursor c edge).symm
    split at tail
    · obtain ⟨_,_,tail⟩ := Option.bind_eq_some_iff.mp tail
      split at tail <;> try contradiction
      cases tail
      exact ⟨by simpa only [Walk.advance,stack] using agrees,known,.refl _⟩
    · cases tail; exact ⟨by simpa only [Walk.advance,stack] using agrees,known,.refl _⟩
  | scored t score =>
    simp only [State.advance] at ok
    split at ok <;> try contradiction
    rename_i top rest shape
    split at ok <;> try contradiction
    rename_i valid
    obtain ⟨a,args,tail⟩ := Option.bind_eq_some_iff.mp ok
    have isPvs : top.node = .pvs := by simpa using (Bool.or_eq_false_iff.mp (Bool.eq_false_iff.mpr valid)).2
    have physical : walk.stack = (top.first,.pvs) :: rest.map (fun a => (a.first,a.node)) := by
      simp [stack,project,shape,isPvs]
    split at tail
    · rename_i other
      cases tail
      simp only [Walk.advance,physical,args,Option.any_some,fromEvent,Option.toList_none,List.nil_append]
      simp_all [project]
    · rename_i exact_
      have isExact : a.exact = true := by simpa using exact_
      split at tail <;> try contradiction
      rename_i unused
      obtain ⟨r,made,ret⟩ := Option.bind_eq_some_iff.mp tail
      cases ret
      simp only [Walk.advance,physical,args,Option.any_some,isExact,ite_true,fromEvent,Option.toList_none,List.nil_append]
      refine ⟨by simp [project,shape,isPvs],?_,?_⟩
      · intro q mem
        rcases List.mem_cons.mp mem with rfl | old
        · rw [← key_matches rest]
          exact (created args isExact made).1
        · apply prior_entry (known q old)
          intro eq
          exact unused (List.any_eq_true.mpr ⟨q,old,by simp [eq]⟩)
      · simp [(created (parents := walk.parents) args isExact made).2]
  | leave t opening node stamp normal =>
    simp only [State.advance] at ok
    obtain ⟨c,edge,tail⟩ := Option.bind_eq_some_iff.mp ok
    have popped_ : walk.stack.tail = project c := stack ▸ (popped s.cursor c edge).symm
    split at tail
    · rename_i pvs
      subst node
      obtain ⟨a,args,tail⟩ := Option.bind_eq_some_iff.mp tail
      cases normal with
      | none => cases tail; exact ⟨popped_,known,.refl _⟩
      | some score =>
        dsimp only at tail
        split at tail
        · rename_i exact_
          obtain ⟨rows,finish,ret⟩ := Option.bind_eq_some_iff.mp tail
          cases ret
          exact ⟨popped_,by simpa [Walk.advance,fromEvent,args,exact_] using finish_rows s.rows known args finish⟩
        · rename_i other
          cases tail
          exact ⟨popped_,known,by simp [fromEvent,args,other]⟩
    · rename_i other
      split at tail <;> try contradiction
      cases tail
      exact ⟨popped_,known,by cases node <;> simp_all [fromEvent]⟩
  | discarded t opening raw =>
    simp only [State.advance] at ok
    obtain ⟨_,_,tail⟩ := Option.bind_eq_some_iff.mp ok
    split at tail <;> try contradiction
    obtain ⟨c,edge,ret⟩ := Option.bind_eq_some_iff.mp tail
    cases ret
    exact ⟨by simpa only [Walk.advance,stack] using (popped s.cursor c edge).symm,known,.refl _⟩

def collect (size w : Nat) (inputs : Nat → Option Args) (walk : Walk) : List Event → List Returned
  | [] => []
  | e :: rest => (fromEvent size w inputs walk e).toList ++ collect size w inputs (walk.advance size w inputs e) rest

/-- Every exact normal return contributes precisely one row with its caller,
    input operands and return word, including nested calls and incomplete prefixes. -/
theorem run_returns {p : Pool} (w : Worker p) (inputs : Nat → Option Args) (events : List Event)
    {out : State w} (ok : run w inputs events = some out) :
    (out.rows.filterMap fromRow).Perm (collect (poolPlan p.size).length w.ordinal inputs {} events) := by
  have loop (s : State w) (walk : Walk) (stack : walk.stack = project s.cursor)
      (known : ∀ r ∈ s.rows, FromEntry (poolPlan p.size).length w.ordinal inputs walk.parents r)
      (ran : events.foldlM (fun s e => State.advance s inputs e) s = some out) :
      (out.rows.filterMap fromRow).Perm ((collect (poolPlan p.size).length w.ordinal inputs walk events).reverse ++
        s.rows.filterMap fromRow) := by
    clear ok
    induction events generalizing s walk with
    | nil => cases Option.some.inj ran; simp [collect]
    | cons e es ih =>
      rw [List.foldlM_cons] at ran
      obtain ⟨next,act,more⟩ := Option.bind_eq_some_iff.mp ran
      obtain ⟨agrees,valid,delta⟩ := next_rows s next walk e stack known act
      have combined := (ih next _ agrees valid more).trans (delta.append_left _)
      cases h : fromEvent (poolPlan p.size).length w.ordinal inputs walk e <;> simpa [collect,h,List.append_assoc] using combined
  have parsed := loop _ {} rfl (by simp) ok
  exact (by simpa using parsed : (out.rows.filterMap fromRow).Perm
    (collect (poolPlan p.size).length w.ordinal inputs {} events).reverse).trans (List.reverse_perm _)

def localEvents (worker : Nat) (stream : List Record) : List Event :=
  stream.filterMap fun
    | .localCall n event => if worker = n then some event else none
    | .table _ => none

def poolReturns (p : Pool) (inputs : Nat → Nat → Option Args) (stream : List Record) : List Returned :=
  let count := (poolPlan p.size).length
  (List.range count).flatMap fun n => collect count n (inputs n) {} (localEvents n stream)

private theorem worker_returns (p : Pool) (inputs : Nat → Nat → Option Args) (stream : List Record)
    {out : (w : Worker p) × State w} (parsed : workerRun p inputs stream n = some out) :
    (out.2.rows.filterMap fromRow).Perm (collect (poolPlan p.size).length n (inputs n) {} (localEvents n stream)) := by
  unfold workerRun at parsed
  obtain ⟨w,boot,next⟩ := Option.bind_eq_some_iff.mp parsed
  obtain ⟨s,ran,eq⟩ := Option.bind_eq_some_iff.mp next
  cases eq
  have worker : w.ordinal = n := by
    unfold EndgameSearch.CacheHistory.NativeTrace.workerAt at boot
    split at boot <;> cases boot
    rfl
  exact worker ▸ run_returns w (inputs n) (localEvents n stream) ran

theorem capture_returns (p : Pool) (inputs : Nat → Nat → Option Args) (stream : List Record)
    {capture : Capture p} (parsed : decode p inputs stream = some capture) :
    (capture.rows.filterMap fromRow).Perm (poolReturns p inputs stream) := by
  obtain ⟨_,ws,workers,rows,_⟩ := decode_keeps parsed
  have combine (ids : List Nat) (states : List ((w : Worker p) × State w))
      (ran : ids.mapM (workerRun p inputs stream) = some states) :
      ((states.flatMap (fun s => s.2.rows)).filterMap fromRow).Perm
        (ids.flatMap fun n => collect (poolPlan p.size).length n (inputs n) {} (localEvents n stream)) := by
    induction ids generalizing states with
    | nil => cases Option.some.inj ran; simp
    | cons n rest ih =>
      rw [List.mapM_cons] at ran
      obtain ⟨s,one,more⟩ := Option.bind_eq_some_iff.mp ran
      obtain ⟨ss,rest,eq⟩ := Option.bind_eq_some_iff.mp more
      cases eq
      simpa only [List.flatMap_cons,List.filterMap_append] using
        (worker_returns p inputs stream one).append (ih ss rest)
  rw [rows]
  exact combine _ ws workers

/-- An exact PVS leave in the raw interleaving has a completed row with the
    entry registers and the word of this leave, even when other calls are nested. -/
theorem row_of_leave {p : Pool} {cap : Capture p} (parsed : decode p inputs stream = some cap)
    (event : Record.localCall worker (.leave finish opening .pvs stamp (some word)) ∈ stream)
    (args : inputs worker opening = some a) (exact_ : a.exact = true) :
    ∃ r ∈ cap.rows, r.frame.worker = worker ∧ r.opening = opening ∧ r.frame.returnStep = some finish ∧
      r.rootLine = a.rootLine ∧ fields r.frame = argsFields a word.raw.toInt := by
  let size := (poolPlan p.size).length
  have fits : worker < size := by
    unfold decode at parsed
    dsimp only at parsed
    split at parsed
    · contradiction
    next guard =>
      simp only [Bool.not_eq_true',Bool.not_eq_false] at guard
      exact of_decide_eq_true (List.all_eq_true.mp guard _ event)
  have local_ : Event.leave finish opening .pvs stamp (some word) ∈ localEvents worker stream :=
    List.mem_filterMap.mpr ⟨_,event,by simp⟩
  have find (es : List Event) (s : Walk) (mem : Event.leave finish opening .pvs stamp (some word) ∈ es) :
      ∃ parent, (⟨worker,opening,frameKey size worker opening,finish,parent,a.rootLine,argsFields a word.raw.toInt⟩ : Returned) ∈
        collect size worker (inputs worker) s es := by
    induction es generalizing s with
    | nil => contradiction
    | cons e tail ih =>
      rcases List.mem_cons.mp mem with rfl | later
      · exact ⟨(s.parents.lookup opening).getD 0,List.mem_append_left _ (by simp [fromEvent,args,exact_])⟩
      · obtain ⟨parent,found⟩ := ih (s.advance ..) later
        exact ⟨parent,List.mem_append_right _ found⟩
  obtain ⟨parent,atLeave⟩ := find _ {} local_
  have inside := (capture_returns p inputs stream parsed).mem_iff.mpr
    (List.mem_flatMap.mpr ⟨worker,List.mem_range.mpr fits,atLeave⟩)
  obtain ⟨r,mem,origin⟩ := List.mem_filterMap.mp inside
  obtain ⟨t,time,eq⟩ := Option.map_eq_some_iff.mp origin
  exact ⟨r,mem,congrArg Returned.worker eq,congrArg Returned.opening eq,time.trans (congrArg (some ∘ Returned.step) eq),
    congrArg Returned.rootLine eq,congrArg Returned.call eq⟩

private theorem collected_key (size w : Nat) (inputs : Nat → Option Args) (events : List Event)
    (walk : Walk) (inside : ret ∈ collect size w inputs walk events) :
    ret.id = frameKey size ret.worker ret.opening := by
  induction events generalizing walk with
  | nil => simp [collect] at inside
  | cons e es ih =>
    rcases List.mem_append.mp inside with now | later
    · have got : fromEvent size w inputs walk e = some ret := Option.mem_toList.mp now
      unfold fromEvent at got
      split at got <;> try contradiction
      obtain ⟨_,_,tail⟩ := Option.bind_eq_some_iff.mp got
      split at tail <;> cases tail
      rfl
    · exact ih _ later

def forOwner (f : FrameReturns.Frame) (ret : Returned) : Option ChildReturn :=
  if ret.rootLine.getD 0 = 0 ∧ ret.parent = f.id ∧ ret.worker = f.worker ∧ ret.step < f.scoreStep then
    some ret.call else none

/-- Select directly from the raw worker events. Grouping calls by owner and
    excluding incomplete, discarded or restricted-root calls gives exactly the
    records used by the owner lookup, in any interleaving of the workers. -/
theorem owner_returns (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (f : FrameReturns.Frame)
    (parsed : decode p inputs stream = some cap) (identify before seq) :
    (FrameReturns.OwnerCalls.observed (logOf mc fc cap.rows cap.table identify before seq) f).Perm
      ((poolReturns p inputs stream).filterMap (forOwner f)) := by
  have projectRows (rows : List Row) :
      FrameReturns.OwnerCalls.observed (logOf mc fc rows cap.table identify before seq) f =
        (rows.filterMap fromRow).filterMap (forOwner f) := by
    induction rows with
    | nil => rfl
    | cons r rest ih =>
      simp only [FrameReturns.OwnerCalls.observed,logOf] at ih ⊢
      cases stamp : r.frame.returnStep with
      | none =>
        by_cases eligible : whole r <;>
          simp [eligible,fromRow,stamp,FrameReturns.OwnerCalls.direct,ih]
      | some tick =>
        by_cases eligible : r.rootLine.getD 0 = 0 <;>
          by_cases own : r.frame.parent = f.id ∧ r.frame.worker = f.worker ∧ tick < f.scoreStep <;>
            simp [whole,fromRow,forOwner,stamp,FrameReturns.OwnerCalls.direct,eligible,own,ih]
  rw [projectRows]
  exact (capture_returns p inputs stream parsed).filterMap _

/-- The raw return instruction names its particular invocation and gives a causal
    predecessor of the owner, including when the owner used a null window or passed. -/
theorem direct_return (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (f : FrameReturns.Frame)
    (parsed : decode p inputs stream = some cap) (identify before seq)
    (arrived : ret ∈ poolReturns p inputs stream) (used : forOwner f ret = some word) :
    let log := logOf mc fc cap.rows cap.table identify before seq
    ∃ r ∈ cap.rows, r.opening = ret.opening ∧ fields r.frame = word ∧
      ⟨identify ⟨ret.worker,ret.step⟩,ConcurrentReplay.Action.returned
        (frameKey (poolPlan p.size).length ret.worker ret.opening) (r.frame.result mc fc)⟩ ∈ log.events ∧
      before (identify ⟨ret.worker,ret.step⟩) (identify f.use) := by
  obtain ⟨w,_,local_⟩ := List.mem_flatMap.mp arrived
  have key := collected_key _ w _ _ _ local_
  obtain ⟨r,member,raw⟩ := List.mem_filterMap.mp ((capture_returns p inputs stream parsed).mem_iff.mpr arrived)
  obtain ⟨t,closed,eq⟩ := Option.map_eq_some_iff.mp raw
  subst ret
  unfold forOwner at used
  split at used <;> try contradiction
  rename_i applies
  rcases applies with ⟨whole_,_,worker,time⟩
  cases used
  let log := logOf mc fc cap.rows cap.table identify before seq
  have included : r.frame ∈ log.frames :=
    List.mem_map.mpr ⟨r,List.mem_filter.mpr ⟨member,by simpa [whole] using whole_⟩,rfl⟩
  change r.frame.id = frameKey _ r.frame.worker r.opening at key
  refine ⟨r,member,rfl,rfl,?_,seq _ _ ⟨worker,time⟩⟩
  change ⟨log.identify ⟨r.frame.worker,t⟩,.returned (frameKey _ r.frame.worker r.opening) (r.frame.result mc fc)⟩ ∈ log.events
  rw [← key]
  exact log.returned r.frame included closed

theorem direct_used (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (f : FrameReturns.Frame)
    (parsed : decode p inputs stream = some cap) (identify before seq)
    (arrived : ret ∈ poolReturns p inputs stream) (used : forOwner f ret = some word) :
    let log := logOf mc fc cap.rows cap.table identify before seq
    ∃ b : PvsBoardTree.Position, b.val = word.board ∧ FrameReturns.Children log f .serial
      (PvsBoardTree.build mc fc b) word.pv word.alpha word.beta word.score := by
  obtain ⟨r,_,_,rfl,event,earlier⟩ := direct_return mc fc f parsed identify before seq arrived used
  exact ⟨r.frame.position,rfl,_,event,earlier,_,rfl⟩

end NeuralReversi.PvsParallel.NativeFrames.Returns

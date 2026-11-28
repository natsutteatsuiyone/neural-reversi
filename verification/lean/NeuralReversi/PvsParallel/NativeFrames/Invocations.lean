module
public import NeuralReversi.PvsParallel.NativeFrames.Returns
public import NeuralReversi.PvsParallel.NativeFrameCache

@[expose] public section

namespace NeuralReversi.PvsParallel.NativeFrames.Invocations

open EndgameSearch.CacheOwner (Node Resource poolPlan)
open EndgameSearch.CacheHistory.NativeTrace (Operands)
open Returns (Walk Stack Returned parentKey argsFields collect fromEvent forOwner)
open FrameReturns.OwnerCalls (ChildReturn)

/-- A normally returning PVS activation emits its score before the return. Other
    calls may intervene, for example while saving the score. A cancelled exit
    carries the Rust word but does not deliver a usable child score. -/
inductive Exit where
  | normal (word : ScoreRefresh.Score)
  | discarded (word : ScoreRefresh.Score)
  | leave (callee : Node) (boundary : Option Nat) (scored : Option ScoreRefresh.Score)

def Exit.node : Exit → Node
  | .normal _ | .discarded _ => .pvs
  | .leave callee .. => callee

def Exit.score : Exit → Option ScoreRefresh.Score
  | .normal word => some word
  | .discarded _ => none
  | .leave _ _ value => value

def Exit.event (atStep opening : Nat) : Exit → Event
  | .normal word => .leave atStep opening .pvs none (some word)
  | .discarded word => .discarded atStep opening word
  | .leave node boundary _ => .leave atStep opening node boundary none

def Exit.result (exit : Exit) (inputs : Nat → Option Args) (size worker opening finish caller : Nat) :
    Option Returned := do
  let .normal word := exit | none
  let args ← inputs opening
  if args.exact then
    return ⟨worker,opening,frameKey size worker opening,finish,caller,args.rootLine,argsFields args word.raw.toInt⟩
  else none

structure Entry where
  site : Nat
  received : Resource
  operands : Option Operands

/-- Consecutive native calls on one worker. Each call contains the activations
    before and after its score instruction, followed by the caller's next call. -/
inductive Calls where
  | nil
  | call (entry : Entry) (exit : Exit) (beforeScore afterScore next : Calls)

def Calls.count : Calls → Nat
  | .nil => 0
  | .call _ exit pre post next => 1 + pre.count + exit.score.toList.length + post.count + 1 + next.count

def Calls.events (clock : Nat → Nat) (i : Nat) : Calls → List Event
  | .nil => []
  | .call entry exit pre post next =>
    let mark := i+1+pre.count
    let afterMark := mark+exit.score.toList.length
    let finish := afterMark+post.count
    [.entry (clock i) entry.site exit.node entry.received entry.operands] ++ pre.events clock (i+1) ++
      (exit.score.map (Event.scored (clock mark))).toList ++ post.events clock afterMark ++
      [exit.event (clock finish) (clock i)] ++ next.events clock (finish+1)

/-- The native return destination is the caller on this worker's stack. -/
def Calls.returns (clock : Nat → Nat) (inputs : Nat → Option Args) (size worker caller i : Nat) : Calls → List Returned
  | .nil => []
  | .call _ exit pre post next =>
    let inner := if exit.node == .pvs then frameKey size worker (clock i) else 0
    let afterMark := i+1+pre.count+exit.score.toList.length
    let finish := afterMark+post.count
    pre.returns clock inputs size worker inner (i+1) ++ post.returns clock inputs size worker inner afterMark ++
      (exit.result inputs size worker (clock i) (clock finish) caller).toList ++
      next.returns clock inputs size worker caller (finish+1)

def walk (size worker : Nat) (inputs : Nat → Option Args) (s : Walk) (events : List Event) : Walk :=
  events.foldl (fun w => w.advance size worker inputs) s

private theorem walk_append : walk size w inputs s (xs ++ ys) = walk size w inputs (walk size w inputs s xs) ys :=
  List.foldl_append

private theorem collect_append (xs ys : List Event) (s : Walk) :
    collect size w inputs s (xs ++ ys) =
      collect size w inputs s xs ++ collect size w inputs (walk size w inputs s xs) ys := by
  induction xs generalizing s with
  | nil => rfl
  | cons e es ih => simp only [List.cons_append,collect,ih,List.append_assoc,walk,List.foldl_cons]

private theorem mark (exit : Exit) (clock : Nat → Nat) (i mark : Nat) (s : Walk)
    (stack : s.stack = (clock i,exit.node)::prior) :
    let events := (exit.score.map (Event.scored (clock mark))).toList
    let after := walk size w inputs s events
    after.stack = s.stack ∧ (∀ t, t < clock i → after.parents.lookup t = s.parents.lookup t) ∧
      (∀ value args, exit = .normal value → inputs (clock i) = some args → args.exact = true →
        after.parents.lookup (clock i) = some (parentKey size w prior)) ∧ collect size w inputs s events = [] := by
  cases exit with
  | normal value =>
    change s.stack = (clock i,.pvs)::prior at stack
    simp only [Exit.score,Option.map_some,Option.toList_some,walk,List.foldl_cons,List.foldl_nil,Walk.advance,
      stack,collect,fromEvent,List.append_nil]
    cases h : inputs (clock i) with
    | none => simp [stack]
    | some a =>
      cases ex : a.exact <;> simp [ex,stack]
      intro t earlier
      simp [List.lookup,show (t == clock i) = false from beq_eq_false_iff_ne.mpr (by omega)]
  | discarded => simp [Exit.score,walk,collect]
  | leave node boundary scored =>
    cases scored with
    | none => simp [Exit.score,walk,collect]
    | some v =>
      simp only [Exit.score,Option.map_some,Option.toList_some,walk,List.foldl_cons,List.foldl_nil,Walk.advance,
        stack,Exit.node,collect,fromEvent,List.append_nil]
      cases node <;> simp [stack,Exit.node]
      split <;> simp [stack,Exit.node]
      intro t earlier
      simp [List.lookup,show (t == clock i) = false from beq_eq_false_iff_ne.mpr (by omega)]

private theorem pop (exit : Exit) (s : Walk) (finish opening : Nat)
    (stack : s.stack = (opening,exit.node)::prior)
    (scored : ∀ value args, exit = .normal value → inputs opening = some args → args.exact = true →
      s.parents.lookup opening = some (parentKey size w prior)) :
    (s.advance size w inputs (exit.event finish opening)).stack = prior ∧
      (s.advance size w inputs (exit.event finish opening)).parents = s.parents ∧
      fromEvent size w inputs s (exit.event finish opening) = exit.result inputs size w opening finish (parentKey size w prior) := by
  cases exit with
  | normal v =>
    cases h : inputs opening with
    | none => simp [Exit.event,Exit.result,Walk.advance,fromEvent,stack,h]
    | some a =>
      cases ex : a.exact
      · simp [Exit.event,Exit.result,Walk.advance,fromEvent,stack,h,ex]
      · simp [Exit.event,Exit.result,Walk.advance,fromEvent,stack,h,ex,scored _ _ rfl h ex]
  | discarded => simp [Exit.event,Exit.result,Walk.advance,fromEvent,stack]
  | leave node stamp score => cases node <;> simp [Exit.event,Exit.result,Walk.advance,fromEvent,stack]

/-- All nested activations restore the caller's stack and preserve the origins
    of earlier scores. The chronological return list follows from call/return
    control flow without reading a checked row or matching two lists. -/
theorem collect_calls (c : Calls) (clock : Nat → Nat) (time : ∀ {a b}, a < b → clock a < clock b)
    (inputs : Nat → Option Args) (size w i : Nat) (s : Walk) :
    let ev := c.events clock i
    let out := walk size w inputs s ev
    out.stack = s.stack ∧ (∀ t, t < clock i → out.parents.lookup t = s.parents.lookup t) ∧
      collect size w inputs s ev = c.returns clock inputs size w (parentKey size w s.stack) i := by
  induction c generalizing i s with
  | nil => simp [Calls.events,Calls.returns,walk,collect]
  | call entry exit pre post next ihPre ihPost ihNext =>
    let pivot := i+1+pre.count
    let afterMark := pivot+exit.score.toList.length
    let finish := afterMark+post.count
    let op := Event.entry (clock i) entry.site exit.node entry.received entry.operands
    let s0 := s.advance size w inputs op
    let s1 := walk size w inputs s0 (pre.events clock (i+1))
    let marks := (exit.score.map (Event.scored (clock pivot))).toList
    let s2 := walk size w inputs s1 marks
    let s3 := walk size w inputs s2 (post.events clock afterMark)
    let endOp := exit.event (clock finish) (clock i)
    let s4 := s3.advance size w inputs endOp
    obtain ⟨preStack,preMap,preOut⟩ := ihPre (i+1) s0
    change s1.stack = (clock i,exit.node)::s.stack at preStack
    obtain ⟨markStack,markMap,scored,markOut⟩ :=
      mark (size:=size) (w:=w) (inputs:=inputs) exit clock i pivot s1 preStack
    change s2.stack = s1.stack at markStack
    have preTime := time (show i < i+1 by omega)
    have afterTime := time (show i < afterMark by dsimp [afterMark,pivot]; omega)
    have endTime := time (show i < finish+1 by dsimp [finish,afterMark,pivot]; omega)
    obtain ⟨postStack,postMap,postOut⟩ := ihPost afterMark s2
    have active : s3.stack = (clock i,exit.node)::s.stack := postStack.trans (markStack.trans preStack)
    obtain ⟨popStack,popMap,popOut⟩ := pop (size:=size) (w:=w) (inputs:=inputs) exit s3 (clock finish) (clock i)
      active (fun v a eq atEntry exact_ => (postMap _ afterTime).trans (scored v a eq atEntry exact_))
    obtain ⟨nextStack,nextMap,nextOut⟩ := ihNext (finish+1) s4
    have inner : parentKey size w ((clock i,exit.node)::s.stack) =
        if exit.node == .pvs then frameKey size w (clock i) else 0 := by cases exit.node <;> rfl
    change collect size w inputs s0 _ = pre.returns _ _ _ _ (parentKey size w ((clock i,exit.node)::s.stack)) _ at preOut
    rw [inner] at preOut
    rw [markStack,preStack,inner] at postOut
    change s4.stack = s.stack at popStack
    rw [popStack] at nextOut
    dsimp only [Calls.events,Calls.returns]
    simp only [collect_append,walk_append]
    refine ⟨nextStack.trans popStack,?_,?_⟩
    · intro t earlier
      exact (nextMap t (Nat.lt_trans earlier endTime)).trans ((congrArg (List.lookup t) popMap).trans
        ((postMap t (Nat.lt_trans earlier afterTime)).trans ((markMap t earlier).trans (preMap t (Nat.lt_trans earlier preTime)))))
    · have single (state : Walk) (e : Event) : collect size w inputs state [e] = (fromEvent size w inputs state e).toList := by simp [collect]
      simp only [single]
      change collect size w inputs s0 (pre.events clock (i+1)) ++ collect size w inputs s1 marks ++
        collect size w inputs s2 (post.events clock afterMark) ++ (fromEvent size w inputs s3 endOp).toList ++
        collect size w inputs s4 (next.events clock (finish+1)) = _
      change collect size w inputs s1 marks = [] at markOut
      change fromEvent size w inputs s3 endOp = _ at popOut
      rw [preOut,markOut,postOut,popOut,nextOut,List.append_nil]

def Exit.direct (exit : Exit) (args : Option Args) : Option ChildReturn := do
  let .normal value := exit | none
  let a ← args
  if a.exact && decide (a.rootLine.getD 0 = 0) then some (argsFields a value.raw.toInt) else none

/-- Only the outermost returning PVS invocations deliver words to this caller. -/
def Calls.direct (clock : Nat → Nat) (inputs : Nat → Option Args) (i : Nat) : Calls → List ChildReturn
  | .nil => []
  | .call _ exit pre post next =>
    (exit.direct (inputs (clock i))).toList ++
      next.direct clock inputs (i+1+pre.count+exit.score.toList.length+post.count+1)

private theorem single_owner (f : FrameReturns.Frame) (exit : Exit) (earlier : finish < f.scoreStep) :
    ((exit.result inputs size f.worker opening finish caller).toList.filterMap (forOwner f)) =
      if caller = f.id then (exit.direct (inputs opening)).toList else [] := by
  cases exit <;> try simp [Exit.result,Exit.direct]
  rename_i word
  cases h : inputs opening with
  | none => simp
  | some a =>
    by_cases eligible : a.rootLine.getD 0 = 0 <;> by_cases local_ : caller = f.id <;>
      cases ex : a.exact <;> simp [forOwner,eligible,local_,ex,earlier]

theorem direct_calls (c : Calls) (clock : Nat → Nat) (time : ∀ {a b}, a < b → clock a < clock b)
    (inputs : Nat → Option Args) (size i opening caller : Nat) (f : FrameReturns.Frame)
    (worker : f.worker < size) (id : f.id = frameKey size f.worker (clock opening)) (active : opening < i)
    (before : clock (i+c.count) ≤ f.scoreStep) :
    ((c.returns clock inputs size f.worker caller i).filterMap (forOwner f)) =
      if caller = f.id then c.direct clock inputs i else [] := by
  induction c generalizing i caller with
  | nil => simp [Calls.returns,Calls.direct]
  | call entry exit pre post next ihPre ihPost ihNext =>
    let afterMark := i+1+pre.count+exit.score.toList.length
    let finish := afterMark+post.count
    have last : clock (finish+1+next.count) ≤ f.scoreStep := by
      have add : finish+1+next.count = i+(Calls.call entry exit pre post next).count := by
        dsimp [Calls.count,finish,afterMark]; omega
      exact add ▸ before
    have finishTime : clock finish < f.scoreStep :=
      Nat.lt_of_lt_of_le (time (by omega : finish < finish+1+next.count)) last
    have leTime {a b} (le : a ≤ b) : clock a ≤ clock b := by
      rcases Nat.lt_or_eq_of_le le with lt | rfl
      · exact Nat.le_of_lt (time lt)
      · exact Nat.le_refl _
    let inner := if exit.node == .pvs then frameKey size f.worker (clock i) else 0
    have other : inner ≠ f.id := by
      dsimp only [inner]
      split
      · intro eq
        exact (Nat.ne_of_gt (time active)) (frameKey_unique worker worker (eq.trans id)).2
      · rw [id]; simp [frameKey]
    have preEnd : clock (i+1+pre.count) ≤ f.scoreStep :=
      Nat.le_trans (leTime (by dsimp [finish,afterMark]; omega : i+1+pre.count ≤ finish)) (Nat.le_of_lt finishTime)
    have left := ihPre (i+1) inner (by omega) preEnd
    have right := ihPost afterMark inner (by dsimp [afterMark]; omega) (Nat.le_of_lt finishTime)
    have tail := ihNext (finish+1) caller (by dsimp [finish,afterMark]; omega) last
    simp only [other,ite_false] at left right
    simp only [Calls.returns,Calls.direct,List.filterMap_append]
    change _ ++ _ ++ ((exit.result inputs size f.worker (clock i) (clock finish) caller).toList.filterMap (forOwner f)) ++ _ = _
    rw [left,right,tail,single_owner f exit finishTime]
    split <;> simp [finish,afterMark]

private def Bounded (size w opening : Nat) (s : Walk) : Prop :=
  (∀ pair ∈ s.stack, pair.1 < opening) ∧
  ∀ t, (s.parents.lookup t).getD 0 < frameKey size w opening

private theorem parent_lt (positive : 0 < size) (old : ∀ pair ∈ stack, pair.1 < opening) :
    parentKey size w stack < frameKey size w opening := by
  cases stack with
  | nil => simp [parentKey,frameKey]
  | cons p tail =>
    obtain ⟨t,node⟩ := p
    have less := old (t,node) List.mem_cons_self
    have mul := Nat.mul_lt_mul_of_pos_right less positive
    cases node <;> simp [parentKey,frameKey] <;> omega

private theorem bounded_step (positive : 0 < size) (s : Walk) (bounded : Bounded size w opening s)
    (ev : Event) (early : ∀ t site node rec a, ev = .entry t site node rec a → t < opening) :
    Bounded size w opening (s.advance size w inputs ev) := by
  obtain ⟨stack,parents⟩ := bounded
  cases ev with
  | entry t site node rec a =>
    refine ⟨?_,parents⟩
    intro p mem
    rcases List.mem_cons.mp mem with rfl | below
    · exact early _ _ _ _ _ rfl
    · exact stack _ below
  | leave | discarded => exact ⟨fun p mem => stack p (List.mem_of_mem_tail mem),parents⟩
  | scored t word =>
    cases shape : s.stack with
    | nil => simpa [Walk.advance,shape,Bounded] using (show Bounded size w opening s from ⟨stack,parents⟩)
    | cons p rest =>
      obtain ⟨first,node⟩ := p
      cases node <;> simp only [Walk.advance,shape] <;> try exact ⟨stack,parents⟩
      split
      · refine ⟨shape ▸ stack,?_⟩
        intro k
        by_cases same : k = first
        · subst k; simp only [List.lookup_cons_self,Option.getD_some]
          exact parent_lt positive (fun p mem => stack p (shape ▸ List.mem_cons_of_mem _ mem))
        · simpa [List.lookup,beq_eq_false_iff_ne.mpr same] using parents k
      · exact ⟨stack,parents⟩

private def early (opening : Nat) (ev : Event) := match ev with
  | .entry t .. => t < opening
  | _ => True

private theorem prefix_silent (events : List Event) (s : Walk) (positive : 0 < size)
    (f : FrameReturns.Frame) (id : f.id = frameKey size f.worker opening)
    (bounded : Bounded size f.worker opening s) (prior : ∀ e ∈ events, early opening e) :
    (collect size f.worker inputs s events).filterMap (forOwner f) = [] := by
  induction events generalizing s with
  | nil => rfl
  | cons e es ih =>
    have first := prior e List.mem_cons_self
    have preserve := bounded_step (inputs:=inputs) positive s bounded e (fun t site node rec a eq => by rw [eq] at first; exact first)
    rw [collect,List.filterMap_append,ih _ preserve (fun v mem => prior v (List.mem_cons_of_mem _ mem)),List.append_nil]
    cases e <;> try rfl
    rename_i t opening node stamp result
    cases node <;> cases result <;> try rfl
    have other : (s.parents.lookup opening).getD 0 ≠ f.id := id ▸ Nat.ne_of_lt (bounded.2 opening)
    cases got : inputs opening with
    | none => simp [fromEvent,got]
    | some a => cases ex : a.exact <;> simp [fromEvent,forOwner,got,ex,other]

private theorem suffix_silent (events : List Event) (s : Walk) (f : FrameReturns.Frame)
    (later : ∀ t opening stamp val, Event.leave t opening .pvs stamp (some val) ∈ events → f.scoreStep ≤ t) :
    (collect size f.worker inputs s events).filterMap (forOwner f) = [] := by
  induction events generalizing s with
  | nil => rfl
  | cons e es ih =>
    rw [collect,List.filterMap_append,ih _ (fun _ _ _ _ mem => later _ _ _ _ (List.mem_cons_of_mem _ mem)),List.append_nil]
    cases e <;> try rfl
    rename_i t opening node stamp result
    cases node <;> cases result <;> try rfl
    have after := later t opening stamp _ List.mem_cons_self
    cases got : inputs opening with
    | none => simp [fromEvent,got]
    | some a => cases ex : a.exact <;> simp [fromEvent,forOwner,got,ex,Nat.not_lt.mpr after]

private theorem foreign (s : Walk) (events : List Event) (f : FrameReturns.Frame) (other : w ≠ f.worker) :
    (collect size w inputs s events).filterMap (forOwner f) = [] := by
  induction events generalizing s with
  | nil => rfl
  | cons e es ih =>
    rw [collect,List.filterMap_append,ih,List.append_nil]
    cases e <;> try rfl
    rename_i t opening node stamp value
    cases node <;> cases value <;> try rfl
    cases src : inputs opening with
    | none => simp [fromEvent,src]
    | some a => cases ex : a.exact <;> simp [fromEvent,src,ex,forOwner,other]

private theorem pool_worker (p : EndgameSearch.CacheHistory.Pool) (inputs : Nat → Nat → Option Args)
    (stream : List Record) (f : FrameReturns.Frame) (worker : f.worker < (poolPlan p.size).length) :
    (Returns.poolReturns p inputs stream).filterMap (forOwner f) =
      (collect (poolPlan p.size).length f.worker (inputs f.worker) {} (Returns.localEvents f.worker stream)).filterMap (forOwner f) := by
  let rows := fun n => collect (poolPlan p.size).length n (inputs n) {} (Returns.localEvents n stream)
  have select (ids : List Nat) (distinct : ids.Nodup) :
      (ids.flatMap rows).filterMap (forOwner f) =
        if f.worker ∈ ids then (rows f.worker).filterMap (forOwner f) else [] := by
    induction ids with
    | nil => simp
    | cons n ns ih =>
      obtain ⟨absent,unique⟩ := List.nodup_cons.mp distinct
      rw [List.flatMap_cons,List.filterMap_append,ih unique]
      by_cases same : n = f.worker
      · subst n; simp [absent]
      · have skip := foreign (inputs:=inputs n) (size:=(poolPlan p.size).length) {} (Returns.localEvents n stream) f same
        change (rows n).filterMap (forOwner f) = [] at skip
        simp [skip,Ne.symm same]
  simpa [Returns.poolReturns,rows,List.mem_range.mpr worker] using select _ (List.nodup_range (n:=(poolPlan p.size).length))

/-- The interval of a single PVS activation on its own worker, starting with
    entry and ending at the scoring instruction. The rest of the pool and
    the suffix after scoring can still be running. -/
structure Scope (p : EndgameSearch.CacheHistory.Pool) (stream : List Record) (f : FrameReturns.Frame) where
  clock : Nat → Nat
  time : ∀ {a b}, a < b → clock a < clock b
  opening : Nat
  entry : Entry
  body : Calls
  prefixEvents : List Event
  suffixEvents : List Event
  word : ScoreRefresh.Score
  worker : f.worker < (poolPlan p.size).length
  id : f.id = frameKey (poolPlan p.size).length f.worker (clock opening)
  scoreStep : clock (opening+1+body.count) = f.scoreStep
  window : Returns.localEvents f.worker stream = prefixEvents ++
    [.entry (clock opening) entry.site .pvs entry.received entry.operands] ++ body.events clock (opening+1) ++
    [.scored f.scoreStep word] ++ suffixEvents

theorem Scope.records (ctx : Scope p stream f) (parsed : decode p inputs stream = some cap) :
    (Returns.poolReturns p inputs stream).filterMap (forOwner f) =
      ctx.body.direct ctx.clock (inputs f.worker) (ctx.opening+1) := by
  let size := (poolPlan p.size).length
  let get := inputs f.worker
  let openEv := Event.entry (ctx.clock ctx.opening) ctx.entry.site .pvs ctx.entry.received ctx.entry.operands
  let scoreEv := Event.scored f.scoreStep ctx.word
  let before := walk size f.worker get {} ctx.prefixEvents
  let pushed := before.advance size f.worker get openEv
  let events := ctx.body.events ctx.clock (ctx.opening+1)
  let finished := walk size f.worker get pushed events
  have sorted := NativeFrameCache.worker_chronology p inputs stream f.worker ctx.worker parsed
  change (Returns.localEvents f.worker stream).Pairwise _ at sorted
  rw [ctx.window] at sorted
  have parts := List.pairwise_append.mp sorted
  have beforeScore := (List.pairwise_append.mp parts.1).1
  have beforeBody := (List.pairwise_append.mp beforeScore).1
  have beforeEntry := (List.pairwise_append.mp beforeBody).2.2
  have old : ∀ e ∈ ctx.prefixEvents, early (ctx.clock ctx.opening) e := by
    intro e mem
    have gap := beforeEntry e mem openEv List.mem_cons_self
    cases e <;> first | exact gap | trivial
  have fresh : Bounded size f.worker (ctx.clock ctx.opening) {} := by simp [Bounded,frameKey]
  have left := prefix_silent (size:=size) (inputs:=get) ctx.prefixEvents {} (Nat.lt_of_le_of_lt (Nat.zero_le _) ctx.worker)
    f ctx.id fresh old
  have right := suffix_silent (inputs:=get) (size:=size) ctx.suffixEvents
    (finished.advance size f.worker get scoreEv) f (by
      intro t op stamp v mem
      have ord := parts.2.2 scoreEv (List.mem_append_right _ List.mem_cons_self) (.leave t op .pvs stamp (some v)) mem
      exact Nat.le_of_lt ord)
  have direct := direct_calls ctx.body ctx.clock ctx.time get size (ctx.opening+1) ctx.opening f.id f
    ctx.worker ctx.id (Nat.lt_succ_self _) (Nat.le_of_eq ctx.scoreStep)
  have nested := (collect_calls ctx.body ctx.clock ctx.time get size f.worker (ctx.opening+1) pushed).2.2
  have parent : parentKey size f.worker pushed.stack = f.id := ctx.id.symm
  rw [parent] at nested
  rw [pool_worker p inputs stream f ctx.worker,ctx.window]
  simp only [List.append_assoc]
  rw [collect_append,List.filterMap_append,left,List.nil_append]
  change (collect size f.worker get pushed (events ++ scoreEv :: ctx.suffixEvents)).filterMap (forOwner f) = _
  rw [collect_append,List.filterMap_append]
  change (collect size f.worker get pushed events).filterMap (forOwner f) ++
    (collect size f.worker get (finished.advance size f.worker get scoreEv) ctx.suffixEvents).filterMap (forOwner f) = _
  rw [right,List.append_nil,nested]
  simpa only [ite_true] using direct

end NeuralReversi.PvsParallel.NativeFrames.Invocations

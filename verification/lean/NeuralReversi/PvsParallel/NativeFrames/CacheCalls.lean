import NeuralReversi.PvsParallel.NativeFrames.OwnerSites.Body

namespace NeuralReversi.PvsParallel.NativeFrames.CacheCalls

open EndgameSearch.CacheOwner (Node Flow Edge Thread Resource transfers)
open EndgameSearch.CacheHistory (Pool Input Arm Visit)
open EndgameSearch.CacheHistory.NativeTrace (Worker Cursor Activation Operands Raw expectedSites)
open OwnerSites (Caller Segment advance inCache)
open Invocations (Calls)

variable {p : Pool} {w : Worker p}

/-- Calls of the worker-local cache MIR. Each instruction selects the callee and
    borrow from the compiler's transfer table. The order and depth are unrestricted. -/
inductive Forest : Node → Type where
  | nil : Forest node
  | call (site : Nat) (child : Node) (flow : Flow) (edge : Edge.mk node site child flow ∈ transfers)
      (body : Forest child) (next : Forest node) : Forest node

def Forest.calls (thread : Thread) : Forest node → Calls
  | .nil => .nil
  | .call site child _ _ body next => .call ⟨site,child.kind.resource thread,none⟩ (.leave child none none)
      (body.calls thread) .nil (next.calls thread)

def Forest.direct : Forest node → List (Nat × Node)
  | .nil => []
  | .call site child _ _ _ next => (site,child)::next.direct

def thread (w : Worker p) : Thread := ⟨p.base+3*w.ordinal+2,p.base+3*w.ordinal,p.base+3*w.ordinal+1⟩

private theorem below (e : Edge) (member : e ∈ transfers) (borrowed : inCache e.caller = true) :
    inCache e.callee = true ∧ e.callee ≠ .pvs ∧ e.callee ≠ .boundary := by
  have table : transfers.all (fun e => !inCache e.caller ||
    (inCache e.callee && e.callee != .pvs && e.callee != .boundary)) = true := by decide
  simpa [borrowed,and_assoc] using List.all_eq_true.mp table e member

private theorem nondesc {clock : Nat → Nat} (time : ∀ {a b : Nat}, a < b → clock a < clock b) {i j : Nat} (le : i ≤ j) :
    clock i ≤ clock j := by
  rcases Nat.eq_or_lt_of_le le with rfl | lt
  · exact Nat.le_refl _
  · exact Nat.le_of_lt (time lt)

/-- No nested cache routine changes search-frame rows or creates a new boundary
    visit. The boundary sees exactly its own direct call sites in reverse order. -/
theorem Forest.run (f : Forest node) (clock : Nat → Nat) (time : ∀ {a b}, a < b → clock a < clock b)
    (i : Nat) (s : State w) (borrowed : inCache node = true) (parent : s.cursor.parent.node = node)
    (rec : s.cursor.parent.received = node.kind.resource (thread w))
    (ready : s.cursor.clock ≤ clock i)
    (empty : ∀ j, i ≤ j → j < i+(f.calls (thread w)).count → inputs (clock j) = none) :
    ∃ out : State w, advance inputs s ((f.calls (thread w)).events clock i) = some out ∧
      out.cursor.stack = s.cursor.stack ∧ out.cursor.clock ≤ clock (i+(f.calls (thread w)).count) ∧
      out.rows = s.rows ∧ out.cursor.finished = s.cursor.finished ∧ out.cursor.boundarySites =
        (if node = .boundary then f.direct.reverse ++ s.cursor.boundarySites else s.cursor.boundarySites) := by
  induction f generalizing i s with
  | nil =>
    exact ⟨s,rfl,rfl,by simpa [Forest.calls,Calls.count] using ready,rfl,rfl,by simp [Forest.direct]⟩
  | @call node site child flow edge body next ihBody ihNext =>
    let e : Edge := ⟨node,site,child,flow⟩
    let inner := body.calls (thread w)
    let later := next.calls (thread w)
    let finish := i+1+inner.count
    have size : i+((Forest.call site child flow edge body next).calls (thread w)).count = finish+1+later.count := by
      change i+(1+inner.count+0+0+1+later.count) = _
      dsimp [finish]; omega
    rw [size] at empty ⊢
    obtain ⟨cacheChild,notPvs,notBoundary⟩ := below e edge borrowed
    have route := EndgameSearch.CacheOwner.flows edge (thread w) p.outside
    obtain ⟨entry,push,tick,rows,top,stack,first,kind,receiver,payload,finished,sites⟩ :=
      OwnerSites.enter_call (inputs:=inputs) s e edge notBoundary parent (rec ▸ route) ready
        (by rw [empty i (Nat.le_refl _) (by dsimp [finish]; omega)]; simp [notPvs])
    have current : entry.cursor.parent = top := by simp [Cursor.parent,stack]
    obtain ⟨last,runBody,bodyStack,bodyTick,bodyRows,bodyFinished,bodySites⟩ := ihBody (i+1) entry
      cacheChild (current ▸ kind) (current ▸ receiver) (by rw [tick]; exact time (Nat.lt_succ_self i))
      (fun j h u => empty j (by omega) (by change j < finish at u; omega))
    have atTop := bodyStack.trans stack
    have pop : last.advance inputs (.leave (clock finish) (clock i) child none none) =
        some ⟨{last.cursor with clock := clock finish+1,stack := s.cursor.stack},last.rows⟩ := by
      change last.cursor.clock ≤ clock finish at bodyTick
      simp [State.advance,Cursor.advance,Raw.step,bodyTick,Cursor.leave,atTop,first,kind,payload,notPvs,e]
    let popped : State w := ⟨{last.cursor with clock := clock finish+1,stack := s.cursor.stack},last.rows⟩
    have atParent : popped.cursor.parent = s.cursor.parent := rfl
    obtain ⟨out,runNext,nextStack,nextTick,nextRows,nextFinished,nextSites⟩ := ihNext (finish+1) popped
      borrowed (atParent ▸ parent) (atParent ▸ rec) (time (Nat.lt_succ_self finish))
      (fun j h u => empty j (by dsimp [finish] at h; omega) u)
    refine ⟨out,?_,nextStack,nextTick,nextRows.trans (bodyRows.trans rows),
      nextFinished.trans (bodyFinished.trans finished),?_⟩
    · change s.advance inputs (.entry (clock i) site child (child.kind.resource (thread w)) none) = some entry at push
      change advance inputs entry (inner.events clock (i+1)) = some last at runBody
      change advance inputs popped (later.events clock (finish+1)) = some out at runNext
      simp only [Forest.calls,Calls.events,Invocations.Exit.node,Invocations.Exit.score,Option.toList_none,List.length_nil,
        Invocations.Exit.event,Option.map_none,Calls.count,Nat.add_zero,List.append_nil,
        advance,List.foldlM_append,List.foldlM_cons,List.foldlM_nil,push,pure,bind,Option.bind] at runBody runNext ⊢
      dsimp only [finish,inner] at pop
      rw [runBody]; dsimp only; rw [pop]; exact runNext
    · change out.cursor.boundarySites = (if node = .boundary then next.direct.reverse ++ last.cursor.boundarySites
        else last.cursor.boundarySites) at nextSites
      change child ≠ Node.boundary at notBoundary
      simp only [ite_eq_right notBoundary] at bodySites
      rw [nextSites,bodySites,sites]
      by_cases h : node = .boundary <;> simp [Forest.direct,e,h,List.append_assoc]

/-- The TT shortcut never borrows the local cache. The fixed solvers call the
    getter, and the two recursive arms then borrow the corresponding table. -/
def plan (deep : Forest .deep) (shallow : Forest .shallow) : Arm → Forest .boundary
  | .failed | .table _ => .nil
  | .fixed _ => .call 0 .access .forward (by decide) .nil .nil
  | .shallow => .call 0 .access .forward (by decide) .nil
      (.call 3 .shallow .borrowShallow (by decide) shallow .nil)
  | .deep true => .call 0 .access .forward (by decide) .nil
      (.call 1 .deep .borrowPair (by decide) deep .nil)
  | .deep false => .call 0 .access .forward (by decide) .nil
      (.call 2 .deep .borrowPair (by decide) deep .nil)

theorem sites (deep shallow) (arm : Arm) (returned : arm ≠ .failed) :
    expectedSites arm = some (plan deep shallow arm).direct := by
  cases arm with
  | failed => contradiction
  | deep poll => cases poll <;> rfl
  | fixed | table | shallow => rfl

def boundaryArgs (w : Worker p) (parentOpening : Nat) (input : Input) : Operands :=
  ⟨frameKey (EndgameSearch.CacheOwner.poolPlan p.size).length w.ordinal parentOpening,input⟩

def boundary (w : Worker p) (site parentOpening : Nat) (input : Input) (score : Option Nat)
    (deep : Forest .deep) (shallow : Forest .shallow) : Calls :=
  .call ⟨site,.thread (thread w),some (boundaryArgs w parentOpening input)⟩ (.leave .boundary score none)
    ((plan deep shallow input.arm).calls (thread w)) .nil .nil

def finishIndex (w : Worker p) (input : Input) (deep shallow) (i : Nat) :=
  i+1+((plan deep shallow input.arm).calls (thread w)).count

/-- The caller identity comes from the active PVS opening. A completed cache
    call creates its own visit, even when its score will be discarded. -/
theorem boundary_visit (w : Worker p) (site owner : Nat)
    (edge : Edge.mk .pvs site .boundary .forward ∈ transfers)
    (input : Input) (returned : input.arm ≠ .failed) (score : Option Nat) (deep shallow)
    (clock : Nat → Nat) (time : ∀ {a b}, a < b → clock a < clock b) (i : Nat)
    (reported : ∀ t, score = some t → clock (finishIndex w input deep shallow i) ≤ t)
    (empty : ∀ j, i ≤ j → j < finishIndex w input deep shallow i → inputs (clock j) = none)
    (s : State w) (atCaller : Caller s.cursor .pvs (.thread (thread w)) owner) (ready : s.cursor.clock ≤ clock i) :
    ∃ (out : State w) (v : Visit p), advance inputs s ((boundary w site owner input score deep shallow).events clock i) = some out ∧
      out.cursor.stack = s.cursor.stack ∧ out.cursor.clock = clock (finishIndex w input deep shallow i)+1 ∧
      out.rows = s.rows ∧ out.cursor.finished = v::s.cursor.finished ∧ v.worker = w.ordinal ∧
      v.first = clock i ∧ v.last = clock (finishIndex w input deep shallow i) ∧ v.scoredAt = score ∧
      v.frame = (boundaryArgs w owner input).frame ∧ v.input = input := by
  let body := plan deep shallow input.arm
  let stop := finishIndex w input deep shallow i
  let args := boundaryArgs w owner input
  let receiver := Resource.thread (thread w)
  let op : Edge := ⟨.pvs,site,.boundary,.forward⟩
  have le : i < stop := by dsimp [stop,finishIndex]; omega
  have vacant := empty i (Nat.le_refl _) le
  have free := atCaller.free rfl
  obtain ⟨entry,routed,entryVisits,entrySites⟩ := OwnerSites.enter_routed (stamp:=clock i) (args:=some args)
    s.cursor op edge atCaller.kind (congrArg some atCaller.resource) (by simp [op,free])
  obtain ⟨top,stack,_,first,node,rec,payload,_⟩ := EndgameSearch.CacheHistory.NativeTrace.checked_entry _ _ routed
  let entered : State w := ⟨entry,s.rows⟩
  have pushed : s.advance inputs (.entry (clock i) site .boundary receiver (some args)) = some entered := by
    simp [State.advance,Cursor.advance,Raw.step,ready,vacant,op,atCaller.kind,atCaller.first,args,boundaryArgs,
      receiver,entered] at routed ⊢
    rw [routed]; rfl
  have tick := EndgameSearch.CacheHistory.NativeTrace.advance_clock s.cursor
    (.enter (clock i) site .boundary receiver (some args)) (by simpa [Cursor.advance,Raw.step,ready] using routed)
  have head : entry.parent = top := by simp [Cursor.parent,stack]
  obtain ⟨last,ran,bodyStack,bodyTime,rows,visits,actualSites⟩ := body.run clock time (i+1) entered rfl
    (head ▸ node) (head ▸ rec) (by change entry.clock ≤ _; rw [tick.2]; exact time (Nat.lt_succ_self i))
    (fun j l u => empty j (by omega) u)
  have kept : last.cursor.stack = top::s.cursor.stack := bodyStack.trans stack
  have direct : last.cursor.boundarySites = body.direct.reverse := by
    simpa only [entered,entrySites,op,ite_true,List.append_nil] using actualSites
  have expected : expectedSites args.input.arm = some last.cursor.boundarySites.reverse := by
    rw [direct,List.reverse_reverse]
    exact sites deep shallow input.arm returned
  have ordered := nondesc time (Nat.le_of_lt le)
  have okScore : (score.map (fun k => decide (clock stop ≤ k))).getD true = true := by
    cases h : score with
    | none => rfl
    | some t => simpa using reported t h
  obtain ⟨next,closed⟩ : ∃ next, last.cursor.leave (clock stop) (clock i) .boundary score = some next := by
    unfold Cursor.leave
    simp [kept,first,node,payload,ordered,okScore,expected,op]
  obtain ⟨v,completed,popped,worker,opened,atEnd,frame,data,stamped,_⟩ :=
    EndgameSearch.CacheHistory.NativeTrace.matched_boundary _ _ top kept payload closed
  change last.cursor.clock ≤ clock stop at bodyTime
  have exit := EndgameSearch.CacheHistory.NativeTrace.advance_clock last.cursor
    (.leave (clock stop) (clock i) .boundary score) (by simpa [Cursor.advance,Raw.step,bodyTime] using closed)
  have pop : last.advance inputs (.leave (clock stop) (clock i) .boundary score none) = some ⟨next,last.rows⟩ := by
    simp [State.advance,Cursor.advance,Raw.step,bodyTime,closed]
  refine ⟨⟨next,last.rows⟩,v,?_,popped,exit.2,rows,completed.trans (by rw [visits]; exact congrArg (v :: ·) entryVisits),
    worker,opened,atEnd,stamped,frame,data⟩
  change advance inputs entered ((body.calls (thread w)).events clock (i+1)) = some last at ran
  simp only [boundary,Calls.events,Invocations.Exit.node,Invocations.Exit.score,Option.toList_none,List.length_nil,
    Invocations.Exit.event,Option.map_none,Nat.add_zero,List.append_nil,advance,List.foldlM_append,List.foldlM_cons,
    List.foldlM_nil,pure,bind,Option.bind] at ran ⊢
  dsimp only [args,receiver] at pushed
  rw [pushed]; dsimp only; rw [ran]; dsimp only
  change (last.advance inputs (.leave (clock stop) (clock i) .boundary score none)).bind some = _
  rw [pop]; rfl

theorem boundary_body (w : Worker p) (site owner : Nat) (edge : Edge.mk .pvs site .boundary .forward ∈ transfers)
    (input : Input) (returned : input.arm ≠ .failed) (score : Option Nat) (deep shallow)
    (clock : Nat → Nat) (time : ∀ {a b}, a < b → clock a < clock b) (i : Nat)
    (reported : ∀ t, score = some t → clock (finishIndex w input deep shallow i) ≤ t)
    (empty : ∀ j, i ≤ j → j < finishIndex w input deep shallow i → inputs (clock j) = none) :
    OwnerSites.Body w inputs (.thread (thread w)) owner clock i (boundary w site owner input score deep shallow) := by
  intro s caller ready _
  obtain ⟨out,_,ran,stack,tick,rows,_⟩ := boundary_visit w site owner edge input returned score deep shallow clock time i
    reported empty s caller ready
  refine ⟨out,[],ran,stack,?_,rows,by simp⟩
  rw [tick]
  apply time
  simp only [finishIndex,boundary,Calls.count,Invocations.Exit.score,Option.toList_none,List.length_nil]
  omega

end NeuralReversi.PvsParallel.NativeFrames.CacheCalls

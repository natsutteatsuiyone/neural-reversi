import NeuralReversi.PvsParallel.NativeStopReturns.Boundary

namespace NeuralReversi.PvsParallel.NativeStopReturns

open Threading.AtomicWord (Event)
open EndgameSearch.CacheHistory.NativeTrace (Raw)
open NativeLeafBranches (BoundaryResult)
open ScoreRefresh (Score scoreZero)

structure FrameEnd where
  worker : Nat
  opening : Nat
  finish : Nat
  pcs : List Nat
  raw : Score

def FrameEnd.edges (s : FrameEnd) (stream : List NativeFrames.Record) : List Raw :=
  (EndgameSearch.CacheHistory.NativeTrace.workerEvents s.worker (NativeFrameCache.project stream)).filter
    (fun e => decide (s.opening ≤ e.step ∧ e.step ≤ s.finish))

def FrameEnd.body (s : FrameEnd) (stream : List NativeFrames.Record) : Option (List Raw) :=
  match s.edges stream with
  | .enter first _ .pvs .. :: tail =>
    match tail.reverse with
    | .leave last opening .pvs _ :: rest =>
      if first = s.opening ∧ opening = s.opening ∧ last = s.finish ∧ first < last then some rest.reverse else none
    | _ => none
  | _ => none

def FrameEnd.direct (s : FrameEnd) (node : EndgameSearch.CacheOwner.Node) (ordinal childWorker childFirst childLast : Nat)
    (stream : List NativeFrames.Record) : Bool :=
  match s.body stream with
  | some (.enter first site entered .. :: rest) =>
    match rest.getLast? with
    | some (.leave last opened left _) => decide (entered = node ∧ left = node ∧ site = ordinal ∧ s.worker = childWorker ∧
        s.opening < first ∧ first = childFirst ∧ last = childLast ∧ opened = first ∧ last < s.finish)
    | _ => false
  | _ => false

def FrameEnd.directPass (s child : FrameEnd) (stream : List NativeFrames.Record) :=
  s.direct .pvs passSite child.worker child.opening child.finish stream

def FrameEnd.route (s : FrameEnd) (mode : Mode) : Option Route :=
  routes.find? fun r => r.mode == mode && r.pcs == s.pcs

def Route.code (r : Route) := r.ops.filter (fun op => op == .zero || op == .negate || op == .solve)

def expected : Mode → List Op
  | .cached | .postJoin => [.zero]
  | .passChild => [.negate]
  | .terminal => [.solve]

private theorem generated : ∀ r ∈ routes, r.publications = [] ∧ r.code = expected r.mode := by decide

theorem route_effects {s : FrameEnd} {route : Route} (found : s.route mode = some route) :
    route.publications = [] ∧ route.code = expected mode ∧ route.pcs = s.pcs := by
  have member := List.mem_of_find?_eq_some found
  have ok := List.find?_some (p := fun r : Route => r.mode == mode && r.pcs == s.pcs) found
  obtain ⟨same,pcs⟩ := Bool.and_eq_true_iff.mp ok
  exact ⟨(generated _ member).1,beq_iff_eq.mp same ▸ (generated _ member).2,beq_iff_eq.mp pcs⟩

def value (checked : Bool) (board : Board.Board) (empties : BitVec 32) (child : Option BoundaryResult) :
    List Op → Option BoundaryResult
  | [.zero] => some ⟨scoreZero,false⟩
  | [.negate] => do
    let c ← child
    return ⟨← (ScoreRefresh.negate checked c.raw).toOption,c.useScore⟩
  | [.solve] => (ScoreRefresh.BoardBridge.solve checked board empties).toOption.map (⟨·,true⟩)
  | _ => none

def finish (checked : Bool) (s : FrameEnd) (mode : Mode) (b : Board.Board) (empties : BitVec 32)
    (child : Option BoundaryResult) : Option BoundaryResult := do
  let r ← s.route mode
  let answer ← value checked b empties child r.code
  if answer.raw = s.raw then some answer else none

theorem finish_effects (accepted : finish checked s mode b empties child = some answer) :
    ∃ r, s.route mode = some r ∧ r.publications = [] ∧
      value checked b empties child (expected mode) = some answer ∧ s.raw = answer.raw := by
  obtain ⟨r,route,h⟩ := Option.bind_eq_some_iff.mp accepted
  obtain ⟨x,word,eq⟩ := Option.bind_eq_some_iff.mp h
  split at eq <;> try contradiction
  cases eq
  obtain ⟨pub,code,_⟩ := route_effects route
  exact ⟨r,route,pub,code ▸ word,(by assumption : answer.raw = s.raw).symm⟩

structure Inputs where
  args : Nat → Nat → Option NativeFrames.Args
  empties : Nat → Nat → Option (BitVec 32)
  stream : List NativeFrames.Record
  checked : Bool

inductive Trace (m : Memory) where
  | stop (frame : FrameEnd) (atPoll : Event) (chain : ChainRead m atPoll) (loads : Loads m atPoll)
  | boundary (frame : FrameEnd) (leaf : Leaf) (atPoll : Event) (chain : ChainRead m atPoll) (loads : Loads m atPoll)
  | terminal (frame : FrameEnd)
  | pass (frame : FrameEnd) (child : Trace m)

def Trace.frame : Trace m → FrameEnd
  | .stop s .. | .boundary s .. | .terminal s | .pass s _ => s

def stopValue (i : Inputs) (s : FrameEnd) (c : ChainRead m atPoll) (loads : Loads m atPoll) :
    Option BoundaryResult := do
  let _ ← s.body i.stream
  if atPoll.worker ≠ s.worker ∨ loads.started ≤ s.opening ∨ s.finish ≤ atPoll.step ∨
      c.within loads.started = false then none else
  if poll c loads != some true then none else
  finish i.checked s c.mode ⟨0,0⟩ 0 none

def terminalValue (i : Inputs) (s : FrameEnd) : Option BoundaryResult := do
  let [] ← s.body i.stream | none
  let args ← i.args s.worker s.opening
  let empties ← i.empties s.worker s.opening
  finish i.checked s .terminal args.board empties none

def Leaf.pollInActivation (l : Leaf) (started : Nat) : Bool :=
  match l.site with | .inherited _ => true | _ => decide (l.first < started)

def leafValue (i : Inputs) (s : FrameEnd) (leaf : Leaf) (chain : ChainRead m t) (loads : Loads m t) :
    Option BoundaryResult := do
  if s.pcs != boundaryPcs || !leaf.pollInActivation loads.started ||
      !s.direct .boundary DeepLeafMir.boundaryPvsSite s.worker leaf.first leaf.last i.stream then none else
  if poll chain loads != some true then none else
  let r ← leaf.resolve i.args i.stream s.worker s.opening chain
  let stopped := match leaf.site with | .storeGuard .. => true | _ => !r.useScore
  if stopped && r.raw == s.raw then some ⟨r.raw,false⟩ else none

theorem leaf_source {chain : ChainRead m t} (hit : leafValue i s leaf chain loads = some r) :
    Cause m t chain.chain ∧ r.raw = s.raw ∧ r.useScore = false ∧
      ∃ child, leaf.resolve i.args i.stream s.worker s.opening chain = some child ∧ child.raw = r.raw ∧
        (match leaf.site with | .storeGuard .. => True | _ => child.useScore = false) := by
  unfold leafValue at hit
  split at hit <;> try contradiction
  split at hit <;> try contradiction
  rename_i positive
  obtain ⟨actual,resolved,last⟩ := Option.bind_eq_some_iff.mp hit
  cases site : leaf.site with
  | recursive _ _ _ | inherited _ =>
    simp only [site] at last
    split at last <;> try contradiction
    rename_i sameWord
    cases last
    obtain ⟨no,raw⟩ := Bool.and_eq_true_iff.mp sameWord
    exact ⟨cause_of_poll chain loads (by simpa using positive),beq_iff_eq.mp raw,rfl,_,resolved,rfl,by simpa using no⟩
  | storeGuard pc before after =>
    simp only [site,Bool.true_and] at last
    split at last <;> try contradiction
    rename_i sameWord
    cases last
    exact ⟨cause_of_poll chain loads (by simpa using positive),beq_iff_eq.mp sameWord,rfl,_,resolved,rfl,True.intro⟩

def Trace.decode (i : Inputs) : Trace m → Option BoundaryResult
  | .stop s _ c loads => stopValue i s c loads
  | .boundary s l _ c loads => leafValue i s l c loads
  | .terminal s => terminalValue i s
  | .pass s child => do
    let word ← child.decode i
    if s.directPass child.frame i.stream then finish i.checked s .passChild ⟨0,0⟩ 0 (some word) else none

def exitEvent (s : FrameEnd) (r : BoundaryResult) : NativeFrames.Record :=
  .localCall s.worker (if r.useScore then .leave s.finish s.opening .pvs none (some r.raw)
    else .discarded s.finish s.opening r.raw)

theorem stopValue_source {c : ChainRead m atPoll} (hit : stopValue i s c loads = some r) :
    Cause m atPoll c.chain ∧ r = ⟨scoreZero,false⟩ ∧
    ∃ route, s.route c.mode = some route ∧ route.publications = [] := by
  unfold stopValue at hit
  obtain ⟨_,_,rest⟩ := Option.bind_eq_some_iff.mp hit
  split at rest <;> try contradiction
  split at rest <;> try contradiction
  rename_i running
  have polled : poll c loads = some true := by simpa using running
  obtain ⟨route,found,pub,word,_⟩ := finish_effects rest
  have ans : r = ⟨scoreZero,false⟩ := by cases c <;> simpa [ChainRead.mode,expected,value] using word.symm
  exact ⟨cause_of_poll c loads polled,ans,route,found,pub⟩

theorem terminal_usable (hit : terminalValue i s = some r) : r.useScore = true := by
  unfold terminalValue at hit
  obtain ⟨body,_,rest⟩ := Option.bind_eq_some_iff.mp hit
  cases body <;> try contradiction
  obtain ⟨_,_,more⟩ := Option.bind_eq_some_iff.mp rest
  obtain ⟨_,_,done⟩ := Option.bind_eq_some_iff.mp more
  obtain ⟨_,_,_,word,_⟩ := finish_effects done
  obtain ⟨_,_,rfl⟩ := Option.map_eq_some_iff.mp word
  rfl

def Trace.CutFrom (trace : Trace m) (atPoll : Event) (chain : List Nat) : Prop :=
  match trace with
  | .stop _ t c _ => t = atPoll ∧ c.chain = chain
  | .boundary _ _ t c _ => t = atPoll ∧ c.chain = chain
  | .terminal _ => False
  | .pass _ child => child.CutFrom atPoll chain

theorem cancellation_origin (tr : Trace m) (hit : tr.decode i = some r) (rejected : r.useScore = false) :
    ∃ atPoll chain, tr.CutFrom atPoll chain ∧ Cause m atPoll chain ∧
      exitEvent tr.frame r = .localCall tr.frame.worker (.discarded tr.frame.finish tr.frame.opening r.raw) := by
  have emits : exitEvent tr.frame r =
      .localCall tr.frame.worker (.discarded tr.frame.finish tr.frame.opening r.raw) := by simp [exitEvent,rejected]
  suffices ∃ atPoll chain, tr.CutFrom atPoll chain ∧ Cause m atPoll chain by
    obtain ⟨t,chain,src,cause⟩ := this
    exact ⟨t,chain,src,cause,emits⟩
  clear emits
  induction tr generalizing r with
  | stop s t c loads => exact ⟨t,c.chain,⟨rfl,rfl⟩,(stopValue_source hit).1⟩
  | boundary s leaf t c loads => exact ⟨t,c.chain,⟨rfl,rfl⟩,(leaf_source hit).1⟩
  | terminal s => simp_all [terminal_usable hit]
  | pass s child ih =>
    obtain ⟨x,childDone,next⟩ := Option.bind_eq_some_iff.mp hit
    split at next <;> try contradiction
    obtain ⟨_,_,_,word,_⟩ := finish_effects next
    change ((ScoreRefresh.negate i.checked x.raw).toOption.bind fun v => some ⟨v,x.useScore⟩) = some r at word
    obtain ⟨_,_,same⟩ := Option.bind_eq_some_iff.mp word
    cases same
    exact ih childDone rejected

theorem discarded_has_no_scored_row {p : EndgameSearch.CacheHistory.Pool}
    {w : EndgameSearch.CacheHistory.NativeTrace.Worker p} {s next : NativeFrames.State w}
    (success : s.advance args (.discarded t opening word) = some next) :
    ∀ r ∈ s.rows, r.opening ≠ opening := by
  unfold NativeFrames.State.advance at success
  obtain ⟨_,_,rest⟩ := Option.bind_eq_some_iff.mp success
  split at rest <;> try contradiction
  rename_i absent
  simpa using absent

end NeuralReversi.PvsParallel.NativeStopReturns

module
public import NeuralReversi.PvsParallel.NativeStopReturns.Poll
public import NeuralReversi.PvsParallel.NativeLeafBranches

@[expose] public section

namespace NeuralReversi.PvsParallel.NativeStopReturns

open EndgameSearch.CacheHistory (Pool Visit)
open EndgameSearch.CacheOwner (Node)
open EndgameSearch.CacheHistory.NativeTrace (Raw)
open Threading.AtomicWord (Event)
open NativeLeafBranches (BoundaryResult)

structure NodeRead where
  opening : Nat
  pc : Nat
  event : Event
  word : BitVec 64

inductive LeafSite
  | recursive (callee : Node) (opening pc : Nat)
  | storeGuard (pc : Nat) (before after : NodeRead)
  | inherited (positiveRead : Event)

structure Leaf where
  pool : Pool
  moves : Bitboard.Moves.X86Config
  flips : Cutoffs.FlipConfig
  caches : EndgameSearch.CacheHistory.Payloads.Values
  regs : NativeLeafBranches.Registers
  first : Nat
  last : Nat
  site : LeafSite

def stackAt (events : List Raw) (time : Nat) : Option (List (Nat × Node)) :=
  (events.filter fun e => decide (e.step < time)).foldlM (init := []) fun stack ev =>
    match ev with
    | .enter first _ node .. => some ((first,node) :: stack)
    | .leave _ first node .. =>
      match stack with
      | top :: tail => if top = (first,node) then some tail else none
      | [] => none

def retainedWorker (events : List Raw) (first last : Nat) : Bool := Id.run do
  let some oldStack := stackAt events first | return false
  let some newStack := stackAt events last | return false
  let inside := fun (s : List (Nat × Node)) => s.find? (fun f => f.2 == .splitSearch)
  let some owned := inside oldStack | return false
  let splitEdge := events.any fun ev =>
    decide (first ≤ ev.step ∧ ev.step < last) && match ev with
      | .enter _ _ .split .. | .leave _ _ .split _ => true
      | _ => false
  return decide (first < last) && inside newStack == some owned && !splitEdge

def Leaf.atSite (l : Leaf) (c : NativeFrames.Capture l.pool) (v : Visit l.pool)
    (cRead : ChainRead m t) (events : List Raw) : Bool := Id.run do
  let some (top :: _) := stackAt events t.step | return false
  let input := v.input
  match l.site with
  | .recursive node first pc =>
    if cRead.mode != .cached || top != (first,node) || !input.active then return false
    let allowed := if node == .deep then deepPollPcs else if node == .deepChild then movePollPcs else []
    if !allowed.contains pc then return false
    let .deep true := input.arm | return false
    let history := (c.timeline.readAt l.moves l.flips v.worker v.first).history
    let out := EndgameSearch.DeepRecursionMir.run input.target input.bmi2 input.paired l.moves l.flips
      l.pool.address input.board (ScoreRefresh.toDiscDiff input.alpha).raw.toInt (input.state history)
    let .cached _ cache .. := cRead | return false
    return out.score.isNone && out.state.polls.cache.latched && cache.latched && cache.seen == out.state.polls.cache.seen
  | .storeGuard pc before after =>
    let current := (v.first,Node.boundary)
    let atBoundary := fun stamp => (stackAt events stamp).bind List.head? == some current
    let here := events.filter fun e => decide (v.first ≤ e.step ∧ e.step ≤ v.last)
    let access := here.findSome? fun | .enter time _ .access .. => some time | _ => none
    let deepEnd := here.reverse.findSome? fun | .leave time _ .deep _ => some time | _ => none
    let some (accessAt,returnedAt) := Prod.mk <$> access <*> deepEnd | return false
    return cRead.mode == .cached && top == current && pc == boundaryPollPc &&
      before.opening == v.first && after.opening == v.first && before.pc == nodesBeforePc && after.pc == nodesAfterPc &&
      before.event.worker == t.worker && after.event.worker == t.worker &&
      decide (v.first < before.event.step ∧ before.event.step < accessAt ∧ returnedAt < after.event.step ∧ after.event.step < t.step) &&
      atBoundary before.event.step && atBoundary after.event.step &&
      input.parallel && input.depth.toNat == 12 && (input.lookup matches some none) &&
      DeepLeafMir.enoughNodes input.checked before.word after.word
  | .inherited positive =>
    let .deep true := input.arm | return false
    let .cached _ cache reads .. := cRead | return false
    return cache.latched && input.polls.cache.latched && cache.seen == input.polls.cache.seen &&
      positive.worker == t.worker && decide (positive.step ≤ t.step ∧ t.step < v.first) &&
      reads.any (fun r => r.event == positive && r.value) && retainedWorker events positive.step v.first

def Leaf.resolve (l : Leaf) (args : Nat → Nat → Option NativeFrames.Args) (records : List NativeFrames.Record)
    (worker opened : Nat) (cRead : ChainRead m t) : Option BoundaryResult := do
  let checked ← NativeFrameCache.decode l.pool l.moves l.flips args l.caches records
  let v ← checked.frames.timeline.visits.find? (fun v => decide (v.worker = worker ∧ v.first = l.first ∧ v.last = l.last))
  let received ← args worker opened
  if !received.endgame || received.pv || received.board != v.input.board.val ||
      received.depth != v.input.depth || received.alpha != v.input.alpha || received.beta != v.input.beta then none else
  let during := match l.site with | .inherited _ => decide (t.step < v.first) | _ => decide (v.first < t.step ∧ t.step < v.last)
  if worker != t.worker || !during then none else
  let raw := EndgameSearch.CacheHistory.NativeTrace.workerEvents worker (NativeFrameCache.project records)
  if !l.atSite checked.frames v cRead raw then none else
  NativeLeafBranches.returned checked.frames records l.regs l.moves l.flips v

end NeuralReversi.PvsParallel.NativeStopReturns

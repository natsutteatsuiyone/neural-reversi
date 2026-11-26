import NeuralReversi.PvsParallel.NativeFrames.CacheCalls
import NeuralReversi.EndgameSearch.CacheHistory.Payloads.Lookup

namespace NeuralReversi.PvsParallel.NativeFrames.CacheCalls

open EndgameSearch.CacheHistory (Pool Input Local Good Visit Timeline)
open EndgameSearch.CacheHistory.NativeTrace (Worker)
open EndgameSearch.CacheHistory.Payloads
open EndgameTreeCache (Table)
open NativeFrameCache (EdgeEvent)

def deepLookup : Forest .deep :=
  .call 1 .deepSlot .deepField (by decide) .nil (.call 2 .deepProbe .deepField (by decide) .nil .nil)

def shallowLookup : Forest .shallow :=
  .call 1 .shallowSlot .forward (by decide) .nil (.call 2 .shallowProbe .forward (by decide) .nil .nil)

def lookupStream (w : Worker p) (site owner : Nat) (i : Input) (stamp : Option Nat) (clock : Nat → Nat) (start : Nat) :=
  NativeFrameCache.project (((boundary w site owner i stamp deepLookup shallowLookup).events clock start).map
    (Record.localCall w.ordinal))

private theorem lookup_sites (w : Worker p) (site owner : Nat) (i : Input) (stamp : Option Nat)
    (clock : Nat → Nat) (n : Nat) (t : Table) (arm : recursiveArm i t) :
    (lookupStream w site owner i stamp clock n).filterMap site? =
      [⟨w.ordinal,clock (n+4),clock (n+5),t,.slot⟩,⟨w.ordinal,clock (n+6),clock (n+7),t,.probe⟩] := by
  unfold lookupStream boundary
  cases t <;> first
    | (obtain ⟨six,arm⟩ := arm; rw [arm]; cases six)
    | (obtain ⟨poll,arm⟩ := arm; rw [arm]; cases poll)
  all_goals simp [plan,deepLookup,shallowLookup,Forest.calls,Invocations.Calls.events,Invocations.Calls.count,
    Invocations.Exit.node,Invocations.Exit.score,Invocations.Exit.event,NativeFrameCache.project,Event.edge?,
    site?,target,List.filterMap_cons,Nat.add_assoc]

/-- Record the two returned operand packets in their worker's own clock. The
    constructors read the array at the actual hash slot, also for warmed cells. -/
def lookupPackets (w : Worker p) (i : Input) (t : Table) (s : Local) (shape : ∀ t, (s.cells t).size = p.sizes t)
    (clock : Nat → Nat) (n : Nat) (other : Values) : Values := fun worker opening =>
  if worker = w.ordinal ∧ opening = clock (n+4) then
    some ⟨clock (n+5),((Instruction.slot t (p.key i.board.val)).observe p s shape).data⟩
  else if worker = w.ordinal ∧ opening = clock (n+6) then
    some ⟨clock (n+7),((Instruction.probe t i.board.val (ScoreRefresh.toDiscDiff i.alpha).raw).observe p s shape).data⟩
  else other worker opening

private theorem requests_append (xs ys : List EdgeEvent) (w first last : Nat) :
    requests (xs ++ ys) w first last = requests xs w first last ++ requests ys w first last := by
  simp [requests,List.filterMap_append]

theorem collect_lookup (w : Worker p) (site owner : Nat) (i : Input) (stamp : Option Nat) (t : Table)
    (branch : recursiveArm i t) (clock : Nat → Nat) (time : ∀ {a b}, a < b → clock a < clock b) (n : Nat)
    (s : Local) (shape : ∀ t, (s.cells t).size = p.sizes t) (other : Values)
    (before after_ : List EdgeEvent)
    (earlier : requests before w.ordinal (clock n) (clock (n+9)) = [])
    (later : requests after_ w.ordinal (clock n) (clock (n+9)) = []) :
    collect (lookupPackets w i t s shape clock n other)
      (before ++ lookupStream w site owner i stamp clock n ++ after_) w.ordinal (clock n) (clock (n+9)) =
      some (observeRun p s shape (lookupOps p i t)) := by
  have start₁ := Nat.le_of_lt (time (show n < n+4 by omega))
  have start₂ := Nat.le_of_lt (time (show n < n+6 by omega))
  have end₁ := Nat.le_of_lt (time (show n+5 < n+9 by omega))
  have end₂ := Nat.le_of_lt (time (show n+7 < n+9 by omega))
  have diff := Nat.ne_of_gt (time (show n+4 < n+6 by omega))
  unfold collect
  rw [requests_append,requests_append,earlier,later,List.nil_append,List.append_nil]
  simp [requests,lookup_sites _ _ _ _ _ _ _ _ branch,Interval.contains,start₁,start₂,end₁,end₂,List.mapM_cons,
    packet,lookupPackets,diff,Instruction.observe,Data.kind,observeRun,lookupOps,Instruction.step,Instruction.write?]

/-- The packet audit of a completed local cache hit follows from the actual
    board lookup, including a boundary whose score the caller discarded. -/
theorem complete_lookup (w : Worker p) (site owner : Nat) (v : Visit p) (timeline : Timeline p)
    (t : Table) (branch : recursiveArm v.input t) (clock : Nat → Nat) (time : ∀ {a b}, a < b → clock a < clock b) (n : Nat)
    (worker : v.worker = w.ordinal) (opened : v.first = clock n) (finished : v.last = clock (n+9))
    (other : Values) (before after_ : List EdgeEvent)
    (earlier : requests before w.ordinal v.first v.last = []) (later : requests after_ w.ordinal v.first v.last = [])
    (moves : PvsBoardTree.legalSquares mc v.input.board.val ≠ [])
    (hit : let s := timeline.readAt mc fc v.worker v.first
           EndgameSearch.CacheMir.probe (s.cells t) (p.address t v.input.board.val) v.input.board.val
             (ScoreRefresh.toDiscDiff v.input.alpha).raw.toInt = some (some value)) :
    let s := timeline.readAt mc fc v.worker v.first
    let g := EndgameSearch.CacheHistory.replay_good mc fc p
      (EndgameSearch.CacheHistory.beforeStep timeline.visits v.worker v.first)
    let records := before ++ lookupStream w site owner v.input v.scoredAt clock n ++ after_
    completeVisit p timeline mc fc (lookupPackets w v.input t s g.shape clock n other) records v =
      some ⟨observeRun p s g.shape (lookupOps p v.input t),s⟩ := by
  dsimp only
  let s := timeline.readAt mc fc v.worker v.first
  have g : Good mc fc p.width p.sizes p.key s := EndgameSearch.CacheHistory.replay_good mc fc p _
  have payload := collect_lookup w site owner v.input v.scoredAt t branch clock time n s g.shape other before after_
    (opened ▸ finished ▸ earlier) (opened ▸ finished ▸ later)
  have audit := check_lookup_hit p v.input t g branch moves (by simpa [g.shape,Pool.address,EndgameSearch.CacheMir.address] using hit)
  unfold completeVisit
  rw [← worker,← opened,← finished] at payload
  rw [payload]
  exact audit.1

private theorem lookup_finish (w : Worker p) (i : Input) (t : Table) (arm : recursiveArm i t) (n : Nat) :
    finishIndex w i deepLookup shallowLookup n = n+9 := by
  unfold finishIndex
  cases t <;> first
    | (obtain ⟨six,arm⟩ := arm; rw [arm]; cases six)
    | (obtain ⟨poll,arm⟩ := arm; rw [arm]; cases poll)
  all_goals rfl

/-- Construct the visit and audit it from the same PVS caller, physical hit and
    two compiler-routed cache calls, without a pre-existing visit or audit. -/
theorem hit_at_caller (w : Worker p) (site owner : Nat)
    (edge : EndgameSearch.CacheOwner.Edge.mk .pvs site .boundary .forward ∈ EndgameSearch.CacheOwner.transfers)
    (i : Input) (t : Table) (branch : recursiveArm i t) (stamp : Option Nat)
    (clock : Nat → Nat) (time : ∀ {a b}, a < b → clock a < clock b) (n : Nat)
    (reported : ∀ k, stamp = some k → clock (n+9) ≤ k)
    (empty : ∀ j, n ≤ j → j < n+9 → inputs (clock j) = none) (s : State w)
    (caller : OwnerSites.Caller s.cursor .pvs (.thread (thread w)) owner) (ready : s.cursor.clock ≤ clock n)
    (timeline : Timeline p) (other : Values) (before after_ : List EdgeEvent)
    (earlier : requests before w.ordinal (clock n) (clock (n+9)) = [])
    (later : requests after_ w.ordinal (clock n) (clock (n+9)) = [])
    (moves : PvsBoardTree.legalSquares mc i.board.val ≠ [])
    (hit : let prior := timeline.readAt mc fc w.ordinal (clock n)
           EndgameSearch.CacheMir.probe (prior.cells t) (p.address t i.board.val) i.board.val
             (ScoreRefresh.toDiscDiff i.alpha).raw.toInt = some (some value)) :
    let prior := timeline.readAt mc fc w.ordinal (clock n)
    let shape := (EndgameSearch.CacheHistory.replay_good mc fc p
      (EndgameSearch.CacheHistory.beforeStep timeline.visits w.ordinal (clock n))).shape
    let raw := boundary w site owner i stamp deepLookup shallowLookup
    let packets := lookupPackets w i t prior shape clock n other
    let records := before ++ lookupStream w site owner i stamp clock n ++ after_
    ∃ (out : State w) (v : Visit p), OwnerSites.advance inputs s (raw.events clock n) = some out ∧
      out.cursor.finished = v :: s.cursor.finished ∧ v.frame = (boundaryArgs w owner i).frame ∧
      v.worker = w.ordinal ∧ v.first = clock n ∧ v.last = clock (n+9) ∧ v.input = i ∧ v.scoredAt = stamp ∧
      completeVisit p timeline mc fc packets records v = some ⟨observeRun p prior shape (lookupOps p i t),prior⟩ := by
  have finish := lookup_finish w i t branch n
  have dispatched : i.arm ≠ .failed := by
    cases t with
    | shallow => obtain ⟨_,arm⟩ := branch; simp [arm]
    | deep => obtain ⟨poll,arm⟩ := branch; simp [arm]
  obtain ⟨out,v,ran,_,_,_,visits,worker,opened,closed,scored,frame,input⟩ :=
    boundary_visit w site owner edge i dispatched stamp deepLookup shallowLookup clock time n
      (finish.symm ▸ reported) (finish.symm ▸ empty) s caller ready
  rw [finish] at closed
  refine ⟨out,v,ran,visits,frame,worker,opened,closed,input,scored,?_⟩
  have audit := complete_lookup w site owner v timeline t (input.symm ▸ branch) clock time n worker opened closed
    other before after_ (opened.symm ▸ closed.symm ▸ earlier) (opened.symm ▸ closed.symm ▸ later)
    (input.symm ▸ moves) (value:=value) (by simpa only [input,worker,opened] using hit)
  let packets (j : Input) (id start : Nat) := lookupPackets w j t (timeline.readAt mc fc id start)
    (EndgameSearch.CacheHistory.replay_good mc fc p _).shape clock n other
  change completeVisit _ _ _ _ (packets v.input v.worker v.first) _ _ = _ at audit
  simpa only [input,worker,opened,scored] using audit

end NeuralReversi.PvsParallel.NativeFrames.CacheCalls

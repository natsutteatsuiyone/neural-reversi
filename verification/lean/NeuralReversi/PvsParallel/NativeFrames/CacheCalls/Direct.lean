import NeuralReversi.PvsParallel.NativeFrames.CacheCalls.Lookup

namespace NeuralReversi.PvsParallel.NativeFrames.CacheCalls

open EndgameSearch.CacheHistory (Pool Input Local Timeline Visit)
open EndgameSearch.CacheHistory.NativeTrace (Worker)
open EndgameSearch.CacheHistory.Payloads
open NativeFrameCache (EdgeEvent)
open PvsBoardTree (legalSquares)

/-- Branches of the boundary routine that reach its return without accessing
    either worker-local array. A terminal node has no legal move for either side. -/
inductive DirectExit (mc : Bitboard.Moves.X86Config) (i : Input) : Prop where
  | fixed (arm : i.arm = .fixed f)
  | table (arm : i.arm = .table e)
  | terminal (branch : recursiveArm i t) (ours : legalSquares mc i.board.val = [])
      (theirs : Board.hasLegalMoves mc (PvsBoardTree.pass i.board).val = false)

private theorem DirectExit.history (path : DirectExit mc i) (p : Pool) (s : Local) :
    (i.run mc fc p.address s.history).history = s.history := by
  cases path with
  | fixed arm => simp [Input.run,arm,Input.eval]
  | table arm => simp [Input.run,arm,Input.eval]
  | @terminal t branch ours theirs =>
    cases t with
    | shallow =>
      obtain ⟨_,arm⟩ := branch
      simp only [Input.run,arm,Input.eval]
      rw [EndgameSearch.ShallowMir.run.eq_def]
      split
      · simp [theirs]
      · simp_all
    | deep =>
      obtain ⟨poll,arm⟩ := branch
      simp only [Input.run,arm,Input.eval]
      rw [EndgameSearch.DeepRecursionMir.run.eq_def]
      split
      · cases poll <;> simp [theirs,DeepLeafMir.pollInput,Input.state]
      · simp_all

private theorem DirectExit.returns (path : DirectExit mc i) : i.arm ≠ .failed := by
  cases path with
  | fixed arm => simp [arm]
  | table arm => simp [arm]
  | @terminal t branch _ _ =>
    cases t with
    | shallow => obtain ⟨_,arm⟩ := branch; simp [arm]
    | deep => obtain ⟨_,arm⟩ := branch; simp [arm]

def directStream (w : Worker p) (site owner : Nat) (i : Input) (stamp : Option Nat) (clock : Nat → Nat) (n : Nat) :=
  NativeFrameCache.project (((boundary w site owner i stamp .nil .nil).events clock n).map (Record.localCall w.ordinal))

private theorem direct_sites (w : Worker p) (site owner : Nat) (i : Input) (stamp : Option Nat) (clock : Nat → Nat) (n : Nat) :
    (directStream w site owner i stamp clock n).filterMap site? = [] := by
  unfold directStream boundary
  cases i.arm
  case' deep poll => cases poll
  case' shallow six => cases six
  all_goals simp [plan,Forest.calls,Invocations.Calls.events,Invocations.Calls.count,
    Invocations.Exit.node,Invocations.Exit.score,Invocations.Exit.event,NativeFrameCache.project,Event.edge?,site?,
    target,List.filterMap_cons]

theorem direct_at_caller (w : Worker p) (site owner : Nat)
    (edge : EndgameSearch.CacheOwner.Edge.mk .pvs site .boundary .forward ∈ EndgameSearch.CacheOwner.transfers)
    (i : Input) (path : DirectExit mc i) (stamp : Option Nat)
    (clock : Nat → Nat) (time : ∀ {a b}, a < b → clock a < clock b) (n : Nat)
    (reported : ∀ k, stamp = some k → clock (finishIndex w i .nil .nil n) ≤ k)
    (empty : ∀ j, n ≤ j → j < finishIndex w i .nil .nil n → inputs (clock j) = none)
    (s : State w) (caller : OwnerSites.Caller s.cursor .pvs (.thread (thread w)) owner) (ready : s.cursor.clock ≤ clock n)
    (timeline : Timeline p) (values : Values) (before after_ : List EdgeEvent)
    (earlier : requests before w.ordinal (clock n) (clock (finishIndex w i .nil .nil n)) = [])
    (later : requests after_ w.ordinal (clock n) (clock (finishIndex w i .nil .nil n)) = []) :
    let events := (boundary w site owner i stamp .nil .nil).events clock n
    let records := before ++ directStream w site owner i stamp clock n ++ after_
    ∃ (out : State w) (v : Visit p), OwnerSites.advance inputs s events = some out ∧
      out.cursor.finished = v::s.cursor.finished ∧ v.frame = (boundaryArgs w owner i).frame ∧ v.input = i ∧
      v.scoredAt = stamp ∧ v.worker = w.ordinal ∧ v.first = clock n ∧ v.last = clock (finishIndex w i .nil .nil n) ∧
      completeVisit p timeline mc fc values records v = some ⟨[],timeline.readAt mc fc w.ordinal (clock n)⟩ := by
  obtain ⟨out,v,ran,_,_,_,visits,worker,opened,closed,scored,frame,input⟩ :=
    boundary_visit w site owner edge i path.returns stamp .nil .nil clock time n reported empty s caller ready
  refine ⟨out,v,ran,visits,frame,input,scored,worker,opened,closed,?_⟩
  unfold completeVisit collect
  rw [worker,opened,closed,input]
  simp only [requests,List.filterMap_append,List.filter_append,List.map_append,direct_sites] at earlier later ⊢
  simp [earlier,later,check,path.history]

end NeuralReversi.PvsParallel.NativeFrames.CacheCalls

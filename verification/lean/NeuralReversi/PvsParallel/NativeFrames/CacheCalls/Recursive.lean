import NeuralReversi.PvsParallel.NativeFrames.CacheCalls.Packets

namespace NeuralReversi.PvsParallel.NativeFrames.CacheCalls

open EndgameSearch.CacheOwner (Node Edge transfers)
open EndgameSearch.CacheHistory (Pool Local Input Arm Visit Timeline)
open EndgameSearch.CacheHistory.NativeTrace (Worker)
open EndgameSearch.CacheHistory.Payloads (Values requests collect completeVisit check observeRun)
open EndgameTreeCache (soundHistory)
open Invocations (Calls)
open NativeFrameCache (EdgeEvent)

inductive Routine where
  | shallow (six : Bool) | deep (poll : Bool)

def Routine.node : Routine → Node | .shallow _ => .shallow | .deep _ => .deep
def Routine.arm : Routine → Arm | .shallow six => .shallow six | .deep poll => .deep poll

def Routine.deepTree (r : Routine) : Forest r.node → Forest .deep :=
  match r with | .shallow _ => fun _ => .nil | .deep _ => id

def Routine.shallowTree (r : Routine) : Forest r.node → Forest .shallow :=
  match r with | .shallow _ => id | .deep _ => fun _ => .nil

def Routine.calls (r : Routine) (w : Worker p) (site owner : Nat) (i : Input) (stamp : Option Nat)
    (f : Forest r.node) := boundary w site owner i stamp (r.deepTree f) (r.shallowTree f)

def Routine.stream (r : Routine) (w : Worker p) (site owner : Nat) (i : Input) (stamp : Option Nat)
    (f : Forest r.node) (clock : Nat → Nat) (n : Nat) :=
  NativeFrameCache.project (((r.calls w site owner i stamp f).events clock n).map (Record.localCall w.ordinal))

private theorem finish (r : Routine) (w : Worker p) (i : Input) (branch : i.arm = r.arm) (f : Forest r.node) (n : Nat) :
    finishIndex w i (r.deepTree f) (r.shallowTree f) n = n+5+(f.calls (thread w)).count := by
  unfold finishIndex
  rw [branch]
  cases r
  case' deep poll => cases poll
  case' shallow six => cases six
  all_goals dsimp only [Routine.node] at *
  all_goals simp [plan,Routine.deepTree,Routine.shallowTree,Routine.arm,Forest.calls,Calls.count,Invocations.Exit.score]
  all_goals omega

private theorem sites_append' (w : Nat) (xs ys : List Event) :
    sitesOf w (xs ++ ys) = sitesOf w xs ++ sitesOf w ys := by simp [sitesOf,NativeFrameCache.project,List.filterMap_append]

private theorem enter_empty (w t site n receiver args) : sitesOf w [.entry t site n receiver args] = [] := rfl
private theorem leave_empty (w t offset n stamp) (noAccess : EndgameSearch.CacheHistory.Payloads.target n = none) :
    sitesOf w [.leave t offset n stamp none] = [] := by
  simp [sitesOf,NativeFrameCache.project,Event.edge?,EndgameSearch.CacheHistory.Payloads.site?,noAccess]

theorem Routine.sites (r : Routine) (w : Worker p) (site owner : Nat) (i : Input) (branch : i.arm = r.arm)
    (stamp : Option Nat) (f : Forest r.node) (clock : Nat → Nat) (n : Nat) :
    (r.stream w site owner i stamp f clock n).filterMap EndgameSearch.CacheHistory.Payloads.site? =
      sitesOf w.ordinal ((f.calls (thread w)).events clock (n+4)) := by
  change sitesOf _ _ = _
  unfold calls boundary
  rw [branch]
  cases r
  case' deep poll => cases poll
  case' shallow six => cases six
  all_goals
    dsimp only [Routine.node] at *
    simp only [arm,deepTree,shallowTree,id_eq,plan,Forest.calls,Calls.events,Invocations.Exit.score,
      Invocations.Exit.node,Invocations.Exit.event,Calls.count,Option.toList_none,Option.map_none,List.length_nil,
      Nat.add_zero,List.append_nil,sites_append',enter_empty]
    simp only [leave_empty _ _ _ .access _ rfl,leave_empty _ _ _ .boundary _ rfl,
      leave_empty _ _ _ .shallow _ rfl,leave_empty _ _ _ .deep _ rfl,List.append_nil,List.nil_append]

/-- No operand request inside the recursive body is lost when the boundary
    also contains getter, routine entry and routine return instructions. -/
theorem Routine.collect_packets (r : Routine) (w : Worker p) (site owner : Nat) (i : Input) (branch : i.arm = r.arm)
    (stamp : Option Nat) (code : Program r.node h last) (clock : Nat → Nat)
    (time : ∀ {a b}, a < b → clock a < clock b) (n : Nat) (s : Local)
    (shape : ∀ t, (s.cells t).size = p.sizes t) (other : Values) (before after_ : List EdgeEvent)
    (earlier : requests before w.ordinal (clock n) (clock (n+5+(code.forest.calls (thread w)).count)) = [])
    (later : requests after_ w.ordinal (clock n) (clock (n+5+(code.forest.calls (thread w)).count)) = []) :
    let xs := code.schedule (thread w) (n+4)
    let values := Issued.install p w.ordinal clock other xs s shape
    collect values (before ++ r.stream w site owner i stamp code.forest clock n ++ after_)
      w.ordinal (clock n) (clock (n+5+(code.forest.calls (thread w)).count)) =
      some (observeRun p s shape (code.accesses.map Access.instruction)) := by
  let xs := code.schedule (thread w) (n+4)
  let lastIndex := n+5+(code.forest.calls (thread w)).count
  let interval : EndgameSearch.CacheHistory.Payloads.Interval := ⟨w.ordinal,clock n,clock lastIndex⟩
  have inside : ((xs.map (fun x => EndgameSearch.CacheHistory.Payloads.Site.mk w.ordinal (x.request clock))).filter
      interval.contains) = xs.map (fun x => ⟨w.ordinal,x.request clock⟩) := by
    apply List.filter_eq_self.mpr
    intro a member
    obtain ⟨x,issued,rfl⟩ := List.mem_map.mp member
    obtain ⟨first,last⟩ := (code.schedule_order (thread w) (n+4)).2 x issued
    have lo := Nat.le_of_lt (time (show n < x.index by omega))
    have hi := Nat.le_of_lt (time (show x.index+1 < lastIndex by dsimp [lastIndex]; omega))
    simp [interval,EndgameSearch.CacheHistory.Payloads.Interval.contains,Issued.request,lo,hi]
  have middle : requests (r.stream w site owner i stamp code.forest clock n) w.ordinal (clock n) (clock lastIndex) =
      xs.map (Issued.request clock) := by
    unfold requests
    rw [r.sites _ _ _ _ branch,code.schedule_sites]
    change (((xs.map _).filter interval.contains).map _) = _
    rw [inside,List.map_map]
    rfl
  have combine (a b : List EdgeEvent) : requests (a ++ b) w.ordinal (clock n) (clock lastIndex) =
      requests a w.ordinal (clock n) (clock lastIndex) ++ requests b w.ordinal (clock n) (clock lastIndex) := by
    simp [requests,List.filterMap_append]
  unfold collect
  rw [combine,combine,earlier,later,middle,List.nil_append,List.append_nil]
  exact code.read_packets _ _ _ _ _ time _ _ _

/-- Replay one complete recursive boundary directly from its native call tree.
    A discarded score changes neither the completed visit nor its array effects. -/
theorem Routine.at_caller (r : Routine) (w : Worker p) (site owner : Nat)
    (edge : Edge.mk .pvs site .boundary .forward ∈ transfers) (i : Input) (branch : i.arm = r.arm)
    (stamp : Option Nat) (clock : Nat → Nat) (time : ∀ {a b}, a < b → clock a < clock b) (n : Nat)
    (timeline : Timeline p)
    (code : let h := (timeline.readAt mc fc w.ordinal (clock n)).history
            Program r.node h (i.run mc fc p.address h).history)
    (reported : ∀ k, stamp = some k → clock (n+5+(code.forest.calls (thread w)).count) ≤ k)
    (empty : ∀ j, n ≤ j → j < n+5+(code.forest.calls (thread w)).count → inputs (clock j) = none)
    (s : State w) (caller : OwnerSites.Caller s.cursor .pvs (.thread (thread w)) owner) (ready : s.cursor.clock ≤ clock n)
    (other : Values) (before after_ : List EdgeEvent)
    (earlier : requests before w.ordinal (clock n) (clock (n+5+(code.forest.calls (thread w)).count)) = [])
    (later : requests after_ w.ordinal (clock n) (clock (n+5+(code.forest.calls (thread w)).count)) = []) :
    let prior := timeline.readAt mc fc w.ordinal (clock n)
    let g := EndgameSearch.CacheHistory.replay_good mc fc p
      (EndgameSearch.CacheHistory.beforeStep timeline.visits w.ordinal (clock n))
    let values := Issued.install p w.ordinal clock other (code.schedule (thread w) (n+4)) prior g.shape
    let records := before ++ r.stream w site owner i stamp code.forest clock n ++ after_
    ∃ (out : State w) (v : Visit p), OwnerSites.advance inputs s ((r.calls w site owner i stamp code.forest).events clock n) = some out ∧
      out.cursor.finished = v::s.cursor.finished ∧ v.frame = (boundaryArgs w owner i).frame ∧ v.input = i ∧
      v.scoredAt = stamp ∧ v.worker = w.ordinal ∧ v.first = clock n ∧ v.last = clock (n+5+(code.forest.calls (thread w)).count) ∧
      completeVisit p timeline mc fc values records v =
        some ⟨observeRun p prior g.shape (code.accesses.map Access.instruction),i.after mc fc p.address prior⟩ := by
  have stop := finish r w i branch code.forest n
  have returned : i.arm ≠ .failed := by rw [branch]; cases r <;> simp [arm]
  obtain ⟨out,v,ran,_,_,_,visits,worker,opened,closed,scored,frame,input⟩ :=
    boundary_visit w site owner edge i returned stamp (r.deepTree code.forest) (r.shallowTree code.forest) clock time n
      (stop.symm ▸ reported) (stop.symm ▸ empty) s caller ready
  rw [stop] at closed
  refine ⟨out,v,ran,visits,frame,input,scored,worker,opened,closed,?_⟩
  let prior := timeline.readAt mc fc w.ordinal (clock n)
  have g : EndgameSearch.CacheHistory.Good mc fc p.width p.sizes p.key prior := EndgameSearch.CacheHistory.replay_good mc fc p _
  have lookup := r.collect_packets w site owner i branch stamp code clock time n prior g.shape other before after_ earlier later
  have sound := (EndgameSearch.CacheHistory.after_call i g (EndgameSearch.CacheMir.budget_fits p.size p.width p.word)).1.valid
  have audited := code.audit sound p g.shape
  have final := (EndgameSearch.CacheHistory.Payloads.complete_state i g _ audited).1
  unfold completeVisit
  rw [worker,opened,closed,input,lookup]
  exact final ▸ audited

end NeuralReversi.PvsParallel.NativeFrames.CacheCalls

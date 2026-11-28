module
public import NeuralReversi.PvsParallel.NativeFrames.CacheCalls.Deep
public import NeuralReversi.PvsParallel.NativeFrames.CacheCalls.Recursive
public import NeuralReversi.PvsParallel.NativeFrames.CacheCalls.Registers

@[expose] public section

namespace NeuralReversi.PvsParallel.NativeFrames.CacheCalls

open EndgameSearch.CacheOwner (Edge transfers)
open EndgameSearch.CacheHistory (Pool Input Local Visit Timeline)
open EndgameSearch.CacheHistory.NativeTrace (Worker)
open EndgameSearch.CacheHistory.Payloads (Values requests completeVisit observeRun)
open NativeFrameCache (EdgeEvent)

def Routine.program (r : Routine) (p : Pool) (i : Input) (mc fc) (s : Local) (branch : i.arm = r.arm) :
    Program r.node s.history (i.run mc fc p.address s.history).history := by
  cases r with
  | shallow => exact Shallow.boundaryProgram p i mc fc s branch
  | deep poll => exact Deep.boundaryProgram p i mc fc s branch

/-- The board, window, local arrays and poll observations determine the entire
    recursive program. No supplied call tree or expected write sequence is needed. -/
theorem Routine.boundary_run (r : Routine) (w : Worker p) (site owner : Nat)
    (edge : Edge.mk .pvs site .boundary .forward ∈ transfers) (i : Input) (branch : i.arm = r.arm)
    (stamp : Option Nat) (clock : Nat → Nat) (time : ∀ {a b}, a < b → clock a < clock b) (n : Nat)
    (timeline : Timeline p) (s : State w) (caller : OwnerSites.Caller s.cursor .pvs (.thread (thread w)) owner)
    (ready : s.cursor.clock ≤ clock n) (other : Values) (before after_ : List EdgeEvent)
    (blocks : List (Block p)) (ordered : blocks.Pairwise Block.earlier) (stream : List (EndgameSearch.CacheHistory.NativeTrace.Event S)) :
    let prior := timeline.readAt mc fc w.ordinal (clock n)
    let code := r.program p i mc fc prior branch
    let endIndex := n+5+(code.forest.calls (thread w)).count
    let g := EndgameSearch.CacheHistory.replay_good mc fc p
      (EndgameSearch.CacheHistory.beforeStep timeline.visits w.ordinal (clock n))
    let values := Block.registers blocks other
    let records := before ++ r.stream w site owner i stamp code.forest clock n ++ after_
    code.block w clock time n prior g.shape ∈ blocks →
    EndgameSearch.CacheHistory.NativeTrace.workerEvents w.ordinal stream =
      EndgameSearch.CacheHistory.NativeTrace.workerEvents w.ordinal records →
    (∀ k, stamp = some k → clock endIndex ≤ k) →
    (∀ j, n ≤ j → j < endIndex → inputs (clock j) = none) →
    requests before w.ordinal (clock n) (clock endIndex) = [] →
    requests after_ w.ordinal (clock n) (clock endIndex) = [] →
    ∃ (out : State w) (v : Visit p), OwnerSites.advance inputs s ((r.calls w site owner i stamp code.forest).events clock n) = some out ∧
      out.cursor.finished = v::s.cursor.finished ∧ v.frame = (boundaryArgs w owner i).frame ∧ v.input = i ∧
      v.scoredAt = stamp ∧ v.worker = w.ordinal ∧ v.first = clock n ∧ v.last = clock endIndex ∧
      completeVisit p timeline mc fc values stream v =
        some ⟨observeRun p prior g.shape (code.accesses.map Access.instruction),i.after mc fc p.address prior⟩ := by
  dsimp only
  intro member interleaved reported empty earlier later
  let prior := timeline.readAt mc fc w.ordinal (clock n)
  have good := EndgameSearch.CacheHistory.replay_good mc fc p
    (EndgameSearch.CacheHistory.beforeStep timeline.visits w.ordinal (clock n))
  let code := r.program p i mc fc prior branch
  let values := Block.registers blocks other
  obtain ⟨out,v,ran,visits,frame,input,score,worker,opened,closed,audit⟩ :=
    r.at_caller w site owner edge i branch stamp clock time n timeline code
      reported empty s caller ready values before after_ earlier later
  have installed := Block.install_registers blocks ordered other _ member
  change Issued.install p w.ordinal clock values (code.schedule (thread w) (n+4)) prior good.shape = values at installed
  rw [installed] at audit
  refine ⟨out,v,ran,visits,frame,input,score,worker,opened,closed,?_⟩
  exact (EndgameSearch.CacheHistory.Payloads.completeVisit_interleaved p timeline values stream _ v (by simpa only [worker] using interleaved)).trans audit

end NeuralReversi.PvsParallel.NativeFrames.CacheCalls

import NeuralReversi.PvsParallel.Interleaving.Joint
import NeuralReversi.PvsParallel.Interleaving.CallReturns

namespace NeuralReversi.PvsParallel.Interleaving.SolveEntry

open Joint (Tape Event SolveArgs)
open CallReturns (State)
open EndgameSearch.CacheOwner (Node)

/-- Get the activation allocated at this particular entry. Both parent frames
    must be on the same worker and at the Rust call sites for this route. -/
def atEntry (es : Tape Op) (s : State) (i : Nat) (route : Node) (site : Nat) (expected : SolveArgs) : Option Nat := do
  let .solveEnter w t pc _ args ← es[i]? | none
  guard (pc = site ∧ args = expected)
  let (a,id) ← s.calls.toList.zipIdx.find? fun (a,_) => a.entered == i
  guard (a.worker = w ∧ a.first = t ∧ a.callee = .solve ∧ a.site = pc)
  let caller ← a.parent.bind (fun j => s.calls[j]?)
  let root ← caller.parent.bind (fun j => s.calls[j]?)
  guard (caller.worker = w ∧ caller.callee = route ∧ caller.site = site ∧ root.worker = w ∧ root.callee = .root)
  return id

/-- After consuming the TT result, this worker must enter solve_root before
    another frame or register use. Other workers may enter their own roots. -/
def direct (es : Tape Op) (s : State) (receipt : Nat) (expected : SolveArgs) : Option Nat := do
  let .seedRegister w b _ ← es[receipt]? | none
  guard (b = expected.board)
  let (_,i) ← (es.zipIdx.drop (receipt+1)).find? fun (e,_) =>
    e.worker == w && match e with | .engine (.other _) => false | _ => true
  atEntry es s i .endRoot 0 expected

/-- The handoff follows the return from midgame's iterative deepening loop in
    the same root invocation. Compare the base, live rows and saved snapshot at entry. -/
def timed (es : Tape Op) (s : State) (i : Nat) (expected : SolveArgs) : Option Nat := do
  let id ← atEntry es s i .midRoot 1 expected
  let solve ← s.calls[id]?
  let prior ← (es.take i).reverse.find? fun e => e.worker == solve.worker && e.frameSample.isSome
  let .iterateEnd _ _ first board base saved rows := prior | none
  guard (board = expected.board ∧ base = expected.base ∧ some saved = expected.saved ∧ rows = expected.rows)
  guard (s.calls.any fun a => a.worker == solve.worker && a.first == first && a.callee == .iterate &&
    a.parent == solve.parent && a.site == 0)
  return id

namespace Regression

private def board : PvsBoardTree.Position := ⟨⟨0,0⟩,rfl⟩
private def args : SolveArgs := ⟨board,512,[],none,false,1,5⟩
private def enter (w t site : Nat) (n : Node) : Event Unit := .engine (.localCall w (.entry t site n .none none) none)
private def seed : Event Unit := .seedRegister 0 board (.miss 0)
private def start (w t site : Nat) : Event Unit := .solveEnter w t site .none args
private def launch (events : Tape Unit) (read : State → Option Nat) :=
  (CallReturns.captureTimed (events.map Event.frameSample)).bind read

private def directEvents : Tape Unit := [enter 0 0 0 .root,enter 0 1 0 .endRoot,seed,
  enter 1 0 0 .root,enter 1 1 0 .endRoot,start 1 2 0,start 0 2 0]
example : launch directEvents (fun s => direct directEvents s 2 args) = some 5 := by rfl
example : launch directEvents (fun s => direct directEvents s 2 {args with base := 256}) = none := by rfl
private def superseded := directEvents.take 6 ++ [seed,start 0 2 0]
example : launch superseded (fun s => direct superseded s 2 args) = none := by rfl

private def completed : PublicationMir.Iteration := ⟨[],13,2,false⟩
private def handoffArgs : SolveArgs := {args with saved := some completed,minimum := 2}
private def handoff (returnedBase : BitVec 32 := 512) : Tape Unit :=
  [enter 0 0 0 .root,enter 0 1 1 .midRoot,enter 0 2 0 .iterate,
   .iterateEnd 0 3 2 board returnedBase completed [],.solveEnter 0 4 1 .none handoffArgs]
example : launch (handoff 512) (fun s => timed (handoff 512) s 4 handoffArgs) = some 3 := by decide +kernel
example : launch (handoff 511) (fun s => timed (handoff 511) s 4 handoffArgs) = none := by rfl
example : launch (handoff 512) (fun s => timed (handoff 512) s 4 {handoffArgs with saved := none}) = none := by rfl
private def withoutIterate := (handoff 512).take 2 ++ [.solveEnter 0 2 1 .none handoffArgs]
example : launch withoutIterate (fun s => timed withoutIterate s 2 handoffArgs) = none := by rfl

end Regression

end NeuralReversi.PvsParallel.Interleaving.SolveEntry

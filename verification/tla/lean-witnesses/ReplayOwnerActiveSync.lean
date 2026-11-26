import NeuralReversi.PvsParallel.Interleaving.ActivePointer

namespace OwnerActiveSync

open Lean NeuralReversi NeuralReversi.Threading NeuralReversi.PvsParallel
open Interleaving
open AtomicWord (Event)
abbrev Tape := Joint.Tape Unit

structure Snapshot where
  phase : String
  detached : Bool
  booked : Bool
  cloned2 : Bool
  bookLock : Int
  interveners : List Nat
  ownerActiveKnownAtBook : Bool
  bookKnownAtOwnerActive : Bool
  bookKnownAtClone2 : Bool
  earlyDispatch : Bool
  deriving FromJson

private def base : CutoffVisibility.History := ⟨[],fun _ => [],[],[],fun _ _ => False⟩
private def addr : TaskRetention.Address := fun w b => some (4*w+b)
private def ptr (w t : Nat) (op : Joint.TaskWord) : Joint.Event Unit := .taskWord w t op
private def enter (w t site : Nat) (n : EndgameSearch.CacheOwner.Node) : Joint.Event Unit :=
  .engine (.localCall w (.entry t site n .none none) none)
private def leave (w t o : Nat) (n : EndgameSearch.CacheOwner.Node) : Joint.Event Unit :=
  .engine (.localCall w (.leave t o n none none) none)

private def site : Joint.Event Unit → Option Event
  | .workerLock w t .. | .splitWord w t .. | .taskWord w t .. => some ⟨w,t⟩
  | _ => none

private def require (p : Bool) (message : String) : Except String Unit :=
  unless p do throw message

private def buildTape (s : Snapshot) (mode : String) : Tape :=
  let ownerInit : Tape := [
    ptr 0 1 (.copyParent 0 none), .splitWord 0 2 (.lock 0 0), .splitWord 0 3 (.mark 0),
    ptr 0 4 (.install 0 none), ptr 0 5 (.ownerActive 0), .splitWord 0 6 (.unlock 0 0)]
  let clone1 : Tape := [
    enter 0 7 0 .active, .workerLock 0 8 0 true, ptr 0 9 (.cloneActive (some 0))] ++
    (if s.earlyDispatch then [] else [.workerLock 0 10 0 false]) ++
    [enter 0 11 0 .dispatch, enter 0 12 0 .splitSearch]
  let finish1 : Tape := if s.detached then
    [leave 0 13 12 .splitSearch, leave 0 14 11 .dispatch, leave 0 15 7 .active,
     .workerLock 0 16 0 true, ptr 0 17 (.detach 0), .workerLock 0 18 0 false]
    else []
  let inter : Tape := s.interveners.flatMap fun w =>
    [.workerLock w 100 0 true, .workerLock w 101 0 false]
  let lockTarget := if mode == "wrongLock" then 2 else 0
  let bookPart : Tape := if s.booked then
    [ptr 1 20 (.copyParent 4 none), .splitWord 1 21 (.lock 1 0), .splitWord 1 22 (.mark 0),
     ptr 1 23 (.install 4 none), ptr 1 24 (.ownerActive 4),
     .workerLock 1 25 lockTarget true, ptr 1 26 (.book 0 4),
     .workerLock 1 27 lockTarget false, .splitWord 1 28 (.unlock 1 0)]
    else []
  let clone2 : Tape := if s.cloned2 then
    [enter 0 30 0 .active, .workerLock 0 31 0 true, ptr 0 32 (.cloneActive (some 4)),
     .workerLock 0 33 0 false, enter 0 34 0 .dispatch, enter 0 35 1 .splitSearch]
    else []
  ownerInit ++ clone1 ++ finish1 ++ inter ++ bookPart ++ clone2

private def checkSplit (es : Tape) (gen : TaskAncestry.Generation) (entry : Event) :
    Option (Event × ActivePointer.Kind × Event) := do
  let ⟨_,c⟩ ← SlotSynchronization.capture base es [] (es.filterMap site)
  let o ← ActivePointer.splitDispatch c addr es gen entry
  return (o.origin.store.at_,o.origin.store.kind,o.origin.drop)

private def replay (mode : String) (trace : List Snapshot) : Except String String := do
  let some last := trace.getLast? | throw "empty TLC trace"
  let es := buildTape last mode
  let first := checkSplit es ⟨0,⟨0,3⟩⟩ ⟨0,12⟩
  let second := checkSplit es ⟨4,⟨1,22⟩⟩ ⟨0,35⟩
  match mode with
  | "step" =>
    require (last.phase == "done" && last.ownerActiveKnownAtBook && last.bookKnownAtClone2 &&
      !last.earlyDispatch) "normal trace did not establish ownerActive -> book -> clone synchronization"
    require (first == some (⟨0,5⟩,.ownerActive,⟨0,10⟩)) "ownerActive dispatch did not replay"
    require (second == some (⟨1,26⟩,.book,⟨0,33⟩)) "foreign book dispatch did not replay"
    require (checkSplit es ⟨0,⟨0,3⟩⟩ ⟨0,35⟩ == none) "wrong generation was accepted on second dispatch"
    return s!"normal: ownerActive={first.isSome}, foreignBook={second.isSome}, interveners={last.interveners}"
  | "raceBook" =>
    require (last.booked && !last.detached && !last.ownerActiveKnownAtBook && !last.bookKnownAtOwnerActive)
      "expected race between ownerActive and foreign book"
    require (first == none) "ActivePointer accepted a data race between ownerActive and foreign book"
    return "raceBook rejected by ActivePointer.splitDispatch"
  | "wrongLock" =>
    require (last.phase == "done" && last.bookLock == 2 && !last.bookKnownAtClone2)
      "expected wrong worker lock on foreign book"
    require (first == none && second == none) "ActivePointer accepted wrong worker lock"
    return "wrongLock rejected by ActivePointer.splitDispatch"
  | "earlyRelease" =>
    require last.earlyDispatch "expected dispatch before worker-lock release"
    require (first == none) "ActivePointer accepted dispatch before worker-lock release"
    return "earlyRelease rejected by ActivePointer.splitDispatch"
  | _ => throw s!"unknown mode {mode}"

def main (args : List String) : IO Unit := do
  let [mode,path] := args | throw (IO.userError "expected mode and TLC snapshots")
  let json ← IO.ofExcept (← Json.parse <$> IO.FS.readFile path)
  IO.println (← IO.ofExcept (fromJson? json >>= replay mode))

end OwnerActiveSync

def main := OwnerActiveSync.main

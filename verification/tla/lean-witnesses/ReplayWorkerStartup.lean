import NeuralReversi.PvsParallel.Interleaving.ActivePointer
import NeuralReversi.PvsParallel.Interleaving.SlotSynchronization
import NeuralReversi.PvsParallel.SplitSchedulerMir.Startup

namespace WorkerStartup

open Lean NeuralReversi NeuralReversi.Threading NeuralReversi.PvsParallel
open Interleaving SplitSchedulerMir
open AtomicWord (Event)
abbrev Tape := Joint.Tape Unit

structure Snapshot where
  phase : String
  late : Bool
  writerLock : Int
  turns : List Nat
  known : List Nat
  active : Nat
  oldClone : Nat
  selected : Nat
  address : Nat
  selectedAddress : Int
  atBook : Nat
  deriving FromJson

/-- Concrete previously retired generation. Owner and helper clock values
    deliberately disagree with the order in which their instructions occur. -/
private def old : Tape := [
  .taskWord 0 100 (.copyParent 7 none), .splitWord 0 101 (.lock 0 0),
  .splitWord 0 102 (.mark 0), .taskWord 0 103 (.install 7 none), .taskWord 0 104 (.ownerActive 7),
  .workerLock 0 105 1 true, .taskWord 0 106 (.book 1 7), .workerLock 0 107 1 false,
  .splitWord 0 108 (.unlock 0 0),
  .engine (.localCall 1 (.entry 9 0 .active .none none) none),
  .workerLock 1 10 1 true, .taskWord 1 11 (.cloneActive (some 7)),
  .workerLock 1 12 1 false, .splitWord 0 109 (.lock 0 0), .workerLock 0 110 0 true,
  .taskWord 0 111 (.detach 7), .workerLock 0 112 0 false, .splitWord 0 113 (.unlock 0 0),
  .splitWord 1 13 (.lock 0 0), .workerLock 1 14 1 true, .taskWord 1 15 (.detach 7),
  .workerLock 1 16 1 false, .splitWord 1 17 (.unlock 0 0),
  .engine (.localCall 1 (.leave 17 9 .active none none) none), .workerLock 0 114 0 true,
  .taskWord 0 115 (.restore 7 none), .workerLock 0 116 0 false, .taskWord 0 117 (.clearTask 7)]

private def reuse : Tape := [
  .taskWord 0 120 (.copyParent 7 none), .splitWord 0 121 (.lock 0 0),
  .splitWord 0 122 (.mark 0), .taskWord 0 123 (.install 7 none), .taskWord 0 124 (.ownerActive 7)]

private def fresh (late : Bool) (target : Nat) : Tape :=
  if late then [.splitWord 0 125 (.unlock 0 0), .splitWord 1 18 (.lock 0 0),
    .workerLock 1 19 target true, .taskWord 1 20 (.book 1 7), .workerLock 1 21 target false,
    .splitWord 1 22 (.unlock 0 0)]
  else [.workerLock 0 125 target true, .taskWord 0 126 (.book 1 7),
    .workerLock 0 127 target false, .splitWord 0 128 (.unlock 0 0)]

private def idleGuard (worker : Nat) : Tape :=
  let tick := if worker == 2 then 450 else 2
  [.workerLock worker tick 1 true,.workerLock worker (tick+1) 1 false]

private def enter_ : Tape := [
  .engine (.localCall 1 (.entry 29 0 .active .none none) none),
  .workerLock 1 30 1 true,
  .taskWord 1 31 (.cloneActive (some 7)), .workerLock 1 32 1 false,
  .engine (.localCall 1 (.entry 33 0 .dispatch .none none) none),
  .engine (.localCall 1 (.entry 34 0 .splitSearch .none none) none)]

private def site : Joint.Event Unit → Option Event
  | .workerLock w t .. | .splitWord w t .. | .taskWord w t .. => some ⟨w,t⟩
  | _ => none

private def addr : TaskRetention.Address :=
  fun owner bank => if owner == 0 && bank == 0 then some 7 else none

private def require (p : Bool) (message : String) : Except String Unit :=
  unless p do throw message

private def viewMark (j : TaskAncestry.Journal) (at_ : Event) : Option TaskAncestry.Generation := do
  let v ← j.clones at_
  let [g] := v.path | none
  return g

private def replay (mode : String) (trace : List Snapshot) : Except String String := do
  let first :: reborn :: booked :: rest := trace | throw "missing reuse or booking"
  let some done_ := rest.getLast? | throw "missing dispatch entry"
  require (first.phase == "reuse" && first.oldClone == 1 && first.active == 0 &&
    reborn.phase == "book" && booked.phase == "between" && done_.phase == "done") "wrong trace shape"
  let late := done_.late
  let wrong := mode == "wrongWorker"
  let stale := mode == "oldClone"
  let target := if wrong then 3 else 1
  let book : Event := if late then ⟨1,20⟩ else ⟨0,126⟩
  let turns := done_.turns
  require (turns.mergeSort == [2,3] && booked.turns.isEmpty && reborn.late == late &&
    booked.writerLock == (if wrong then 1 else 0) && booked.atBook == 2) "unexpected target or guard sequence"
  for (row,k) in rest.dropLast.zipIdx do
    require (row.phase == "between" && row.turns == turns.take (k+1)) "wrong intervening guard"
  let es := old ++ reuse ++ fresh late target ++ turns.flatMap idleGuard ++ enter_
  let some j := TaskAncestry.capture addr es |
    throw "task ancestry did not replay"
  let some earlier := viewMark j ⟨1,11⟩ | throw "missing old clone"
  let some current := viewMark j ⟨1,31⟩ | throw "missing current clone"
  require (earlier == ⟨7,⟨0,102⟩⟩ && current == ⟨7,⟨0,122⟩⟩ &&
    j.assignments ⟨0,106⟩ == some [earlier] && j.assignments book == some [current])
    "reused address did not resolve to the two book/clone generations"
  let selected := if stale then earlier else current
  require (done_.address == current.slot && done_.selectedAddress == Int.ofNat selected.slot &&
    done_.selected == (if selected == current then 2 else 1)) "generation or address mismatch"
  let slot := SlotSynchronization.operations 0 0 es
  let states := (SlotSynchronization.workers es).map fun w => (w,SlotSynchronization.workerOps w es)
  let entry : Event := ⟨1,31⟩
  let ready (e : Event) := Startup.ready slot states e entry
  let init : Event := ⟨0,124⟩
  let lockOK := Mutex.covers (SlotSynchronization.workerOps 1 es)
    [(if late then (⟨1,19⟩,⟨1,21⟩) else (⟨0,125⟩,⟨0,127⟩)),(⟨1,30⟩,⟨1,32⟩)]
  require (lockOK == !wrong && (Mutex.guards slot).isSome &&
    states.all (fun row => (Mutex.guards row.2).isSome)) "incorrect physical lock bracket"
  let [left,right] := turns | throw "missing intervening owners"
  let firstGuard (w : Nat) : Event := ⟨w,if w == 2 then 450 else 2⟩
  let lastGuard (w : Nat) : Event := ⟨w,(firstGuard w).step+1⟩
  let dropped : Event := if late then ⟨1,21⟩ else ⟨0,127⟩
  let enterLock : Event := ⟨1,30⟩
  let some transfers := SlotSynchronization.recorded es | throw "malformed physical lock index"
  let targetEdges := SlotSynchronization.lockEdges (SlotSynchronization.workerOps 1 es)
  require (targetEdges.contains (dropped,firstGuard left) == !wrong &&
    targetEdges.contains (lastGuard left,firstGuard right) && targetEdges.contains (lastGuard right,enterLock) &&
    !targetEdges.contains (dropped,enterLock) && targetEdges.all transfers.contains)
    "guard chain did not follow consecutive releases and acquires"
  -- Reconstruct reachability from unlock->next-acquire transfers and local
  -- sequencing. No independent searching-flag handoff is added to this history.
  let base : CutoffVisibility.History := ⟨[],fun _ => [],[],[],fun _ _ => False⟩
  let some ⟨_,clock⟩ := SlotSynchronization.capture base es [] (es.filterMap site) |
    throw "invalid synchronization chronology"
  let visible (from_ to_ : Event) : Bool := (FlagSegment.published clock.transfers from_ to_).isSome
  let backward := visible book ⟨1,11⟩
  require (!backward && !Startup.ready slot states book ⟨1,11⟩)
    "earlier clone accepted a subsequent booking"
  let bookingVisible := visible book entry
  let assignmentVisible := visible init entry
  let plantVisible := visible init book
  let oldVisible := visible ⟨0,106⟩ entry
  require (plantVisible && oldVisible && ready ⟨0,106⟩ && ready init == assignmentVisible &&
    ready book == bookingVisible && assignmentVisible == bookingVisible &&
    bookingVisible == (!wrong || late)) "lock publication disagrees with recorded transfers"
  let some seen := done_.known[1]? | throw "missing target worker's knowledge"
  require ((seen == 3) == bookingVisible && (seen == 1) == !bookingVisible) "TLC visibility differs"
  let slotStart := Mutex.carries slot init entry
  require (slotStart == late && booked.active == 2 && booked.atBook == done_.atBook)
    "unexpected late handoff through slot lock"
  let dispatched := ActivePointer.splitDispatch clock addr es selected ⟨1,34⟩
  let expectedAccept := !wrong && !stale
  require (dispatched.isSome == expectedAccept)
    "ActivePointer.splitDispatch disagreed with expected acceptance"
  if expectedAccept then
    let some sd := dispatched | throw "missing ActivePointer.splitDispatch result"
    require (sd.origin.store.at_ == book && sd.origin.generation == current &&
      sd.origin.drop == ⟨1,32⟩) "ActivePointer.splitDispatch returned wrong writer or generation"
    -- Reject claiming the old generation at the exact same address even though it also precedes startup.
    require ((ActivePointer.splitDispatch clock addr es earlier ⟨1,34⟩).isNone)
      "ActivePointer.splitDispatch accepted old generation at reused address"
  return s!"{if late then "late" else "initial"}, worker-lock bracket {lockOK}, current booking HB {bookingVisible}, old booking HB {oldVisible}, mark {selected.mark.step} at address {selected.slot}, activePointer {dispatched.isSome}"

def main (args : List String) : IO Unit := do
  let [mode,path] := args | throw (IO.userError "expected policy and TLC snapshots")
  let json ← IO.ofExcept (← Json.parse <$> IO.FS.readFile path)
  IO.println (← IO.ofExcept (fromJson? json >>= replay mode))

end WorkerStartup

def main := WorkerStartup.main

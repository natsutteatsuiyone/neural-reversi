import NeuralReversi.PvsParallel.Interleaving.SplitTapeReplay

namespace SplitLifecycleReplay

open Lean NeuralReversi NeuralReversi.Threading NeuralReversi.PvsParallel
open Interleaving SplitTapeReplay
open AtomicWord (Event)
open TaskAncestry (Generation)

structure Snapshot where
  scoutSrc : Int
  wideSrc : Int
  hitEpochSrc : Int
  workerAbortSrc : Int
  ownerAbortSrc : Int
  ownerScoreSrc : Int
  deriving FromJson

private def slotAddr : Nat := 10
private def addr : TaskRetention.Address := fun w b => if w == 0 && b == 0 then some slotAddr else none

private def wrappedPrev : BitVec 64 := BitVec.allOnes 32 ++ BitVec.allOnes 32
private def wrappedGen0 : BitVec 64 := SearchAbort.begin wrappedPrev
private def wrappedReq0 : BitVec 64 := SearchAbort.request wrappedGen0
private def nextGen1 : BitVec 64 := SearchAbort.begin wrappedReq0

private def site : Joint.Event Unit → Option Event
  | .epochWord w t _ | .abortWord w t _ | .helperAbortWord w t _ | .maskWord w t _
  | .splitWord w t _ | .taskWord w t _ | .workerLock w t _ _ | .alphaWord w t _ _ _
  | .ownerStore w t _ _ _ | .ownerLoad w t _ _ _ _ => some ⟨w,t⟩
  | _ => none

private def require (p : Bool) (message : String) : Except String Unit :=
  unless p do throw message

private def buildTape (s : Snapshot) : Joint.Tape Unit :=
  let hitSrc : Event := if s.hitEpochSrc == 3 then ⟨3,25⟩ else ⟨3,3⟩
  let wideAlphaSrc : Event := if s.wideSrc == 0 then ⟨0,7⟩ else ⟨2,45⟩
  let wideAlphaVal : BitVec 32 := if s.wideSrc == 0 then 0 else 256
  let ownerAbortSrc : Event := if s.ownerAbortSrc == 0 then ⟨3,2⟩ else ⟨3,60⟩
  let ownerAbortVal : BitVec 64 := if s.ownerAbortSrc == 0 then wrappedGen0 else wrappedReq0
  let ownerScoreSrc : Event := if s.ownerScoreSrc == 2 then ⟨0,91⟩ else ⟨2,46⟩
  [ .epochWord 3 0 (.init 0),
    .abortWord 3 1 (.init wrappedPrev),
    .abortWord 3 2 (.begin wrappedGen0),
    .epochWord 3 3 (.bump 1),
    .helperAbortWord 1 1 (.store 1 0 false),
    .maskWord 0 1 (.clear 0),
    .taskWord 0 2 (.copyParent slotAddr none),
    .splitWord 0 3 (.lock 0 0),
    .splitWord 0 4 (.mark 0),
    .ownerStore 0 5 0 0 (.best 0),
    .ownerStore 0 6 0 0 (.picked 0),
    .alphaWord 0 7 0 0 (.store 0),
    .taskWord 0 8 (.install slotAddr none),
    .taskWord 0 9 (.storeFlag slotAddr false),
    .taskWord 0 10 (.ownerActive slotAddr),
    .workerLock 0 11 1 true,
    .splitWord 0 12 (.maskRmw 0 0 1 true 1 3),
    .taskWord 0 13 (.book 1 slotAddr),
    .workerLock 0 14 1 false,
    .splitWord 0 15 (.unlock 0 0),
    .engine (.localCall 1 (.entry 20 0 .active .none none) none),
    .workerLock 1 21 1 true,
    .taskWord 1 22 (.cloneActive (some slotAddr)),
    .workerLock 1 23 1 false,
    .engine (.localCall 1 (.entry 24 0 .dispatch .none none) none),
    .engine (.localCall 1 (.entry 25 1 .splitSearch .none none) none),
    .epochWord 3 24 (.bump 2),
    .epochWord 3 25 (.bump 1),
    .alphaWord 1 30 0 0 (.load ⟨0,7⟩ 0),
    .taskWord 1 31 (.latch false),
    .taskWord 1 33 (.epoch ⟨1,32⟩ ⟨3,3⟩ 1 0),
    .taskWord 1 34 (.beginWalk (some slotAddr)),
    .taskWord 1 35 (.loadFlag slotAddr false ⟨0,9⟩),
    .taskWord 1 36 (.endWalk false),
    .taskWord 1 37 (.checkStop 31),
    .splitWord 2 44 (.lock 0 0),
    .alphaWord 2 45 0 0 (.store 256),
    .ownerStore 2 46 0 0 (.best 256),
    .ownerStore 2 47 0 0 (.picked 12),
    .splitWord 2 48 (.unlock 0 0),
    .splitWord 1 49 (.lock 0 0),
    .alphaWord 1 50 0 0 (.load wideAlphaSrc wideAlphaVal),
    .splitWord 1 51 (.unlock 0 0),
    .abortWord 3 60 (.request wrappedReq0),
    .taskWord 1 61 (.latch false),
    .taskWord 1 63 (.epoch ⟨1,62⟩ hitSrc 1 1),
    .abortWord 1 64 (.load ⟨3,60⟩ wrappedReq0 true),
    .taskWord 1 65 (.checkStop 61),
    .workerLock 1 68 1 true,
    .taskWord 1 69 (.detach slotAddr),
    .workerLock 1 70 1 false,
    .splitWord 1 71 (.maskRmw 0 0 1 false 3 2),
    .workerLock 0 72 0 true,
    .taskWord 0 73 (.detach slotAddr),
    .workerLock 0 74 0 false,
    .splitWord 0 75 (.maskRmw 0 0 0 false 2 0),
    .maskWord 0 76 (.shutdown 0 false),
    .maskWord 0 77 (.load 0 ⟨0,75⟩ 0 true),
    .maskWord 0 78 (.returned 0),
    .workerLock 0 79 0 true,
    .taskWord 0 80 (.restore slotAddr none),
    .workerLock 0 81 0 false,
    .taskWord 0 82 (.clearTask slotAddr),
    .taskWord 0 83 (.afterSplit none),
    .taskWord 0 84 (.endWalk false),
    .abortWord 0 85 (.load ownerAbortSrc ownerAbortVal true),
    .ownerLoad 0 86 0 0 ownerScoreSrc (.best 256),
    .ownerLoad 0 87 0 0 ⟨2,47⟩ (.picked 12),
    .maskWord 0 90 (.clear 0),
    .ownerStore 0 91 0 0 (.best 256),
    .ownerStore 0 92 0 0 (.picked 12),
    .abortWord 3 95 (.begin nextGen1)]

private def checkTape (s : Snapshot) : Option (Bool × Bool) := do
  let es := buildTape s
  let order := es.filterMap site
  let ext ← extract es [(⟨0,85⟩,⟨3,95⟩)] order
  let c := ext.chronology
  let m := ext.memory
  guard (CutoffPollReplay.flagMemoryOK c && CutoffPollReplay.epochMemoryOK c && abortMemoryOK m c)
  let gen : Generation := ⟨slotAddr,⟨0,4⟩⟩
  let _ ← ActivePointer.splitDispatch c addr es gen ⟨1,25⟩
  let alphaOps := AlphaReplay.accesses (⟨0,by decide⟩,0) es
  let alphaWs := AlphaReplay.stores alphaOps
  let alphaLoads ← (AlphaReplay.samples alphaOps).mapM fun r => r.decode (⟨0,by decide⟩,0) alphaWs
  guard (alphaLoads.all (AlphaReplay.coherent c alphaWs) && AlphaReplay.readOrder c alphaLoads)
  let bWs := bestStores 0 0 es
  let scoreRank := if s.ownerScoreSrc == 2 then 2 else 1
  guard (AlphaReplay.inOrder c bWs ⟨0,86⟩ scoreRank)
  let j ← TaskAncestry.capture addr es
  let check ← decodeOwnerCheck m j ⟨0,83⟩ (some ⟨0,85⟩) false
  let v ← j.polls ⟨1,65⟩
  let rec ← CutoffPollReplay.cacheAt m j es ⟨1,65⟩ v
  let spec : PollSpec := {
    started := fun _ => 61
    stamp := fun _ => ⟨0,9⟩
    abortBegin := ⟨3,2⟩
    abortFinal := ⟨3,60⟩
    helperClear := fun _ => some ⟨1,1⟩ }
  let pm ← pollMemoryFromTape m c j es gen check ⟨0,77⟩ spec [] ⟨1,65⟩ rec.reads
  let polled ← NativeStopReturns.poll
    (.cached (v.path.map Generation.slot) rec.cache rec.reads rec.assignment rec.localClock)
    pm.loads
  return (check.stopped, polled)

private def replay (mode : String) (trace : List Snapshot) : Except String String := do
  let some last := trace.getLast? | throw "empty TLC trace"
  let res := checkTape last
  match mode with
  | "step" =>
    require (res == some (true,true)) "normal SplitLifecycle trace was rejected by Lean"
    return s!"normal: alpha=[{last.scoutSrc},{last.wideSrc}], epochHit={last.hitEpochSrc}, abort=[{last.workerAbortSrc},{last.ownerAbortSrc}], scoreSrc={last.ownerScoreSrc}"
  | "backwardAlpha" =>
    require (res == none) "Lean accepted backward alpha load"
    return "backwardAlpha rejected by Lean AlphaReplay"
  | "reusedEpoch" =>
    require (res == none) "Lean accepted reused 64-bit cutoff_epoch hit"
    return "reusedEpoch rejected by Lean CutoffPollReplay.cacheAt"
  | "missedWrappedAbort" =>
    require (res == none) "Lean accepted missed wrapped generation-0 abort"
    return "missedWrappedAbort rejected by Lean SplitTapeReplay.abortMemoryOK"
  | "reuseScoreRead" =>
    require (res == none) "Lean accepted post-reuse identical-value score read"
    return "reuseScoreRead rejected by Lean AlphaReplay.inOrder"
  | _ => throw s!"unknown mode {mode}"

def main (args : List String) : IO Unit := do
  let [mode,path] := args | throw (IO.userError "expected mode and TLC snapshots")
  let json ← IO.ofExcept (← Json.parse <$> IO.FS.readFile path)
  IO.println (← IO.ofExcept (fromJson? json >>= replay mode))

end SplitLifecycleReplay

def main := SplitLifecycleReplay.main

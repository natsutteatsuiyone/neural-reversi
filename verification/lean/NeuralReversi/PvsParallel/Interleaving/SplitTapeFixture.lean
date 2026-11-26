import NeuralReversi.PvsParallel.Interleaving.SplitTapeReplay

namespace NeuralReversi.PvsParallel.Interleaving.SplitTapeFixture

open Threading
open AtomicWord (Event Write)
open TaskAncestry (Generation)
open SplitTapeReplay (extract decodeOwnerCheck pollMemoryFromTape PollSpec abortMemoryOK)

private def slotAddr : Nat := 10
private def addr : TaskRetention.Address := fun w b => if w == 0 && b == 0 then some slotAddr else none

private def wrappedPrev : BitVec 64 := BitVec.allOnes 32 ++ BitVec.allOnes 32
private def wrappedGen0 : BitVec 64 := SearchAbort.begin wrappedPrev
private def wrappedReq0 : BitVec 64 := SearchAbort.request wrappedGen0
private def nextGen1 : BitVec 64 := SearchAbort.begin wrappedReq0

-- Sanity check: 32-bit generation wraps from 0xFFFFFFFF to 0 and clears Stopped;
-- requesting abort in generation 0 sets Stopped.
example :
    SearchAbort.generation wrappedPrev = BitVec.allOnes 32 ∧
    SearchAbort.generation wrappedGen0 = 0#32 ∧
    SearchAbort.requested wrappedGen0 = BitVec.allOnes 32 ∧
    !decide (SearchAbort.Stopped wrappedGen0) ∧
    decide (SearchAbort.Stopped wrappedReq0) := by
  decide

/-- ONE unified instruction tape combining:
    - 32-bit AbortState wrap (0xFFFFFFFF -> 0) and external abort request in gen 0,
    - split init (`clear`, `mark`, `seed`, `seedMove`, `seedAlpha`, `ownerActive`),
    - helper 1 booking + ActivePointer clone + lock release + dispatch,
    - worker 1 scout alpha load (0), concurrent worker 2 alpha update (+256),
      worker 1 wide re-search alpha load (+256),
    - worker 1 cutoff_epoch miss + hit, then non-flag AbortState stop poll,
    - helper_job_aborted clear on worker 1,
    - final participation mask bit release -> owner acquire zero -> owner post-wait
      ancestor scan & AbortState load -> immediate slot reuse writing identical values. -/
private def unifiedTape (reuseEpochHit backwardAlpha badOwnerAbort : Bool) : Joint.Tape Unit :=
  let hitSrc : Event := if reuseEpochHit then ⟨3,25⟩ else ⟨3,3⟩
  let wideAlphaSrc : Event := if backwardAlpha then ⟨0,7⟩ else ⟨2,45⟩
  let wideAlphaVal : BitVec 32 := if backwardAlpha then 0 else 256
  let ownerAbortSrc : Event := if badOwnerAbort then ⟨3,2⟩ else ⟨3,60⟩
  let ownerAbortVal : BitVec 64 := if badOwnerAbort then wrappedGen0 else wrappedReq0
  [ -- Global initializers & 32-bit wrapped AbortState generation 0
    .epochWord 3 0 (.init 0),
    .abortWord 3 1 (.init wrappedPrev),
    .abortWord 3 2 (.begin wrappedGen0),
    .epochWord 3 3 (.bump 1),
    .helperAbortWord 1 1 (.store 1 0 false),
    -- Owner 0 initializes split slot (0,0) at flag address `slotAddr`
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
    -- Owner books worker 1 under worker 1 state lock and split lock
    .workerLock 0 11 1 true,
    .splitWord 0 12 (.maskRmw 0 0 1 true 1 3),
    .taskWord 0 13 (.book 1 slotAddr),
    .workerLock 0 14 1 false,
    .splitWord 0 15 (.unlock 0 0),
    -- Worker 1 clones active pointer under worker 1 lock, releases, and dispatches
    .engine (.localCall 1 (.entry 20 0 .active .none none) none),
    .workerLock 1 21 1 true,
    .taskWord 1 22 (.cloneActive (some slotAddr)),
    .workerLock 1 23 1 false,
    .engine (.localCall 1 (.entry 24 0 .dispatch .none none) none),
    .engine (.localCall 1 (.entry 25 1 .splitSearch .none none) none),
    -- Concurrent epoch reuse writes (value 2, then reused value 1) on thread 3
    .epochWord 3 24 (.bump 2),
    .epochWord 3 25 (.bump 1),
    -- Worker 1 scout alpha load reads initial 0
    .alphaWord 1 30 0 0 (.load ⟨0,7⟩ 0),
    -- Worker 1 first cutoff poll: miss on epoch 1 (at ⟨3,3⟩), flag is false
    .taskWord 1 31 (.latch false),
    .taskWord 1 33 (.epoch ⟨1,32⟩ ⟨3,3⟩ 1 0),
    .taskWord 1 34 (.beginWalk (some slotAddr)),
    .taskWord 1 35 (.loadFlag slotAddr false ⟨0,9⟩),
    .taskWord 1 36 (.endWalk false),
    .taskWord 1 37 (.checkStop 31),
    -- Worker 2 acquires split lock, updates alpha & best_score to +256, releases
    .splitWord 2 44 (.lock 0 0),
    .alphaWord 2 45 0 0 (.store 256),
    .ownerStore 2 46 0 0 (.best 256),
    .ownerStore 2 47 0 0 (.picked 12),
    .splitWord 2 48 (.unlock 0 0),
    -- Worker 1 synchronizes on split lock, then does wide re-search alpha load
    .splitWord 1 49 (.lock 0 0),
    .alphaWord 1 50 0 0 (.load wideAlphaSrc wideAlphaVal),
    .splitWord 1 51 (.unlock 0 0),
    -- External thread 3 requests abort in wrapped generation 0
    .abortWord 3 60 (.request wrappedReq0),
    -- Worker 1 second poll: epoch hit (flag still false), then AbortState acquire load sees Stopped
    .taskWord 1 61 (.latch false),
    .taskWord 1 63 (.epoch ⟨1,62⟩ hitSrc 1 1),
    .abortWord 1 64 (.load ⟨3,60⟩ wrappedReq0 true),
    .taskWord 1 65 (.checkStop 61),
    -- Worker 1 detaches and releases participation mask bit; owner releases its bit & acquires zero
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
    -- Owner post-wait checks: restore, clearTask, afterSplit ancestor scan, AbortState load, owner score/move reads
    .workerLock 0 79 0 true,
    .taskWord 0 80 (.restore slotAddr none),
    .workerLock 0 81 0 false,
    .taskWord 0 82 (.clearTask slotAddr),
    .taskWord 0 83 (.afterSplit none),
    .taskWord 0 84 (.endWalk false),
    .abortWord 0 85 (.load ownerAbortSrc ownerAbortVal true),
    .ownerLoad 0 86 0 0 ⟨2,46⟩ (.best 256),
    .ownerLoad 0 87 0 0 ⟨2,47⟩ (.picked 12),
    -- Immediate slot reuse writing IDENTICAL values (256, 12) + next abort generation 1
    .maskWord 0 90 (.clear 0),
    .ownerStore 0 91 0 0 (.best 256),
    .ownerStore 0 92 0 0 (.picked 12),
    .abortWord 3 95 (.begin nextGen1)]

private def site : Joint.Event Unit → Option Event
  | .epochWord w t _ | .abortWord w t _ | .helperAbortWord w t _ | .maskWord w t _
  | .splitWord w t _ | .taskWord w t _ | .workerLock w t _ _ | .alphaWord w t _ _ _
  | .ownerStore w t _ _ _ | .ownerLoad w t _ _ _ _ => some ⟨w,t⟩
  | _ => none

private def checkAll (reuseEpochHit backwardAlpha badOwnerAbort reuseOwnerScore : Bool) :
    Option (Bool × Bool × Bool) := do
  let es := unifiedTape reuseEpochHit backwardAlpha badOwnerAbort
  let order := es.filterMap site
  -- Explicit HB from owner 0's post-wait abort load (0,85) to next search's begin_search (3,95)
  let ext ← extract es [(⟨0,85⟩,⟨3,95⟩)] order
  let c := ext.chronology
  let m := ext.memory
  guard (CutoffPollReplay.flagMemoryOK c && CutoffPollReplay.epochMemoryOK c && abortMemoryOK m c)
  -- Task 1: ActivePointer dispatch resolves worker 1's splitSearch entry to current generation
  let gen : Generation := ⟨slotAddr,⟨0,4⟩⟩
  let sd ← ActivePointer.splitDispatch c addr es gen ⟨1,25⟩
  guard (sd.origin.store.at_ == ⟨0,13⟩ && sd.origin.drop == ⟨1,23⟩)
  -- Task 2a: AlphaReplay checks scout (0) -> remote store (+256) -> wide (+256) coherence & readOrder
  let alphaOps := AlphaReplay.accesses (⟨0,by decide⟩,0) es
  let alphaWs := AlphaReplay.stores alphaOps
  let alphaLoads ← (AlphaReplay.samples alphaOps).mapM fun r => r.decode (⟨0,by decide⟩,0) alphaWs
  guard (alphaLoads.all (AlphaReplay.coherent c alphaWs) && AlphaReplay.readOrder c alphaLoads)
  -- Task 2b: Owner best_score readback rejects reading the identical value 256 from immediate slot reuse
  let bWs := SplitTapeReplay.bestStores 0 0 es
  guard (AlphaReplay.inOrder c bWs ⟨0,86⟩ (if reuseOwnerScore then 2 else 1))
  -- Task 2c: EpochHit + OwnerCheck + AbortEra (across 32-bit wrap) + PollMemory directly from ONE tape
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
  return (check.stopped, polled, rec.hits.length == 1)

-- Valid unified tape: owner observes stopped=true across 32-bit wrap, worker poll returns true,
-- epoch hit is attributed to exact write source (1 hit), alpha scout->update->wide passes,
-- ActivePointer dispatch passes, and immediate slot reuse with identical values is separated.
#guard checkAll false false false false == some (true,true,true)

-- Mutants: reused 64-bit epoch hit, backward alpha load, missed wrapped abort on owner,
-- and post-reuse identical-value score read are all rejected.
#guard checkAll true false false false == none
#guard checkAll false true false false == none
#guard checkAll false false true false == none
#guard checkAll false false false true == none

end NeuralReversi.PvsParallel.Interleaving.SplitTapeFixture

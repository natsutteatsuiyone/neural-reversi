import NeuralReversi.PvsParallel.NativeStopReturns.Check

namespace NeuralReversi.PvsParallel.NativeStopReturns.Regression

open Threading
open NativeFrames (Record Args)
open CutoffVisibility (FlagRead FlagWrite History)
open AtomicWord (Event)

private def flags (_ : Nat) : List FlagWrite := [⟨⟨0,1⟩,true⟩]
private def flag : FlagRead flags := ⟨7,⟨1,9⟩,⟨0,by decide⟩⟩
private def epochs : List AtomicWord.Write := [⟨⟨0,0⟩,0,false,false⟩,⟨⟨0,2⟩,1,true,true⟩]
private def reading : AtomicWord.Read epochs := ⟨⟨1,8⟩,⟨1,by decide⟩,true⟩
private def memory : Memory := ⟨⟨epochs,flags,[reading],[flag],fun _ _ => False⟩,[],[],fun _ => [],fun _ => []⟩
private def scan : SplitReturn.Observation.AncestorScan memory.history :=
  ⟨⟨1,8⟩,[7],[flag],.found flag [] rfl,by intro r h; obtain rfl := List.mem_singleton.mp h; exact ⟨h,rfl,by decide⟩⟩
private def post : ChainRead memory ⟨1,10⟩ :=
  .postJoin scan (by intro r h; obtain rfl := List.mem_singleton.mp h; exact ⟨rfl,by decide⟩)
private def loads (t : Event) : Loads memory t := ⟨t.step-2,none,none,Nat.sub_le ..,by simp,by simp⟩

private def enter (step site : Nat) : Record := .localCall 1 (.entry step site .pvs .none none)
private def leave (step opening : Nat) (raw : ScoreRefresh.Score) : Record := .localCall 1 (.discarded step opening raw)
private def inner : FrameEnd := ⟨1,6,12,[307,308,309,435,438,439,441,443,446,447],⟨0⟩⟩
private def outer : FrameEnd := ⟨1,4,15,[60,61,62,63,442,443,446,447],⟨0⟩⟩
private def halt : Trace memory := .stop inner ⟨1,10⟩ post (loads _)
private def passTrace : Trace memory := .pass outer halt
private def sources : Evidence memory := fun w first tail =>
  if w != 1 then none else if first=6 ∧ tail=12 then some halt else if first=4 ∧ tail=15 then some passTrace else none
private def nested (site : Nat) (extra : List Record) : Inputs :=
  ⟨fun _ _ => none,fun _ _ => none,[enter 4 0,enter 6 site] ++ extra ++ [leave 12 6 ⟨0⟩,leave 15 4 ⟨0⟩],true⟩

#guard check (nested passSite []) sources
#guard !check (nested (passSite+1) []) sources
#guard !check (nested passSite []) (fun _ _ _ => none : Evidence memory)
example (e : ConcurrentReplay.Event) : normalize (nested passSite [.table e]) sources =
    some (nested passSite [.table e]).stream := by rfl

private def drawn : Board.Board := ⟨0x5555555555555555,0xaaaaaaaaaaaaaaaa⟩
private def solved : FrameEnd := ⟨1,1,4,[65,66,442,443,446,447],⟨0⟩⟩
private def ordinary : Inputs :=
  ⟨fun _ _ => some ⟨drawn,0,⟨-1⟩,⟨0⟩,true,Cutoffs.exactSelectivity,false,false,none⟩,
   fun _ _ => some 0,[enter 1 0,.localCall 1 (.leave 4 1 .pvs none (some ⟨0⟩))],true⟩
#guard (Trace.terminal solved : Trace memory).decode ordinary == some ⟨⟨0⟩,true⟩
#guard (Trace.stop solved ⟨1,10⟩ post (loads _) : Trace memory).decode ordinary == none
#guard !check {ordinary with stream := [enter 1 0,leave 4 1 ⟨0⟩]} (fun _ _ _ => some (.terminal solved) : Evidence memory)

private abbrev mc : Bitboard.Moves.X86Config := ⟨false,false⟩
private abbrev fc : Cutoffs.FlipConfig := ⟨false,false,false⟩
private def pool : EndgameSearch.CacheHistory.Pool := ⟨2,64,by decide,fun _ => 0,0,⟨100,101,102⟩,⟨110,111,112⟩⟩
private def w : EndgameSearch.CacheHistory.NativeTrace.Worker pool := ⟨1,.worker,by decide⟩
private def board : PvsBoardTree.Position := ⟨⟨0x84f88080808080,0x3077f7f7f7f7f⟩,rfl⟩
private def input : EndgameSearch.CacheHistory.Input :=
  let mask := Board.emptySquares board.val
  let empty := EndgameSearch.Cancellation.EmptyState.mk (EmptyListInit.newLinks mask)
    (EmptyListParity.fromEmpties mask) (Board.emptyCount board.val)
  let vacant : TranspositionTable.ProbeRetry.RawAttempt :=
    ⟨fun slot => if slot = 0 then some ((0,0),TranspositionTable.Packing.zeroData) else none,0⟩
  ⟨board,empty,by
     simpa only [EndgameSearch.Cancellation.EmptyState.matches,empty,mask] using EndgameSearch.emptyState_initial board,
   ⟨-1⟩,⟨0⟩,12,Cutoffs.exactSelectivity,[vacant],⟨⟨0,false⟩,fun _ => ⟨1,[true]⟩,0⟩,true,true,false,true,.other,false,false⟩

private def cacheRecords : List Record :=
  let go := fun t n => Record.localCall 1 (.entry t 0 n w.root none)
  [go 0 .active,go 1 .dispatch,go 2 .splitSearch,go 3 .pvs,
   .localCall 1 (.entry 4 DeepLeafMir.boundaryPvsSite .boundary w.root (some ⟨8,input⟩)),
   go 5 .access,.localCall 1 (.leave 6 5 .access none none),
   .localCall 1 (.entry 7 1 .deep (.pair ⟨3,4⟩) none),
   .localCall 1 (.entry 8 7 .deepChild (.pair ⟨3,4⟩) none),
   .localCall 1 (.leave 12 8 .deepChild none none),.localCall 1 (.leave 13 7 .deep none none),
   .localCall 1 (.leave 18 4 .boundary none none),leave 20 3 ⟨0⟩]
private def leafInputs : Inputs :=
  ⟨fun n t => if n=1 ∧ t=3 then some ⟨board.val,12,⟨-1⟩,⟨0⟩,true,Cutoffs.exactSelectivity,false,false,none⟩ else none,
   fun _ _ => none,cacheRecords,true⟩

private def attempt : CutoffVisibility.PollRead memory.history (SplitCutoff.reset 1) :=
  ⟨reading,List.mem_singleton_self _,rfl,⟨[7],[flag],scan.afterStart⟩,fun _ => scan.walk⟩
private def cached : ChainRead memory ⟨1,10⟩ := .cached [7] ⟨1,true⟩ [flag]
  (.miss (.reset [7] 1) rfl attempt (by decide +revert) rfl)
  (by intro r hr; obtain rfl := List.mem_singleton.mp hr; exact ⟨rfl,by decide⟩)
private def leaf (site : LeafSite) : Leaf :=
  ⟨pool,mc,fc,fun _ _ => none,
   ⟨fun n t => if n=1 ∧ t=18 then some ⟨⟨0⟩,some ⟨0⟩⟩ else none,
    fun n t => if n=1 ∧ t=13 then some ⟨0⟩ else none⟩,4,18,site⟩
private def returnedLeaf (s : LeafSite) := (Trace.boundary ⟨1,3,20,boundaryPcs,⟨0⟩⟩
  (leaf s) ⟨1,10⟩ cached (loads _)).decode leafInputs
set_option maxRecDepth 8192 in
#guard returnedLeaf (.recursive .deepChild 8 0) == some ⟨⟨0⟩,false⟩
#guard returnedLeaf (.recursive .deep 7 46) == none
#guard returnedLeaf (.storeGuard boundaryPollPc ⟨4,nodesBeforePc,⟨1,5⟩,17⟩ ⟨4,nodesAfterPc,⟨1,13⟩,17⟩) == none
-- Relative to the extracted threshold, so that tuning it needs no edit here.
private def minNodes : BitVec 64 := BitVec.ofNat 64 DeepLeafMir.boundaryStoreMinNodes
#guard !DeepLeafMir.enoughNodes false 37 (37 + minNodes - 1) && DeepLeafMir.enoughNodes false 37 (37 + minNodes)
#guard DeepLeafMir.enoughNodes false (0 - minNodes) 0 && !DeepLeafMir.enoughNodes true (0 - minNodes) 0

private def quick (a : Int) := Id.run do
  let i := {input with alpha := ⟨BitVec.ofInt 32 (256*a)⟩,beta := ⟨BitVec.ofInt 32 (256*a+1)⟩}
  let out := EndgameSearch.DeepRecursionMir.run i.target i.bmi2 i.paired mc fc pool.address board a (i.state [])
  let res := NativeLeafBranches.result mc fc pool.address i [] i.arm
  return (a,out.score,out.state.polls.cursor, out.state.polls.cache.latched,out.state.polls.occurred,
    (res.map (fun r => ((r.operand.score true).map (·.raw.toInt),r.useScore))))
#guard quick 50 == (50,some 8,0,false,true,some (some 2048,true))
#guard quick (-1) == (-1,none,1,true,true,some (some 0,false))
#guard (EndgameStability.tryCutoff board.val (Board.emptyCount board.val).toNat ⟨50⟩).map (·.value) == some 8

private def workerScope : List EndgameSearch.CacheHistory.NativeTrace.Raw :=
  [.enter 2 0 .splitSearch .none none, .enter 4 0 .pvs .none none, .leave 12 4 .pvs none]
#guard retainedWorker workerScope 9 14
#guard !retainedWorker (workerScope ++ [.enter 13 5 .split .none none,.leave 15 13 .split none]) 9 17
#guard !retainedWorker (workerScope ++ [.leave 13 2 .splitSearch none,.enter 15 0 .splitSearch .none none]) 9 17

private def priorInput := {input with alpha := ⟨12800⟩, beta := ⟨12801⟩, polls := ⟨⟨1,true⟩,fun _ => ⟨1,[]⟩,0⟩}

private def priorRecords : List Record :=
  let go := fun t n => Record.localCall 1 (.entry t 0 n w.root none)
  [go 0 .active,go 1 .dispatch,go 2 .splitSearch,go 3 .pvs,.localCall 1 (.leave 12 3 .pvs none none),
   .localCall 1 (.entry 20 1 .pvs w.root none),
   .localCall 1 (.entry 21 DeepLeafMir.boundaryPvsSite .boundary w.root (some ⟨42,priorInput⟩)),
   go 22 .access,.localCall 1 (.leave 23 22 .access none none),
   .localCall 1 (.entry 24 1 .deep (.pair ⟨3,4⟩) none),.localCall 1 (.leave 25 24 .deep none none),
   .localCall 1 (.leave 26 21 .boundary none none),leave 27 20 ⟨2048⟩]

private def priorArgs (worker step : Nat) : Option Args :=
  if worker=1 ∧ (step=3 ∨ step=20) then some ⟨board.val,12,⟨12800⟩,⟨12801⟩,true,Cutoffs.exactSelectivity,false,false,none⟩
  else none
private def prelatched (readAt : Event) := (Trace.boundary ⟨1,20,27,boundaryPcs,⟨2048⟩⟩
  ⟨pool,mc,fc,fun _ _ => none,⟨fun n t => if n=1 ∧ t=26 then some ⟨⟨2048⟩,some ⟨8⟩⟩ else none,
    fun n t => if n=1 ∧ t=25 then some ⟨8⟩ else none⟩,21,26,.inherited readAt⟩
  ⟨1,10⟩ cached (loads _)).decode ⟨priorArgs,fun _ _ => none,priorRecords,true⟩
#guard prelatched ⟨1,9⟩ == some ⟨⟨2048⟩,false⟩
#guard prelatched ⟨0,9⟩ == none

end NeuralReversi.PvsParallel.NativeStopReturns.Regression

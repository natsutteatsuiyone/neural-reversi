module
public import NeuralReversi.PvsParallel.NativeLeafBranches
public import NeuralReversi.PvsParallel.NativeWorkerFeed
public import NeuralReversi.Perft
meta import NeuralReversi.Perft
meta import NeuralReversi.PvsParallel.NativeLeafBranches
meta import NeuralReversi.PvsParallel.NativeWorkerFeed
meta import NeuralReversi.ScoreRefresh.Model

@[expose] public section

namespace NeuralReversi.PvsParallel.NativeLeafBranches.Regression

open PvsBoardTree (Position legalSquares play pass)
open NativeFrames (Record Args)
open EndgameSearch.CacheHistory (Pool Input)

private abbrev mc : Bitboard.Moves.X86Config := ⟨false,false⟩
private abbrev fc : Cutoffs.FlipConfig := ⟨false,false,false⟩
private def pool : Pool := ⟨2,64,by decide,fun _ => 0,0,⟨100,101,102⟩,⟨110,111,112⟩⟩
private def worker : EndgameSearch.CacheHistory.NativeTrace.Worker pool := ⟨1,.worker,by decide⟩

private def firstMoves : Nat → Position → Position
  | 0,b => b
  | n+1,b => if Board.emptyCount b.val = 12 then b else
      match legalSquares mc b.val with
      | [] => firstMoves n (pass b)
      | m :: _ => firstMoves n (play mc fc b m).val

private def position := firstMoves 64 Perft.root
private def draw : Position := ⟨⟨0x5555555555555555,0xaaaaaaaaaaaaaaaa⟩,rfl⟩

private def input (b : Position) (active : Bool) : Input :=
  let mask := Board.emptySquares b.val
  let empties : EndgameSearch.Cancellation.EmptyState :=
    ⟨Empties.encode mask,Board.emptyCount b.val⟩
  let vacant : TranspositionTable.ProbeRetry.RawAttempt :=
    ⟨fun slot => if slot = 0 then some ((0,0),TranspositionTable.Packing.zeroData) else none,0⟩
  ⟨b,empties,by
      simpa only [EndgameSearch.Cancellation.EmptyState.matches,empties,mask] using EndgameSearch.emptyState_initial b,
    ⟨BitVec.ofInt 32 (-1)⟩,⟨0⟩,empties.count,Cutoffs.exactSelectivity,if empties.count = 12 then [vacant] else [],
    ⟨⟨0,false⟩,fun _ => ⟨1,[true]⟩,0⟩,true,active,false,true,.other,false,false⟩

private def summary (i : Input) := (result mc fc pool.address i [] i.arm).map fun r =>
  ((r.operand.score true).map (·.raw.toInt),r.useScore)

#guard position.val.player == 0x84f88080808080 && position.val.opponent == 0x3077f7f7f7f7f
#guard summary (input position true) == some (some 0,false)
#guard summary (input position false) == some (some (-6144),true)
#guard summary (input draw true) == some (some 0,true)

private def start (i : Input) (site : Nat) : List Record :=
  let enter := fun t n => Record.localCall 1 (.entry t 0 n worker.root none)
  [enter 0 .active,enter 1 .dispatch,enter 2 .splitSearch,enter 3 .pvs,
    .localCall 1 (.entry 4 site .boundary worker.root (some ⟨8,i⟩)),
    enter 5 .access,.localCall 1 (.leave 6 5 .access none none)]

private def tail (value : BoundaryResult) (atEnd : Nat) := [
    TailEvent.boundary atEnd 4 (atEnd+1),.scored (atEnd+1),.pvs (atEnd+2) 3].filterMap
    (fun event => (value.event event).map (Record.localCall 1))

private def stream (i : Input) (site : Nat) (useScore deep : Bool) : List Record :=
  start i site ++ (if deep then [.localCall 1 (.entry 7 1 .deep (.pair ⟨3,4⟩) none),
    .localCall 1 (.leave 8 7 .deep none none)] else []) ++ tail ⟨⟨0⟩,useScore⟩ (if deep then 9 else 7)

private def registers (deep : Bool) (childRaw : BitVec 32) : Registers :=
  ⟨fun w step => if w=1 ∧ step=(if deep then 9 else 7) then some ⟨⟨0⟩,some ⟨0⟩⟩ else none,
   fun w step => if w=1 ∧ step=8 then some ⟨childRaw⟩ else none⟩

private def args (i : Input) (w t : Nat) : Option Args := if w=1 ∧ t=3 then
  some ⟨i.board.val,i.depth,i.alpha,i.beta,true,i.selectivity,false,false,none⟩ else none

private def observed (i : Input) (site : Nat) (useScore deep : Bool) (childRaw : BitVec 32) := do
  let records := stream i site useScore deep
  let out ← NativeFrameCache.decode pool mc fc (args i) (fun _ _ => none) records
  let v ← out.physical.timeline.visits.head?
  let got ← returned out.physical records (registers deep childRaw) mc fc v
  return (out.physical.rows.length,got.raw.raw.toInt,got.useScore)

#guard observed (input draw true) 1 true false 0 == some (1,0,true)
#guard observed (input draw true) 0 true false 0 == none
#guard observed (input position true) 1 false true 0 == some (0,0,false)
#guard observed (input position true) 1 true true 0 == some (1,0,false)
#guard observed (input position true) 1 false true 1 == none

private def audited {log : FrameReturns.Log mc fc} (table : FrameTableReads.Table log) (i : Input) (deep : Bool) :=
  let records := stream i 1 true deep
  (NativeFrameCache.decode pool mc fc (args i) (fun _ _ => none) records).map fun out =>
    checkRows table out records (registers deep 0) (endian:=false) (fun _ => none)

set_option maxRecDepth 8192 in
example {log : FrameReturns.Log mc fc} (table : FrameTableReads.Table log) :
    audited table (input draw true) false = some true := by rfl

private def scout := {input position true with alpha := ⟨0⟩, beta := ⟨1⟩}

private def scoutReturn : Option BoundaryResult := do
  let raw ← result mc fc pool.address scout [] scout.arm
  let word ← raw.operand.score true
  return ⟨word,raw.useScore⟩

private def splitCalls (scoutResult : BoundaryResult) (wide : Bool) : List Record :=
  let enter := fun n site callee => Record.localCall 1 (.entry n site callee worker.root none)
  [enter 0 0 .active,enter 1 0 .dispatch,enter 2 0 .splitSearch,enter 5 0 .pvs] ++
    ([TailEvent.scored 12,.pvs 13 5].filterMap (fun e => (scoutResult.event e).map (Record.localCall 1))) ++
    (if wide then [enter 15 1 .pvs,.localCall 1 (.scored 16 ⟨256⟩),.localCall 1 (.leave 17 15 .pvs none (some ⟨256⟩))] else [])

private def loads (wide : Bool) : Threading.AtomicWord.Event := ⟨1,if wide then 14 else 4⟩
private def ends (wide : Bool) : Threading.AtomicWord.Event := ⟨1,if wide then 17 else 13⟩
private def task : NativeWorkerFeed.Task := ⟨13,true,false,false,1024⟩
private def clock : NativeWorkerFeed.Prefix := ⟨1,2,3,none,5,15,fun
  | .load wide => loads wide | .returned q => ends q.pv | .lock => ⟨1,18⟩ | .poll => ⟨1,19⟩ | .unlock => ⟨1,20⟩⟩

private def ticket (r : BoundaryResult) (jump stopped : Bool) (alpha : BitVec 32) :
    Option (List Bool × Bool × Bool) := do
  let words ← NativeWorkerFeed.returns 1 (splitCalls r jump)
  let i := NativeWorkerFeed.received task clock words (1#32) alpha alpha
  let searched ← SplitWorkerMir.search i
  let queries := searched.steps.filterMap fun | SplitWorkerMir.Step.returned q => some q.pv | _ => none
  return (queries,NativeWorkerFeed.completeControl clock task words alpha loads ends ⟨1,18⟩ stopped,
    (NativeWorkerFeed.credit i ⟨stopped,alpha,BitVec.ofInt 32 (-32767)⟩ 7).isSome)

#guard scoutReturn == some ⟨⟨0⟩,false⟩
#guard (scoutReturn.bind fun r => ticket r true true (BitVec.ofInt 32 (-1))) == some ([false,true],true,false)
#guard (scoutReturn.bind fun r => ticket r false true (BitVec.ofInt 32 (-1))) == some ([false,true],false,false)
#guard (scoutReturn.bind fun r => ticket r true false (BitVec.ofInt 32 (-1))).map (·.2.1) == some false
#guard (scoutReturn.bind fun r => ticket r true true 0).map (·.2.1) == some false
#guard ticket ⟨⟨0⟩,true⟩ true false (BitVec.ofInt 32 (-1)) == some ([false,true],true,true)

end NeuralReversi.PvsParallel.NativeLeafBranches.Regression

module
public import NeuralReversi.PvsParallel.NativeTableWrites
public import NeuralReversi.PvsParallel.NativeTableWrites.Schedule
public import NeuralReversi.PvsParallel.FrameTableReads.WriterWindow
public import NeuralReversi.TranspositionTable.Duplicate.Image
meta import NeuralReversi.EndgameSearch.CacheHistory.Boundary
meta import NeuralReversi.PvsParallel.Cutoffs
meta import NeuralReversi.PvsParallel.FrameTableReads
meta import NeuralReversi.PvsParallel.FrameTableReads.WriterWindow
meta import NeuralReversi.PvsParallel.NativeFrames
meta import NeuralReversi.PvsParallel.NativeTableWrites
meta import NeuralReversi.PvsParallel.NativeTableWrites.Journal
meta import NeuralReversi.PvsParallel.NativeTableWrites.Schedule
meta import NeuralReversi.TranspositionTable.Duplicate.Image
meta import NeuralReversi.TranspositionTable.Duplicate.Program
meta import NeuralReversi.TranspositionTable.ReaderMir.Program
meta import NeuralReversi.TranspositionTable.SeqLock
meta import NeuralReversi.TranspositionTable.WriterMir.SourceProof
meta import NeuralReversi.TranspositionTable.WriterMir.SourceSyntax

@[expose] public section

namespace NeuralReversi.PvsParallel.NativeTableWrites.Regression

open TranspositionTable
open EndgameSearch.CacheHistory (Pool Input)
open EndgameSearch.CacheHistory.NativeTrace (Operands Worker)
open NativeFrames (Args Record)

private def fresh := FrameTableReads.saveObservation false (fun _ => 0) 0
private def proposed := Packing.newData ⟨1⟩ none .lower 12 Cutoffs.exactSelectivity 2 true false
private def firstSave := WriterMir.saveRound false 4 5 proposed fresh

#guard (ReaderMir.run false (fun _ => 0)).actions.length == 4
#guard WriterMir.SourceOp.snapshot ∈ firstSave.ops
#guard (WriterMir.publishRound id false 0 firstSave).isSome

example : WriterMir.observations (fun _ => False) false fresh firstSave := by
  intro _ s sampled occupied
  have actual : Snapshot.read false fresh.first fresh.last fresh.sample = some ⟨0,0,0,0⟩ := by decide
  rw [actual] at sampled
  cases sampled
  exact (occupied rfl).elim

private def occupied : Save.Observation := ⟨0,0,⟨4,5,Packing.pack false proposed⟩,0⟩
private def update := WriterMir.saveRound false 4 5 proposed occupied

example : ¬WriterMir.observations (fun _ => False) false occupied update := by
  intro recorded
  obtain ⟨_,impossible,_⟩ := recorded (by decide) ⟨4,5,0,Packing.pack false proposed⟩ (by decide) (by decide)
  exact impossible

private def pool : Pool := ⟨2,64,by decide,fun _ => 0,0,⟨100,101,102⟩,⟨110,111,112⟩⟩
private def mainWorker : Worker pool := ⟨0,.main,by decide⟩
private def position : PvsBoardTree.Position := ⟨⟨0x84f88080808080,0x3077f7f7f7f7f⟩,rfl⟩

private def input : Input :=
  let mask := Board.emptySquares position.val
  let empties : EndgameSearch.Cancellation.EmptyState :=
    ⟨Empties.encode mask,Board.emptyCount position.val⟩
  ⟨position,empties,by
    simpa only [EndgameSearch.Cancellation.EmptyState.matches,empties,mask] using EndgameSearch.emptyState_initial position,
    ⟨0⟩,⟨1⟩,12,⟨2,by decide⟩,[],⟨⟨0,false⟩,fun _ => ⟨0,[]⟩,0⟩,true,false,false,true,.other,false,false⟩

private def entry (n : Nat) (node : EndgameSearch.CacheOwner.Node) (op : Option Operands := none) : Record :=
  .localCall 0 (.entry n 0 node mainWorker.root op)

private def opened : List Record :=
  [entry 0 .root,entry 1 .endRoot,entry 2 .solve,entry 3 .endWindow,entry 4 .pvs,
    .localCall 0 (.entry 5 DeepLeafMir.boundaryPvsSite .boundary mainWorker.root (some ⟨9,input⟩)),
    entry 6 .access,.localCall 0 (.leave 7 6 .access none none),.localCall 0 (.entry 8 2 .deep (.pair ⟨0,1⟩) none)]

private def pending := opened ++ [.localCall 0 (.leave 9 8 .deep none none)]
private def stamp : Stamp := ⟨⟨0,0⟩,0,10,11,12,13,14,15⟩
private def args (worker step : Nat) : Option Args :=
  if worker = 0 ∧ step = 4 then some ⟨position.val,12,input.alpha,input.beta,true,input.selectivity,false,false,none⟩ else none

#guard input.empties.count.toNat == 12
#guard (NativeFrames.decode pool args pending).map (fun c => c.rows.length) == some 0
#guard (boundaryAt pool pending stamp).map (fun (first,op) => (first,op.frame,op.input.selectivity.val)) == some (5,9,2)
#guard (boundaryAt pool opened stamp).isNone
#guard (boundaryAt pool pending {stamp with worker := 1}).isNone
#guard (boundaryAt pool (pending ++ [.localCall 0 (.leave 10 5 .boundary none none)]) {stamp with claim := 11}).isNone

private def boundaryPublication (i : Input) : Option (Int × Nat × Nat) := do
  let raw ← (i.run ⟨false,false⟩ ⟨false,false,false⟩ pool.address []).score
  let score : Score := ⟨BitVec.ofInt 32 raw⟩
  let save := FrameReturns.boundarySave i score 7 false fresh
  let (req,_) ← WriterMir.storeRound id false 0 i.board save (classify false score save.alpha save.beta)
  return (req.entry.score.toInt,req.entry.selectivity.toNat,req.entry.flags.toNat)

#guard boundaryPublication input == some (-6144,Cutoffs.exactSelectivity.val,1)
#guard (boundaryPublication {input with aborted := true}).isNone

private def writers : Journal.Calls := fun k =>
  let expected : {v : SeqLock.Word // v &&& 1 = 0} :=
    if k ≤ 1 then ⟨0xfffffffffffffffe,by decide⟩ else ⟨0,by decide⟩
  ⟨⟨if k = 2 then 1 else 0,0⟩,expected,if k = 2 then .clear else .store none position proposed,
    decide (k ≠ 1),fun pc => 10*k+pc⟩

/- Writer 1 loses its CAS. Writer 2 clears a different allocation while writer
   0 wraps the sequence. Writer 3 stores only the player field in this prefix. -/
private def writerTurns := [0,1,2,0,0,1,2,2,2,2,2,0,0,0,3,3,3].map Journal.Turn.writer
private def interleaved := Journal.schedule writers false writerTurns

#guard (Journal.stateAfter writers false (writerTurns.take 14)).idleAt writers false ⟨0,0⟩
#guard !(Journal.stateAfter writers false writerTurns).idleAt writers false ⟨0,0⟩
#guard (Journal.stateAfter writers false writerTurns).idleAt writers false ⟨1,0⟩

private def ids (a : Nat) (loc : SeqLock.Location) :=
  (Journal.mods writers interleaved ⟨a,0⟩ loc).map (Journal.Site.event writers)

#guard ids 0 .sequence == [0,5,30]
#guard ids 0 (.payload .player) == [2,32]
#guard ids 0 (.payload .opponent) == [3]
#guard ids 0 (.payload .metadata) == [4]
#guard ids 1 .sequence == [20,25]
#guard ids 1 (.payload .metadata) == [24]

/-- A boundary may interrupt a writer after its CAS. Count its pending
    publication even when the suffix contains no further claim. -/
private def windowAt (turn allocation : Nat) :=
  let pre := Journal.schedule writers false (writerTurns.take turn)
  let left := Journal.mods writers pre ⟨allocation,0⟩ .sequence
  let right := Journal.mods writers (interleaved.drop pre.length) ⟨allocation,0⟩ .sequence
  (left.getLast?.map (Journal.Site.event writers),right.findIdx Journal.isClaim,
    (right.find? Journal.isClaim).map (Journal.Site.event writers))

#guard windowAt 1 0 == (some 0,1,some 30)
#guard windowAt 14 0 == (some 5,0,some 30)
#guard windowAt 3 1 == (some 20,1,none)
#guard windowAt 0 0 == (none,0,some 0)

#guard Journal.writes true ((writers 0).program false).1[5]? == some (.sequence,0)
#guard Journal.writes false ((writers 1).program false).1[0]? == none
#guard (NativeFrames.tableEvents (interleaved.filterMap (Journal.frameRecord writers false))).map
  ConcurrentReplay.Event.id == [25,5]

private def imageBytes : Duplicate.Bytes := fun addr => BitVec.ofNat 8 addr
private def copiedSlot (i : Nat) := Duplicate.image (Duplicate.copyEntries imageBytes 64 2048 32) 2048 i

#guard Duplicate.readWord false (copiedSlot 0) (.payload .player) == 0x4746454443424140
#guard Duplicate.readWord true (copiedSlot 0) (.payload .opponent) == 0x48494a4b4c4d4e4f
#guard Duplicate.readWord false (copiedSlot 31) (.payload .metadata) == 0x3736353433323130
#guard Duplicate.readWord true (copiedSlot 31) .sequence == 0x38393a3b3c3d3e3f
#guard Duplicate.readWord false (Duplicate.image (Duplicate.zeroEntries imageBytes 2048 32) 2048 31) .sequence == 0
#guard Duplicate.clearEntry (fun _ => 0xffffffffffffffff) (.payload .metadata) == 0

end NeuralReversi.PvsParallel.NativeTableWrites.Regression

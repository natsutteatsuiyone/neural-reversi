module
public import NeuralReversi.PvsParallel.NativeFrames.CacheCalls.Lookup
public import NeuralReversi.PvsParallel.NativeFrameCache
public import NeuralReversi.Board.Hash
meta import NeuralReversi.Board.Hash
meta import NeuralReversi.Board.Model
meta import NeuralReversi.EndgameSearch.CacheHistory.Payloads
meta import NeuralReversi.EndgameSearch.CacheOwner.Syntax
meta import NeuralReversi.PvsParallel.NativeFrameCache
meta import NeuralReversi.PvsParallel.NativeFrames.CacheCalls.Lookup

@[expose] public section

namespace NeuralReversi.PvsParallel.NativeFrames.CacheCalls.Regression

open EndgameSearch.CacheHistory (Pool Input)
open EndgameSearch.CacheHistory.Payloads (Instruction)
open EndgameSearch.CacheHistory.NativeTrace (Worker)
open EndgameSearch.CacheOwner (Node)
open PvsBoardTree (Position)

private def pool : Pool := ⟨2,64,by decide,Board.Hash.hash false,0,⟨100,101,102⟩,⟨110,111,112⟩⟩
private def worker : Worker pool := ⟨1,.worker,by decide⟩
private abbrev mc : Bitboard.Moves.X86Config := ⟨false,false⟩
private abbrev fc : Cutoffs.FlipConfig := ⟨false,false,false⟩

private def input (b : Position) (active hit : Bool) : Input :=
  let mask := Board.emptySquares b.val
  let empty : EndgameSearch.Cancellation.EmptyState :=
    ⟨Empties.encode mask,Board.emptyCount b.val⟩
  let entry : TranspositionTable.EntryData := ⟨512,0,3,12,5,0,1⟩
  let attempts : List TranspositionTable.ProbeRetry.RawAttempt :=
    if hit then [⟨fun _ => some (Cutoffs.key b,entry),0⟩] else []
  ⟨b,empty,EndgameSearch.emptyState_initial b,⟨BitVec.ofInt 32 (-1)⟩,⟨0⟩,empty.count,Cutoffs.exactSelectivity,
    attempts,⟨⟨0,false⟩,fun _ => ⟨0,[]⟩,0⟩,true,active,false,true,.other,false,false⟩

private def terminal (low : BitVec 64) : Position := ⟨⟨~~~low,0⟩,by simp [Board.wellFormed]⟩
private def fixed := input (terminal 0) false false
private def five := input (terminal 31) false false
private def seven (active : Bool) := input (terminal 127) active false
private def shortcut := input (terminal 4095) false true
private def live := input ⟨⟨0xffffffffffffff81,2⟩,rfl⟩ false false

private def events (i : Input) (owner := 3) (stamp : Option Nat := none)
    (shallow : Forest .shallow := .nil) : List Event :=
  let enter := fun t node => Event.entry t 0 node worker.root none
  [enter 0 .active,enter 1 .dispatch,enter 2 .splitSearch,enter 3 .pvs] ++
    (boundary worker 1 owner i stamp .nil shallow).events id 4

private def read (i : Input) (ev : List Event) (values : EndgameSearch.CacheHistory.Payloads.Values := fun _ _ => none) :=
  let args w t := if w = 1 ∧ t = 3 then
    some ⟨i.board.val,i.depth,i.alpha,i.beta,true,i.selectivity,false,false,none⟩ else none
  NativeFrameCache.decode pool mc fc args values (ev.map (Record.localCall 1))

private def visits (i : Input) := (read i (events i)).map fun parsed =>
  parsed.physical.timeline.visits.map (fun v => (v.frame,v.first,v.last))

#guard visits fixed == some [(8,4,7)]
#guard visits five == some [(8,4,9)]
#guard visits (seven false) == some [(8,4,9)]
#guard visits (seven true) == some [(8,4,9)]
#guard visits shortcut == some [(8,4,5)]
#guard !(read fixed (events fixed 2)).isSome
#guard !(read fixed (events fixed 3 (some 6))).isSome
#guard !(read fixed ((events fixed).take 5 ++ (events fixed).drop 7)).isSome
#guard !(read (seven true) ((events (seven true)).set 7 (.entry 7 2 .deep (.pair ⟨3,4⟩) none))).isSome
#guard !(read five ((events five).set 7 (.entry 7 3 .shallow (.pair ⟨3,4⟩) none))).isSome

/-- Stop after the shallow probe returns, before its containing call returns. -/
private def partialEvents := (events live 3 none shallowLookup).take 12
private def slots (op : Instruction) := (op.observe pool pool.empty (fun t => by simp [Pool.empty,EndgameSearch.CacheState.empty])).data
private def operands : EndgameSearch.CacheHistory.Payloads.Values := fun w t =>
  if w != 1 then none else match t with
  | 8 => some ⟨9,slots (.slot .shallow (pool.key live.board.val))⟩
  | 10 => some ⟨11,slots (.probe .shallow live.board.val (ScoreRefresh.toDiscDiff live.alpha).raw)⟩
  | _ => none

#guard ((read live partialEvents operands).map fun out =>
  (out.cache.finished.length,out.cache.pending[1]?.join.map (·.ops.length))) == some (0,some 2)
#guard !(read live partialEvents (fun w t => if t = 10 then none else operands w t)).isSome
#guard !(read live partialEvents (fun w t => if t = 8 then some ⟨9,.slot (pool.key live.board.val) 0⟩ else operands w t)).isSome
#guard !(read live partialEvents (fun w t => if t = 10 then (operands w t).map ({· with returned := 12}) else operands w t)).isSome

/-- Two slot histories observed during the four-worker 16-empty solve. The
    second hit reads a negative upper bound from its packed byte. -/
private def warmed (b : Board.Board) (score : Int) := do
  let p := {pool with size := 4}
  let put := Instruction.store .shallow b (BitVec.ofInt 32 (-3)) (BitVec.ofInt 32 score)
  let get := Instruction.probe .shallow b (BitVec.ofInt 32 (-3))
  let shape t : (p.empty.cells t).size = p.sizes t := by simp [Pool.empty,EndgameSearch.CacheState.empty]
  let ops := EndgameSearch.CacheHistory.Payloads.observeRun p p.empty shape [get,put,get]
  let [miss,write,hit] := ops | none
  let .probe _ _ _ _ none := miss.data | none
  let .store _ _ _ _ _ saved := write.data | none
  let .probe _ _ _ loaded (some raw) := hit.data | none
  if loaded != saved then none else
    return (p.key b,loaded.lower.toInt,loaded.upper.toInt,raw.toInt)

#guard warmed ⟨51886142996481574,18394294980759599385⟩ 4 == some (6218505270546074495,4,65,4)
#guard warmed ⟨49891594409739814,18324794885261834521⟩ (-8) == some (5844023493259412151,-65,-8,-8)

end NeuralReversi.PvsParallel.NativeFrames.CacheCalls.Regression

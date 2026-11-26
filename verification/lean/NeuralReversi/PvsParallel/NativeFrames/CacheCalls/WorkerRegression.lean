import NeuralReversi.PvsParallel.NativeFrames.CacheCalls.Boundary
import NeuralReversi.Board.Hash

namespace NeuralReversi.PvsParallel.NativeFrames.CacheCalls.WorkerRegression

open EndgameSearch.CacheHistory (Pool Input)
open EndgameSearch.CacheHistory.NativeTrace (Worker)
open EndgameSearch.CacheHistory.Payloads (Audit Values)
open PvsBoardTree (Position)

private def pool : Pool := ⟨3,64,by decide,Board.Hash.hash false,0,⟨100,101,102⟩,⟨110,111,112⟩⟩
private def a : Worker pool := ⟨1,.worker,by decide⟩
private def b : Worker pool := ⟨2,.worker,by decide⟩
private abbrev mc : Bitboard.Moves.X86Config := ⟨false,false⟩
private abbrev fc : Cutoffs.FlipConfig := ⟨false,false,false⟩

private def input (board : Position) : Input :=
  let mask := Board.emptySquares board.val
  let empty := EndgameSearch.Cancellation.EmptyState.mk
    (EmptyListInit.newLinks mask) (EmptyListParity.fromEmpties mask) (Board.emptyCount board.val)
  ⟨board,empty,EndgameSearch.emptyState_initial board,⟨BitVec.ofInt 32 (-768)⟩,⟨BitVec.ofInt 32 (-767)⟩,
    empty.count,Cutoffs.exactSelectivity,[],⟨⟨0,false⟩,fun _ => ⟨0,[]⟩,0⟩,true,false,false,true,.other,false,false⟩

private def left := input ⟨⟨17891997908486940929,49777857980438270⟩,rfl⟩
private def right := input ⟨⟨432408240905326881,17942278238766247454⟩,rfl⟩

private def invocation (w : Worker pool) (i : Input) (arm : i.arm = .shallow) : List Record × Block pool :=
  let code := Routine.program .shallow pool i mc fc pool.empty arm
  let push := fun t node => Event.entry t 0 node w.root none
  let start := [push 0 .active,push 1 .dispatch,push 2 .splitSearch,push 3 .pvs]
  let events := start ++ ((Routine.shallow.calls w 1 3 i none code.forest).events id 4)
  (events.map (Record.localCall w.ordinal),code.block w id (fun {_ _} h => h) 4 pool.empty (pool.fresh (mc:=mc) (fc:=fc)).1.shape)

private def mixed : List Record → List Record → List Record
  | [],ys => ys
  | x::xs,[] => x::xs
  | x::xs,y::ys => x::y::mixed xs ys

/-- Names each stored board by its key rather than its slot, which depends on the cache budget. -/
private def stores (audit : Audit) := audit.ops.filterMap fun op => match op.data with
  | .store _ board _ raw .. => some (pool.key board,raw.toInt)
  | _ => none

private def replay (cut : Option Nat) (swap : Bool) := do
  let (one,x) := invocation a left rfl
  let (two,y) := invocation b right rfl
  let values := Block.registers [x,y] (fun _ _ => none)
  let read : Values := if swap then fun w => values (if w = a.ordinal then b.ordinal else a.ordinal) else values
  let args (w t : Nat) : Option Args := if t = 3 ∧ (w = a.ordinal ∨ w = b.ordinal) then
    let i := if w = a.ordinal then left else right
    some ⟨i.board.val,i.depth,i.alpha,i.beta,true,i.selectivity,false,false,none⟩ else none
  let out : NativeFrameCache.Checked pool ← NativeFrameCache.decode pool mc fc args read (mixed one (two.take (cut.getD two.length)))
  return (out.physical.timeline.visits.map EndgameSearch.CacheHistory.Visit.worker,out.cache.finished.map stores,out.cache.pending.map (Option.map stores))

/- The two workers use the same opening times for different boards. Worker 2
    can stop between its two stores while worker 1 still finishes both. -/
#guard replay none false == some
  ([1,2],[[(600936225978133602,-16),(7122331998918454071,16)],[(9481734207839425253,2),(3903768271292950320,-2)]],[none,none,none])
#guard replay (some 20) false == some
  ([1],[[(600936225978133602,-16),(7122331998918454071,16)]],[none,none,some [(9481734207839425253,2)]])
#guard !(replay none true).isSome

end NeuralReversi.PvsParallel.NativeFrames.CacheCalls.WorkerRegression

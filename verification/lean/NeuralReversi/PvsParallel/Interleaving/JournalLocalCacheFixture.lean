import NeuralReversi.PvsParallel.Interleaving.JournalBodies

namespace NeuralReversi.PvsParallel.Interleaving.JournalLocalCacheFixture

open EndgameSearch.CacheHistory (Pool Input)
open EndgameTreeCache (Store entryFor)

private def pool : Pool := ⟨1,64,by decide,Board.Hash.hash false,0,⟨100,101,102⟩,⟨110,111,112⟩⟩
private abbrev mc : Bitboard.Moves.X86Config := ⟨false,false⟩
private abbrev fc : Cutoffs.FlipConfig := ⟨false,false,false⟩

private def parent : PvsBoardTree.Position := ⟨⟨0x0e76681c28108181,0x210107e397ef7e7c⟩,rfl⟩
private def node : PvsBoardTree.Position := ⟨⟨0x004602466a1283ff,0x3f317db995ed7c00⟩,rfl⟩

private def query (b : PvsBoardTree.Position) (alpha : Int) (level : TranspositionTable.Selectivity := 1) : Input :=
  let mask := Board.emptySquares b.val
  let empty : EndgameSearch.Cancellation.EmptyState :=
    ⟨Empties.encode mask,Board.emptyCount b.val⟩
  ⟨b,empty,EndgameSearch.emptyState_initial b,⟨BitVec.ofInt 32 alpha⟩,⟨BitVec.ofInt 32 (alpha+1)⟩,
    empty.count,level,[],⟨⟨0,false⟩,fun _ => ⟨0,[]⟩,0⟩,false,false,false,true,.other,false,false⟩

private def run (b : PvsBoardTree.Position) (alpha : Int) (history : List Store) := (query b alpha).run mc fc pool.address history

/-- These were the two depth-nine boundary calls that wrote to the five-empty
    board's slot in the first public search. Their windows differ by one disc. -/
private def stored : List Store := (run parent 255 (run parent (-228) []).history).history

private def slotWrites : List Store := [⟨.shallow,⟨node.val,-1,0⟩⟩,⟨.shallow,⟨node.val,0,2⟩⟩]

-- The two depth-nine replay calls also reconstruct the slot from their full subtrees.
#guard EndgameCache.probe (entryFor pool.address stored .shallow node.val) node.val (-1) = some 2
#guard (run node (-1) stored).score = some 512

def Contract : Prop :=
  pool.address .shallow node.val = 6905 ∧ (run node (-1) []).score = some 0 ∧
  EndgameCache.probe (entryFor pool.address slotWrites .shallow node.val) node.val (-1) = some 2 ∧
  (run node (-1) slotWrites).score = some 512

set_option maxRecDepth 8192 in
set_option maxHeartbeats 800000 in
theorem from_two_searches : Contract := by
  unfold Contract
  refine ⟨?_,?_,?_,?_⟩ <;> decide +kernel

structure Replay where
  continuous : List Store := []
  restarted : List Store := []
  firstCalls : Nat := 0
  secondCalls : Nat := 0

/-- Replay every cache-using boundary in both native searches. The one-worker
    routine for zero through four empties cannot issue a local cache store. -/
private def step (state : Replay) (line : String) : Option Replay := do
  let [session,p,o,d,lo,z,level,score] := line.splitOn " " | none
  let side ← session.toNat?
  let board : Board.Board := ⟨BitVec.ofNat 64 (← p.toNat?),BitVec.ofNat 64 (← o.toNat?)⟩
  if h : board.player &&& board.opponent = 0 then
    let b : PvsBoardTree.Position := ⟨board,h⟩
    let depth ← d.toNat?
    let a ← lo.toInt?
    let beta ← z.toInt?
    let lv ← level.toNat?
    let expected ← score.toInt?
    guard (depth = (Board.emptyCount board).toNat ∧ 4 < depth ∧ depth < 12 ∧ beta = a+1)
    if valid : lv < 6 then
      let input := query b a ⟨lv,valid⟩
      let actual := input.run mc fc pool.address state.continuous
      guard (actual.score = some expected)
      if side = 0 then
        guard (state.secondCalls = 0)
        return {state with continuous := actual.history, firstCalls := state.firstCalls+1}
      else
        guard (side = 1 ∧ 0 < state.firstCalls)
        let cold := input.run mc fc pool.address state.restarted
        if state.secondCalls = 0 then
          guard (b = node ∧ expected = 512 ∧ cold.score = some 0 ∧
            EndgameCache.probe (entryFor pool.address state.continuous .shallow board) board (-1) = some 2)
        return {state with continuous := actual.history,restarted := cold.history, secondCalls := state.secondCalls+1}
    else none
  else none

def replayFile (file : System.FilePath) : IO Unit := do
  let lines ← IO.FS.lines file
  let mut state := ({} : Replay)
  for line in lines do
    let some next := step state line | throw (IO.userError s!"warm local-cache replay failed at {line}")
    state := next
  unless state.firstCalls > 0 && state.secondCalls > 0 do
    throw (IO.userError "expected cache-using calls from both searches")
  IO.println s!"Lean matched all {state.firstCalls} + {state.secondCalls} cache-using calls and reproduced the warm/cold discrepancy."

end NeuralReversi.PvsParallel.Interleaving.JournalLocalCacheFixture

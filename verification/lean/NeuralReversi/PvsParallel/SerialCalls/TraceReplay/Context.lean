import NeuralReversi.SearchContextState

namespace NeuralReversi.PvsParallel.SerialCalls.TraceReplay.Context

open SearchContextState (EmptyState)

private structure Snapshot where
  state : EmptyState
  side : Nat

private def snapshot (word : String) : Option Snapshot := do
  let [count, ply, side] ← (word.splitOn ",").mapM String.toNat? | none
  guard (count ≤ 60 && ply == 60 - count && side < 2)
  return ⟨⟨BitVec.ofNat 32 count⟩, side⟩

private def same (x y : EmptyState) : Bool := x.count == y.count

theorem same_iff (x y : EmptyState) : same x y = true ↔ x = y := by
  cases x
  cases y
  simp [same]

private def check : List String → Option Bool
  | ["root", p, o, observed] => do
    let b : Board.Board := ⟨BitVec.ofNat 64 (← p.toNat?), BitVec.ofNat 64 (← o.toNat?)⟩
    let s ← snapshot observed
    return s.side == 0 && same s.state ⟨Board.emptyCount b⟩
  | ["edge", square, before, active, restored] => do
    let p ← snapshot before
    let c ← snapshot active
    let r ← snapshot restored
    guard (p.side != c.side && p.side == r.side)
    if square == "-1" then return same p.state c.state && same p.state r.state
    let i ← square.toNat?
    guard (i < 64)
    return same (SearchContextState.remove p.state) c.state &&
      same (SearchContextState.restore c.state) r.state && same r.state p.state
  | _ => none

def run (args : List String) : IO Unit := do
  let [file] := args | throw (IO.userError "expected physical context snapshots")
  let lines := (← IO.FS.readFile file).trimAscii.toString.splitOn "\n"
  for line in lines do
    unless check (line.splitOn " ") == some true do throw (IO.userError s!"unexpected physical context: {line}")
  IO.println s!"{lines.length} root and update/undo rows matched the Lean empty count, ply and side."

end NeuralReversi.PvsParallel.SerialCalls.TraceReplay.Context

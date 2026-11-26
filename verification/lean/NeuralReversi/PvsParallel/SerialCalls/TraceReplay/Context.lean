import NeuralReversi.EndgameSearch.Cancellation

namespace NeuralReversi.PvsParallel.SerialCalls.TraceReplay.Context

open EndgameSearch.Cancellation

private structure Snapshot where
  state : EmptyState
  side : Nat

private def nodes : List Nat → Option (List EmptyListLinks.Node)
  | [] => some []
  | n :: p :: rest => do
    if hn : n < 65 then
      if hp : p < 65 then return ⟨⟨n, hn⟩, ⟨p, hp⟩⟩ :: (← nodes rest) else none
    else none
  | _ => none

private def snapshot (word : String) : Option Snapshot := do
  let count :: parity :: ply :: side :: raw ← (word.splitOn ",").mapM String.toNat? | none
  guard (count ≤ 60 && parity < 16 && ply == 60 - count && side < 2)
  let table ← nodes raw
  if h : table.length = 65 then
    let links : EmptyListLinks.Links := fun i => table[i.val]'(by simp [h])
    return ⟨⟨links, BitVec.ofNat 8 parity, BitVec.ofNat 32 count⟩, side⟩
  else none

private def same (x y : EmptyState) : Bool :=
  x.count == y.count && x.parity == y.parity &&
    (List.finRange 65).all fun i => decide (x.links i = y.links i)

theorem same_iff (x y : EmptyState) : same x y = true ↔ x = y := by
  cases x with | mk xl xp xc =>
    cases y with | mk yl yp yc =>
      simp only [same, Bool.and_eq_true, beq_iff_eq, List.all_eq_true, decide_eq_true_eq,
        List.mem_finRange, forall_const, EmptyState.mk.injEq]
      exact ⟨fun ⟨⟨count,parity⟩,links⟩ => ⟨funext links,parity,count⟩,
        fun ⟨links,parity,count⟩ => ⟨⟨count,parity⟩,congrFun links⟩⟩

private def check : List String → Option Bool
  | ["root", p, o, observed] => do
    let b : Board.Board := ⟨BitVec.ofNat 64 (← p.toNat?), BitVec.ofNat 64 (← o.toNat?)⟩
    let s ← snapshot observed
    let empties := Board.emptySquares b
    return s.side == 0 && same s.state
      ⟨EmptyListInit.newLinks empties, EmptyListParity.fromEmpties empties, Board.emptyCount b⟩
  | ["edge", square, before, active, restored] => do
    let p ← snapshot before
    let c ← snapshot active
    let r ← snapshot restored
    guard (p.side != c.side && p.side == r.side)
    if square == "-1" then return same p.state c.state && same p.state r.state
    let i ← square.toNat?
    if h : i < 64 then
      let sq : Bitboard.Square := ⟨i, h⟩
      return same (p.state.remove sq) c.state && same (c.state.restore sq) r.state && same r.state p.state
    else none
  | _ => none

def run (args : List String) : IO Unit := do
  let [file] := args | throw (IO.userError "expected physical empty-list snapshots")
  let lines := (← IO.FS.readFile file).trimAscii.toString.splitOn "\n"
  for line in lines do
    unless check (line.splitOn " ") == some true do throw (IO.userError s!"unexpected physical links: {line}")
  IO.println s!"{lines.length} root and update/undo rows matched all 65 Lean link pairs, count, parity, ply and side."

end NeuralReversi.PvsParallel.SerialCalls.TraceReplay.Context

module
public import NeuralReversi.PvsParallel.SerialCalls.Sites
public import NeuralReversi.PvsParallel.SerialCalls.TraceReplay.Context

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialCalls.TraceReplay

def position (p o : String) : Option PvsBoardTree.Position := do
  let player ← p.toNat?
  let opponent ← o.toNat?
  guard (player < 2^64 && opponent < 2^64)
  let b : Board.Board := ⟨BitVec.ofNat 64 player, BitVec.ofNat 64 opponent⟩
  if h : b.player &&& b.opponent = 0 then some ⟨b, h⟩ else none

def bit : String → Option Bool
  | "0" => some false
  | "1" => some true
  | _ => none

def children : List String → Option (List (Request × Int))
  | [] => some []
  | p::o::d::pv::cut::a::z::w::rest => do
    let board ← position p o
    let depth ← d.toNat?
    guard (depth == (Board.emptyCount board.val).toNat)
    let req : Request := ⟨board, BitVec.ofNat 32 depth, ← bit pv, ← bit cut, ← a.toInt?, ← z.toInt?⟩
    return (req, ← w.toInt?) :: (← children rest)
  | _ => none

def replay : List (Request × Int) → Program Request Int → Option Int
  | [], .pure v => some v
  | (r,w)::rest, .call issued resume => if r == issued then replay rest (resume w) else none
  | _, _ => none

theorem replay_iff (calls : List (Request × Int)) (p : Program Request Int) (word : Int) :
    replay calls p = some word ↔ Program.Runs p calls word := by
  induction calls generalizing p with
  | nil =>
    cases p with
    | pure v => exact ⟨fun eq => Option.some.inj eq ▸ .pure, fun h => by cases h; rfl⟩
    | fail | call => constructor <;> intro h <;> cases h
  | cons t tail ih =>
    obtain ⟨q,v⟩ := t
    cases p with
    | fail | pure => constructor <;> intro h <;> cases h
    | call req resume =>
      by_cases same : q = req
      · subst q
        simp only [replay, beq_self_eq_true, ite_true, ih]
        exact ⟨Program.Runs.call, fun h => by cases h; assumption⟩
      · simp only [replay, beq_eq_false_iff_ne.mpr same, Bool.false_eq_true, ite_false]
        constructor
        · intro h; contradiction
        · intro h; cases h; contradiction

def check (words : List String) : Option Bool := do
  let kind::p::o::d::pv::cut::first::sq::a::z::v::tail := words | none
  let board ← position p o
  let depth ← d.toNat?
  guard (depth == (Board.emptyCount board.val).toNat)
  let alpha ← a.toInt?
  let beta ← z.toInt?
  let wanted ← v.toInt?
  let calls ← children tail
  let moves : Bitboard.Moves.X86Config := ⟨false, false⟩
  let program ← if kind == "pass" then do
      let next := PvsBoardTree.pass board
      guard (sq == "-1" && !Board.hasLegalMoves moves board.val && Board.hasLegalMoves moves next.val)
      pure <| Sites.pass next (BitVec.ofNat 32 depth) (← bit pv) (← bit cut) alpha beta
    else do
      let i ← sq.toNat?
      if h : i < 64 then
        let square : Bitboard.Square := ⟨i, h⟩
        if legal : (Board.getMoves moves board.val).getLsbD i = true then
          let next := (PvsBoardTree.play moves ⟨false, false, false⟩ board ⟨square, legal⟩).val
          if kind == "pre" then
            pure <| Sites.presearch next (BitVec.ofNat 32 depth) (← bit cut) alpha
          else
            guard (kind == "owner")
            pure <| Sites.ownerMove next (BitVec.ofNat 32 depth) (← bit pv) (← bit cut) (← bit first) alpha beta
        else none
      else none
  guard (replay calls program == some wanted)
  -- A successful re-search transcript must fail if its second call is omitted or
  -- if its two completed calls are presented to the suspended caller in reverse order.
  if let [scout, wide] := calls then
    guard ((replay [scout] program).isNone && (replay [wide, scout] program).isNone)
  return true

def run (args : List String) : IO Unit := do
  let [path] := args | throw (IO.userError "expected trace file")
  let lines := (← IO.FS.readFile path).trimAscii.toString.splitOn "\n"
  for line in lines do
    unless check (line.splitOn " ") == some true do throw (IO.userError s!"invalid continuation: {line}")
  IO.println s!"{lines.length} Rust continuations matched Lean."

end NeuralReversi.PvsParallel.SerialCalls.TraceReplay

import NeuralReversi
import Std.Tactic.BVDecide.LRAT.Parser

open Lean NeuralReversi NeuralReversi.Bitboard NeuralReversi.Board

private def hexNibble (c : Char) : Except String Nat :=
  let n := c.toNat
  if 48 ≤ n ∧ n ≤ 57 then .ok (n - 48)
  else if 97 ≤ n ∧ n ≤ 102 then .ok (n - 87)
  else if 65 ≤ n ∧ n ≤ 70 then .ok (n - 55)
  else .error "invalid hex digit"

private def decodeHexBytes : List Char → Except String (List UInt8)
  | [] => .ok []
  | high :: low :: rest => do
    let high ← hexNibble high
    let low ← hexNibble low
    let tail ← decodeHexBytes rest
    pure (UInt8.ofNat (16 * high + low) :: tail)
  | [_] => .error "odd hex input length"

private def decodeText (encoded : String) : Except String String := do
  if encoded == "-" then return ""
  let bytes ← decodeHexBytes encoded.toList
  match String.fromUTF8? (ByteArray.mk bytes.toArray) with
  | some text => pure text
  | none => throw "invalid UTF-8 protocol input"

private def encodeText (text : String) : String :=
  String.ofList (text.toUTF8.data.toList.flatMap fun byte =>
    let digit (n : Nat) := Char.ofNat (if n < 10 then 48 + n else 87 + n)
    [digit (byte.toNat / 16), digit (byte.toNat % 16)])

private def parseNat (label field : String) : Except String Nat :=
  match field.toNat? with
  | some n => .ok n
  | none => .error s!"invalid {label}: {field}"

private def parseInt (label field : String) : Except String Int :=
  match field.toInt? with
  | some n => .ok n
  | none => .error s!"invalid {label}: {field}"

private def parseU64 (label field : String) : Except String (BitVec 64) := do
  let n ← parseNat label field
  if n < 2 ^ 64 then pure (BitVec.ofNat 64 n)
  else throw s!"out-of-range {label}: {field}"

private def parseFlag (label field : String) : Except String Bool :=
  match field with
  | "0" => .ok false
  | "1" => .ok true
  | _ => .error s!"invalid {label}: {field}"

private def parseCsv (field : String) : List String :=
  if field == "-" then [] else field.splitOn ","

private def parseSquare (label field : String) : Except String Square.Square := do
  let n ← parseNat label field
  if h : n < 64 then pure ⟨n, h⟩
  else throw s!"square out of range ({label}): {field}"

private def boolBit (b : Bool) : String := if b then "1" else "0"

private def runCaseFileWith (label : String) (check : String → Except String Unit)
    (expectedCount : Option Nat) (summary : Option String) (path : String) : IO UInt32 := do
  let cases ← IO.FS.lines path
  match expectedCount with
  | some n =>
    unless cases.size == n do
      throw (IO.userError s!"expected {n} {label} cases, got {cases.size}")
  | none =>
    if cases.isEmpty then throw (IO.userError s!"missing {label} cases")
  for line in cases do
    IO.ofExcept (check line)
  let desc := summary.getD s!"{label} cases"
  (← IO.getStdout).putStrLn s!"{cases.size} {desc}, 0 mismatches"
  return 0

private def runCaseFile (label : String) (check : String → Except String Unit) :
    String → IO UInt32 :=
  runCaseFileWith label check none none

private def runHeaderCaseFile {α : Type} (label : String)
    (parseHeader : String → Except String (α × (Nat → String)))
    (check : α → String → Except String Unit) (path : String) : IO UInt32 := do
  let lines ← IO.FS.lines path
  let some header := lines[0]? | throw (IO.userError s!"missing {label} header")
  let (cfg, formatSummary) ← IO.ofExcept (parseHeader header.trimAscii.toString)
  let cases := lines.extract 1 lines.size
  if cases.isEmpty then throw (IO.userError s!"missing {label} cases")
  for line in cases do
    IO.ofExcept (check cfg line)
  (← IO.getStdout).putStrLn (formatSummary cases.size)
  return 0

structure FlipScalarCase where
  sq : Square.Square
  masks : Fin 8 → Flip.Word
  player : Flip.Word
  opponent : Flip.Word
  expected : Flip.Word

private def parseFlipScalarCase (line : String) : Except String FlipScalarCase := do
  let normalized := line.trimAscii.toString.replace "\t" " "
  let fields := (normalized.splitOn " ").filter (· != "")
  let sqField :: wordFields := fields | throw s!"unparsable line: {line}"
  let sq ← parseSquare "flip scalar square" sqField
  let words ← wordFields.mapM (parseU64 "flip scalar word")
  let [m0, m1, m2, m3, m4, m5, m6, m7, player, opponent, expected] := words
    | throw s!"unparsable line: {line}"
  let maskVec : Vector Flip.Word 8 := ⟨#[m0, m1, m2, m3, m4, m5, m6, m7], rfl⟩
  pure { sq, masks := fun d => maskVec.get d, player, opponent, expected }

private def formatFlipOutcome : Except String Flip.Word → String
  | .ok w => toString w.toNat
  | .error msg => s!"error:{msg}"

private def runFlipScalarCases (path : String) : IO UInt32 := do
  let stderr ← IO.getStderr
  let stdout ← IO.getStdout
  let rawLines ← try
    IO.FS.lines path
  catch err =>
    stderr.putStrLn s!"failed to read {path}: {err}"
    return 2
  let lines := (rawLines.map (·.trimAscii.toString)).filter (· != "")
  let mut mismatches := 0
  for line in lines do
    let c ← match parseFlipScalarCase line with
      | .ok c => pure c
      | .error msg =>
        stderr.putStrLn msg
        return 2
    let maskMismatch? := (List.finRange 8).find? fun d => Flip.mask c.sq d != c.masks d
    let releaseGot := Flip.flip false c.sq c.player c.opponent
    let checkedGot := Flip.flip true c.sq c.player c.opponent
    let matchesExpected := fun
      | .ok w => w == c.expected
      | .error _ => false
    let releaseOk := matchesExpected releaseGot
    let checkedOk := matchesExpected checkedGot
    if maskMismatch?.isSome || !releaseOk || !checkedOk then
      mismatches := mismatches + 1
      if mismatches ≤ 10 then
        if let some d := maskMismatch? then
          stderr.putStrLn s!"square {c.sq.val}: mask[{d.val}] model {(Flip.mask c.sq d).toNat}, Rust {(c.masks d).toNat}"
        if !releaseOk then
          stderr.putStrLn s!"square {c.sq.val}, player {c.player.toNat}, opponent {c.opponent.toNat}: release model {formatFlipOutcome releaseGot}, Rust {c.expected.toNat}"
        if !checkedOk then
          stderr.putStrLn s!"square {c.sq.val}, player {c.player.toNat}, opponent {c.opponent.toNat}: checked model {formatFlipOutcome checkedGot}, Rust {c.expected.toNat}"
  stdout.putStrLn s!"{lines.size} cases, {mismatches} mismatches"
  return if mismatches == 0 && lines.size > 0 then 0 else 1

private def moveCasesHeader (backend : String) :
    Except String (Flip.Dispatch.TargetConfig × (Nat → String)) := do
  let cfg ← match backend with
    | "neon-sha3" => .ok (.aarch64 true true)
    | "neon" => .ok (.aarch64 true false)
    | "avx512" => .ok (.x86_64 ⟨false, true, false⟩)
    | "avx2" => .ok (.x86_64 ⟨false, false, true⟩)
    | "portable" => .ok .other
    | _ => .error s!"unknown move generation backend: {backend}"
  pure (cfg, fun n => s!"{backend}: {n} move generation cases, 0 mismatches")

private def checkMoveCase (cfg : Flip.Dispatch.TargetConfig) (line : String) :
    Except String Unit := do
  let ws ← (line.trimAscii.toString.splitOn " ").mapM (parseU64 "move generation word")
  let [p, o, rustLegal, rustCombined, rustPotential] := ws
    | throw s!"invalid move generation case: {line}"
  let legal := Moves.legalMovesTarget cfg p o
  let (combined, potential) := Moves.legalMovesAndPotentialTarget cfg p o
  unless (legal, combined, potential) == (rustLegal, rustCombined, rustPotential) do
    throw s!"move generation mismatch: {line}; model {legal.toNat} {combined.toNat} {potential.toNat}"

open NeuralReversi.Eval.Activations in
private def activationBackend (name : String) : Except String Backend :=
  match name with
  | "scalar" => .ok .scalar
  | "avx2" => .ok .avx2
  | "avx512" => .ok .avx512
  | "neon" => .ok .neon
  | _ => .error s!"unknown activation backend: {name}"

open NeuralReversi.Eval.Activations in
private def activationHeader (header : String) :
    Except String ((Backend × Backend) × (Nat → String)) := do
  let [fusedName, scReluName] := header.splitOn " "
    | throw "expected fused and SCReLU backend names"
  let fused ← activationBackend fusedName
  let scRelu ← activationBackend scReluName
  pure ((fused, scRelu), fun n => s!"{fusedName}/{scReluName}: {n} activation cases, 0 mismatches")

private def parseActivationValues (text : String) : Except String (Array Int) :=
  ((text.splitOn " ").filter (· != "")).toArray.mapM (parseInt "activation value")

private def parseActivationInput (text : String) : Except String (Array Int32) := do
  let values ← parseActivationValues text
  values.mapM fun n => do
    if n < -(2^31) || n ≥ 2^31 then throw s!"input outside i32 domain: {n}"
    pure (Int32.ofInt n)

open NeuralReversi.Eval.Activations in
private def checkActivationCase (fused scRelu : Backend) (line : String) : Except String Unit := do
  match line.trimAscii.toString.splitOn "|" with
  | ["fused", input, output] =>
    let xs ← parseActivationInput input
    let ys ← parseActivationValues output
    unless xs.size == 16 && ys.size == 32 do throw "wrong fused activation length"
    for i in List.finRange 32 do
      let want := fusedOutput fused (fun j => xs[j.val]!) i
      unless ys[i.val]! == want do
        throw s!"fused activation mismatch at {i.val} for [{input}]: Rust {ys[i.val]!}, model {want}"
  | ["screlu", input, output, order] =>
    let xs ← parseActivationInput input
    let ys ← parseActivationValues output
    let indices ← parseActivationValues order
    unless ys.size == xs.size && indices.size == xs.size do throw "wrong SCReLU length"
    for i in [:xs.size] do
      let wantIdx := Int.ofNat (scReluStoreOrder scRelu xs.size i)
      let want := scReluOutput scRelu xs.size (fun j => xs[j]!) i
      unless indices[i]! == wantIdx && ys[i]! == want do
        throw s!"SCReLU mismatch for size {xs.size} at {i}: Rust index {indices[i]!}, value {ys[i]!}; model index {wantIdx}, value {want}"
  | _ => throw s!"invalid activation case: {line}"

open NeuralReversi.EndgameTiming in
private def checkEndgameTimingCase (line : String) : Except String Unit := do
  let values ← (line.trimAscii.toString.splitOn " ").mapM (parseU64 "timing value")
  let [elapsed, budget, factorBits, elapsedBits, budgetBits, productBits, permits] := values
    | throw s!"invalid endgame timing case: {line}"
  let some factor := decode factorBits
    | throw s!"unsupported negative factor: {line}"
  unless decode elapsedBits = some (.finite (castUnits elapsed.toNat)) do
    throw s!"elapsed cast disagrees with the binary64 model: {line}"
  unless decode budgetBits = some (.finite (castUnits budget.toNat)) do
    throw s!"budget cast disagrees with the binary64 model: {line}"
  unless decode productBits = some (elapsedProduct elapsed.toNat factor) do
    throw s!"projected time disagrees with the binary64 model: {line}"
  unless permits.toNat = (if underBudget budget.toNat elapsed.toNat factor then 1 else 0) do
    throw s!"budget decision disagrees with the binary64 model: {line}"

private def timeModeOutput : TimeSettings.Mode → String
  | .infinite => "infinite"
  | .byoyomi time => s!"byo {time}"
  | .fischer main increment => s!"fischer {main} {increment}"
  | .japaneseByo main time => s!"japanese {main} {time}"

private def checkTimeSettingsCase (line : String) : Except String Unit := do
  let [input, fields, initial, perMove] := line.trimAscii.toString.splitOn "|"
    | throw s!"malformed time settings case: {line}"
  let parameters ← (input.splitOn " ").mapM (parseNat "time setting")
  let [main, extra, stones, remaining, left] := parameters
    | throw s!"malformed time settings input: {input}"
  let settings := TimeSettings.fromGtpSecs main extra stones
  let (bank, period, encodedStones) := TimeSettings.gtpFields settings
  let enabled := boolBit (TimeSettings.enabled settings)
  let expected := s!"{enabled} {bank} {period} {encodedStones}"
  let expectedInitial := timeModeOutput (TimeSettings.initialMode settings)
  let expectedMove := timeModeOutput (TimeSettings.moveMode remaining left settings)
  unless fields == expected && initial == expectedInitial && perMove == expectedMove do
    throw s!"time settings differ: {line}, Lean returned {expected}|{expectedInitial}|{expectedMove}"

private def fixedEndgameHeader (header : String) : Except String (Unit × (Nat → String)) := do
  let expected := String.intercalate " " (EndgameProgress.ladder.map toString)
  unless header == expected do
    throw s!"Rust endgame ladder differs: {header}, expected {expected}"
  pure ((), fun n => s!"{n} fixed endgame cases, 0 mismatches")

private def checkFixedEndgameCase (line : String) : Except String Unit := do
  let fields := line.trimAscii.toString.splitOn " "
  let [empties, firstCap, secondCap, d1, d2, d3, d4, d5, result] := fields
    | throw s!"malformed fixed endgame case: {line}"
  let values ← [empties, firstCap, secondCap, d1, d2, d3, d4, d5].mapM fun field => do
    let n ← parseNat "fixed endgame value" field
    if n < 2 ^ 32 then pure n else throw s!"outside u32 range: {field}"
  let [empties, firstCap, secondCap, d1, d2, d3, d4, d5] := values
    | throw s!"invalid fixed endgame case: {line}"
  let observed ← if result == "-" then pure none else some <$> parseNat "endgame result" result
  let depths := [d1, d2, d3, d4, d5]
  let expected := EndgameProgress.endSelectivity (min firstCap secondCap)
    (fun step => depths[step - 1]?.getD 0) empties
  unless observed == expected do
    throw s!"fixed endgame cutoff differs: {line}, Lean returned {expected}"

private def checkGameAnalysisCase (line : String) : Except String Unit := do
  let [input, outcome, progress] := line.trimAscii.toString.splitOn "|"
    | throw s!"malformed game analysis case: {line}"
  let [player, opponent, history, cancellation, mode] := input.splitOn " "
    | throw s!"malformed game analysis input: {input}"
  let p ← parseU64 "player bits" player
  let o ← parseU64 "opponent bits" opponent
  let cancelAt ← parseNat "cancellation poll" cancellation
  let moves ← (parseCsv history).mapM fun token => do
    if token == "p" then return GameAnalysis.Action.pass
    return GameAnalysis.Action.play (← parseSquare "game analysis action" token)
  let board : Board := ⟨p, o⟩
  let probe : GameAnalysis.Search :=
    fun _ searched =>
      if mode == "error" then .error "fixture"
      else
        let player := (playerCount searched).toNat
        let opponent := (opponentCount searched).toNat
        let index := 3 * player + opponent
        let first : Square.Square := ⟨index % 64, Nat.mod_lt _ (by decide)⟩
        let next : Square.Square := ⟨(index + 17) % 64, Nat.mod_lt _ (by decide)⟩
        let score : BitVec 32 := if mode == "invalid" then 0x80000000#32
          else BitVec.ofInt 32 (64 * ((player : Int) - opponent) + 17)
        .ok ⟨⟨score⟩, first, [first, next]⟩
  let model := GameAnalysis.reportGame ⟨false, false⟩ ⟨false, false, false⟩
    (fun index => probe (index.getD 0)) (fun poll => decide (cancelAt ≤ poll)) board moves
  let (expected, emitted) := match model with
    | .error (.illegalMove square) => (s!"illegal:{square.val}", "")
    | .error (.illegalMoveAt index square) => (s!"illegal:{index}:{square.val}", "")
    | .error .passWithLegalMoves => ("pass-with-moves", "")
    | .error .passAfterGameOver => ("pass-after-game", "")
    | .ok report =>
      let status := match report.outcome with
        | .complete | .cancelled => "ok"
        | .searchError message => s!"search:{message}"
        | .invalidScore raw => s!"invalid:{raw}"
      let encoded := report.progress.map fun move =>
        let pv := String.intercalate "," (move.pv.map fun square => toString square.val)
        s!"{move.index},{move.best},{move.played},{move.loss},{move.bestMove.val}:{pv}"
      (status, String.intercalate ";" encoded)
  unless outcome == expected && progress == emitted do
    throw s!"game analysis disagrees: {line}, Lean returned {expected}|{emitted}"

private def obfOption {α : Type} [ToString α] (value : Option α) : String :=
  value.map toString |>.getD "-"

private def observeObf : Except Obf.Error (Option Obf.Position) → String
  | .ok none => "blank"
  | .ok (some pos) =>
    let white := boolBit (pos.side == .white)
    let passing := boolBit (Obf.isPass pos)
    let expected := obfOption (Obf.expectedScore pos)
    let best := String.intercalate "," ((Obf.bestMoves pos).map (fun sq => toString sq.val))
    let scores := String.intercalate "," ((List.finRange 64).map fun sq =>
      s!"{obfOption (Obf.scoreOf pos sq)}:{obfOption (Obf.rankOf pos sq)}")
    s!"ok {pos.board.player.toNat} {pos.board.opponent.toNat} {white} {passing} {expected}|{best}|{scores}"
  | .error error =>
    let category := match error with
      | .nonAsciiHeader _ => "non-ascii"
      | .shortHeader _ => "short"
      | .invalidSide _ => "side"
      | .invalidBoard _ => "board"
      | .invalidSegment _ => "segment"
      | .invalidMove _ _ => "move"
      | .invalidScore _ _ => "score"
    let cause := (error.sourceMessage.map encodeText).getD "-"
    s!"error {category}|{encodeText error.message}|{cause}"

private def checkObfCase (line : String) : Except String Unit := do
  let input :: observed := line.trimAscii.toString.splitOn "|"
    | throw s!"malformed OBF comparison: {line}"
  let text ← decodeText input
  let predicted := observeObf (Obf.parse text)
  unless (String.intercalate "|" observed) == predicted do
    throw s!"OBF disagrees on {repr text}: Rust returned {String.intercalate "|" observed}, Lean returned {predicted}"

private def emptyListOutput (order : List Square.Square) : String :=
  let quadrant := EmptyListParity.quadrantId
  let parity := order.foldl EmptyListParity.toggle 0#8
  String.intercalate " " ([toString order.length, toString parity.toNat] ++
    order.map (fun square => s!"{square.val}:{(quadrant square).toNat}"))

private def checkEmptyListCase (line : String) : Except String Unit := do
  let input :: snapshots := line.trimAscii.toString.splitOn "|" | throw "missing empty list input"
  let [empty, removed] := input.splitOn " " | throw s!"malformed empty list input: {input}"
  let bits ← parseU64 "empty mask" empty
  let n ← parseNat "removed square" removed
  let order := EmptyListParity.presorted.filter (fun sq => bits.getLsbD sq.val)
  let initial := emptyListOutput order
  let expected ← if fits : n < 64 then do
    let sq : Square.Square := ⟨n, fits⟩
    unless order.contains sq do throw s!"removed square not in initial empty list: {input}"
    pure [initial, emptyListOutput (order.erase sq), initial]
  else if n == 64 && order.isEmpty then pure [initial]
  else throw s!"invalid empty-list action: {input}"
  unless snapshots == expected do
    throw s!"empty list differs for {input}: Lean returned {String.intercalate "|" expected}"

private def counterSnapshotOutput (enabled : Bool) (s : SearchCounters.Snapshot) : String :=
  let stats := if enabled then (List.finRange 10).map (fun i => (s.stats i).toNat) else []
  String.intercalate " " ((s.nodes.toNat :: stats).map toString)

private def checkCounterCase (line : String) : Except String Unit := do
  let input :: observed := line.trimAscii.toString.splitOn "|" | throw "missing counter case input"
  let [flag, nodes, otherNodes, mask, otherMask] := input.splitOn " "
    | throw s!"invalid counter case input: {input}"
  let enabled ← parseFlag "counter flag" flag
  let nodes ← parseU64 "counter nodes" nodes
  let otherNodes ← parseU64 "counter nodes" otherNodes
  let mask ← parseNat "counter mask" mask
  let otherMask ← parseNat "counter mask" otherMask
  unless mask < 1024 && otherMask < 1024 do
    throw s!"counter input out of range: {input}"
  let incremented (mask : Nat) := (List.finRange 10).foldl (fun state (i : Fin 10) =>
    if mask.testBit i.val then SearchCounters.increment enabled state i else state) (fun _ => 0)
  let current := SearchCounters.withNodes enabled nodes (incremented mask)
  let other := SearchCounters.withNodes enabled otherNodes (incremented otherMask)
  let total := SearchCounters.merge enabled current other
  let updated := SearchCounters.withNodes enabled current.nodes total.stats
  let delta := if enabled then counterSnapshotOutput enabled (SearchCounters.deltaSince total current) else "-"
  let expected := [counterSnapshotOutput enabled current, counterSnapshotOutput enabled total,
    counterSnapshotOutput enabled updated, delta]
  unless observed == expected do
    throw s!"search counters differ for {input}: Rust returned {observed}, Lean returned {expected}"

private def checkBoardHashCase (line : String) : Except String Unit := do
  let [endian, player, opponent, observed] := line.trimAscii.toString.splitOn " "
    | throw s!"invalid board hash case: {line}"
  let bigEndian ← parseFlag "board hash endianness" endian
  let board : Board := ⟨← parseU64 "board hash player" player, ← parseU64 "board hash opponent" opponent⟩
  let observed ← parseU64 "board hash value" observed
  let predicted := Board.Hash.hash bigEndian board
  unless predicted == observed do
    throw s!"board hash comparison failed: {line}, Lean returned {predicted.toNat}"

private def parseStopAction (input : String) : Except String Threading.StopToken.FixtureAction := do
  if input == "s" then return .stop
  let [verb, poolCode] := input.toList | throw s!"invalid stop action: {input}"
  let pool : Fin 2 ← match poolCode with
    | '0' => pure 0
    | '1' => pure 1
    | _ => throw s!"invalid stop pool: {input}"
  match verb with
  | 'b' => pure (.begin pool)
  | 'a' => pure (.attach pool)
  | 'd' => pure (.detach pool)
  | _ => throw s!"invalid stop action: {input}"

private def checkStopTimeline (state : Threading.StopToken.FixtureState) :
    List String → Except String Unit
  | [] => .ok ()
  | action :: bits :: rest => do
    let step ← parseStopAction action
    let next ← Threading.StopToken.fixtureStep state step
    let expected := Threading.StopToken.fixtureBits next
    unless bits.toNat? == some expected do
      throw s!"stop action {action}: native abort bits {bits}, Lean abort bits {expected}"
    checkStopTimeline next rest
  | _ => .error "missing observed stop bits"

private def checkStopTokenCase (line : String) : Except String Unit :=
  (checkStopTimeline Threading.StopToken.fixtureStart (line.trimAscii.toString.splitOn " ")).mapError
    fun error => s!"{error} in {line}"

private def checkInFlightCase (line : String) : Except String Unit := do
  let [first, second, dropFirst, observed] := line.splitOn " " |
    throw s!"malformed in-flight case: {line}"
  let a ← parseU64 "first hash" first
  let b ← parseU64 "second hash" second
  let dropA ← parseFlag "in-flight drop flag" dropFirst
  let index := Threading.InFlight.slot
  let claim := Threading.InFlight.claim
  let code : Threading.InFlight.Claim → Char
    | .owned => 'o'
    | .busy => 'b'
    | .untracked => 'u'
  let set (slots : BitVec 16 → BitVec 64) hash word :=
    fun i => if i = index hash then word else slots i
  let start : BitVec 16 → BitVec 64 := fun _ => 0
  let (wordA, claimA) := claim 0 a
  let slotsA := set start a wordA
  let (wordB, claimB) := claim (slotsA (index b)) b
  let both := set slotsA b wordB
  let dropGuard slots hash outcome :=
    if outcome = Threading.InFlight.Claim.owned then set slots hash 0 else slots
  let one := if dropA then dropGuard both a claimA else dropGuard both b claimB
  let probe slots hash := code (claim (slots (index hash)) hash).2
  let expected := String.ofList
    [code claimA, code claimB, probe both a, probe both b, probe one a, probe one b,
      probe start a, probe start b]
  unless observed == expected do
    throw s!"in-flight outcomes differ: native {observed}, Lean {expected} in {line}"

private def checkLmrCase (line : String) : Except String Unit := do
  let (expected, actual) ← match line.splitOn " " with
    | ["log", index, value] =>
      let i ← parseNat "LMR natural number" index
      unless i ≤ 60 do throw s!"invalid LMR log index: {line}"
      pure (LateMoveReduction.logAt i, ← parseNat "LMR natural number" value)
    | ["curve", depth, ordinal, value] =>
      pure (LateMoveReduction.curve (← parseNat "LMR natural number" depth)
        (← parseNat "LMR natural number" ordinal), ← parseNat "LMR natural number" value)
    | ["reduce", depth, ordinal, total, flags, alpha, value, result] =>
      let bits ← parseNat "LMR natural number" flags
      unless bits < 32 do throw s!"invalid LMR flags: {line}"
      let node : LateMoveReduction.Node :=
        ⟨← parseNat "LMR natural number" depth, ← parseNat "LMR natural number" ordinal,
          ← parseNat "LMR natural number" total, bits.testBit 4, bits.testBit 3, bits.testBit 0⟩
      let resultLean :=
        LateMoveReduction.scheduled node (bits.testBit 1) (bits.testBit 2)
          (← parseInt "LMR integer" alpha) (← parseInt "LMR integer" value)
      pure (resultLean, ← parseNat "LMR natural number" result)
    | _ => throw s!"malformed LMR case: {line}"
  unless actual == expected do
    throw s!"LMR result differs: native {actual}, Lean {expected} in {line}"

open Threading.SplitWindow in
private def checkSplitReportCase (line : String) : Except String Unit := do
  let pv :: alpha :: best :: beta :: score :: observed := line.splitOn " " |
    throw s!"malformed split report case: {line}"
  let pv ← parseFlag "split node type" pv
  let state : State := ⟨← parseInt "split report score" alpha, ← parseInt "split report score" best⟩
  let beta ← parseInt "split report score" beta
  let score ← parseInt "split report score" score
  let update := publish pv beta state score
  let expected := [toString update.alpha, toString update.best,
    boolBit (decide (raisesAlpha pv beta state score)),
    boolBit (decide (state.best < score ∧ state.alpha < score)),
    boolBit (marksCutoff pv beta state score)]
  unless observed == expected do
    throw s!"split report differs (Lean {String.intercalate " " expected}): {line}"

private def checkLargePageCase (line : String) : Except String Unit := do
  let [width, bytes, boundary, flags, allocated, requested, observed] := line.splitOn " " |
    throw s!"malformed Windows large-page case: {line}"
  let bits ← parseNat "pointer width" width
  let size ← parseNat "request size" bytes
  let page ← parseNat "page size" boundary
  let mask ← parseNat "reply flags" flags
  unless (bits == 32 || bits == 64) && size < 2 ^ bits && page < 2 ^ bits && mask < 64 do
    throw s!"out-of-range Windows large-page case: {line}"
  let reply : LargePages.WindowsReply :=
    ⟨mask.testBit 0, mask.testBit 1, mask.testBit 2, mask.testBit 3, mask.testBit 4, mask.testBit 5⟩
  let (ok, calls) := LargePages.windowsRun (LargePages.requestSize bits size page) reply true
  let actualBytes := calls.findSome? fun
    | .alloc size => some size
    | _ => none
  let code : LargePages.WindowsCall → Char
    | .openToken => 'o'
    | .lookup => 'l'
    | .enable => 'a'
    | .lastError => 'e'
    | .alloc _ => 'v'
    | .restore changed => if changed then 's' else 'r'
    | .close => 'c'
    | .free => 'f'
  let trace := if calls.isEmpty then "-" else String.ofList (calls.map code)
  unless allocated == boolBit ok &&
      requested == (actualBytes.map toString).getD "-" && observed == trace do
    throw s!"Windows large-page outcome differs: Lean {ok}, {actualBytes}, {trace} in {line}"

private def checkWeightStreamCase (line : String) : Except String Unit := do
  let fields := line.splitOn " "
  if let ["layout", network, bytes] := fields then
    let expected ← match network with
      | "main" => pure Eval.WeightStream.mainFields.sum
      | "small" => pure Eval.WeightStream.smallFields.sum
      | _ => throw s!"unknown network: {line}"
    unless bytes.toNat? == some expected do
      throw s!"weight layout length differs: Lean {expected} in {line}"
    return ()
  let [_, payload, terminal, maximum, interrupts, consumed, observed] := fields |
    throw s!"malformed weight stream case: {line}"
  let maxRead ← parseNat "read cap" maximum
  let retries ← parseNat "interrupt count" interrupts
  unless 0 < maxRead do throw s!"empty read cap: {line}"
  let bytes ← (parseCsv payload).mapM fun byte => do
    let n ← parseNat "byte" byte
    unless n < 256 do throw s!"out-of-range byte: {line}"
    pure (BitVec.ofNat 8 n)
  let terminalErr : Option Eval.WeightStream.ReadError ← match terminal with
    | "clean" => pure none
    | "eof" => pure (some .unexpectedEof)
    | "invalid" => pure (some .invalidData)
    | "denied" => pure (some .permissionDenied)
    | "other" => pure (some .other)
    | _ => throw s!"invalid terminal result: {line}"
  let widths := [2, 2, 4, 1, 2]
  let bytesRead := Eval.WeightStream.consumed widths bytes.length
  let expected ← match Eval.WeightStream.load widths bytes.length retries terminalErr with
    | .ok () =>
      let some (decoded, []) := Eval.WeightStream.readSlices widths bytes |
        throw s!"weight stream model accepted an incomplete read: {line}"
      pure (String.intercalate "," (decoded.map
        fun value => toString (Eval.WeightStream.littleWord value).toInt))
    | .error .unexpectedEof => pure "eof"
    | .error .invalidData => pure "invalid"
    | .error .permissionDenied => pure "denied"
    | .error .other => pure "other"
  unless bytesRead == consumed.toNat?.getD (bytes.length + 1) && expected == observed do
    throw s!"weight stream outcome differs: Lean {bytesRead} {expected} in {line}"

private def checkMoveStorageCase (line : String) : Except String Unit := do
  match line.splitOn " " with
  | ["order", values, sorted, selected] =>
    let scores ← (parseCsv values).mapM fun token => do
      let value ← parseInt "ordering value" token
      unless -2147483648 ≤ value && value ≤ 2147483647 do throw s!"out-of-range score: {line}"
      pure (Int32.ofInt value)
    let ids := List.range scores.length
    let input := ids.zip scores
    let sorted ← (parseCsv sorted).mapM (parseNat "move storage integer")
    let best ← (parseCsv selected).mapM (parseNat "move storage integer")
    let expected := MoveList.Ordering.sortByKeyDesc (fun row => row.2.toInt) input
    let expectedBest := (MoveList.BestFirst.ordered Prod.snd input).map Prod.fst
    if scores.length ≤ 3 then
      unless sorted == expected.map Prod.fst do throw s!"small move sort differs: {line}"
    else
      unless sorted.mergeSort == ids &&
          sorted.filterMap (fun i => scores[i]?) == expected.map Prod.snd do
        throw s!"large move sort differs: {line}"
    unless best == expectedBest do throw s!"best-first selection differs: Lean {expectedBest} in {line}"
  | ["claims", width, count, offset, claimed] =>
    let bits ← parseNat "move storage integer" width
    let len ← parseNat "move storage integer" count
    let start ← parseNat "move storage integer" offset
    unless (bits == 32 || bits == 64) && len ≤ 35 && start < 2 ^ bits do
      throw s!"out-of-range move claim: {line}"
    let expected := (List.range' start (len - start)).map fun i => s!"{i}:{i+1}"
    unless parseCsv claimed == expected do throw s!"move claims differ: Lean {expected} in {line}"
  | ["retain", count, pattern, retained] =>
    let len ← parseNat "move storage integer" count
    let mask ← parseNat "move storage integer" pattern
    unless len ≤ 35 && mask < 2 ^ 64 do
      throw s!"out-of-range retain case: {line}"
    let visible := MoveList.Retain.compact (fun i => mask.testBit i) len 0 0 (fun i => some i)
    let expected := (List.range visible.len).filterMap visible.memory
    let actual ← (parseCsv retained).mapM (parseNat "move storage integer")
    unless actual == expected do
      throw s!"retained moves differ: Lean {expected} in {line}"
  | _ => throw s!"malformed move storage case: {line}"

private def moveGenerationHeader (header : String) : Except String (Bool × (Nat → String)) := do
  let ["backend", backend] := header.splitOn " "
    | throw "missing move generation backend"
  unless ["avx512", "avx2", "neon", "wasm", "portable"].contains backend do
    throw "invalid move generation backend"
  pure (backend == "avx512", fun n => s!"{n} move generation cases ({backend}), 0 mismatches")

private def checkMoveGenerationCase (paired : Bool) (line : String) : Except String Unit := do
  let p :: o :: mask :: mode :: output := line.splitOn " "
    | throw s!"malformed move generation case: {line}"
  let board : Board.Board :=
    ⟨← parseU64 "move generation player bits" p, ← parseU64 "move generation opponent bits" o⟩
  let filter ← parseU64 "move generation mask" mask
  unless board.player &&& board.opponent == 0 do throw s!"overlapping board: {line}"
  let selected := Board.getMoves ⟨false, false⟩ board &&& filter
  unless (mode == "new" && filter == -1) || mode == "subset" || (mode == "nonempty" && selected != 0) do
    throw s!"invalid move generation input: {line}"
  let generated := MoveList.Generation.fill (mode == "nonempty") paired ⟨false, false, false⟩
    board selected (fun _ => none)
  let some length := generated.len | throw "move generation left length uninitialized"
  let some wipeout := generated.wipeout | throw "move generation left wipeout cache uninitialized"
  let square (sq : Option (Fin 64)) := toString ((sq.map Fin.val).getD 64)
  let header := [toString length, square ((generated.slots 0).map Move.Move.sq), square wipeout]
  let moves ← (List.range length).mapM fun i => do
    let some mv := generated.slots i | throw s!"move generation left slot {i} uninitialized"
    pure s!"{mv.sq.val}:{mv.flipped.toNat}:{mv.value.toInt}"
  unless output == header ++ moves do
    throw s!"move generation differs: Lean {String.intercalate " " (header ++ moves)} in {line}"

open TranspositionTable in
private def checkTtSaveCase (line : String) : Except String Unit := do
  let [endian, seq, oldPlayer, oldOpponent, oldData, newPlayer, newOpponent, newData,
      access, snapshot, lookup, afterSeq, afterPlayer, afterOpponent, afterData] := line.splitOn " "
    | throw s!"malformed TT save case: {line}"
  let bigEndian ← parseFlag "TT save byte order" endian
  let seq ← parseU64 "TT save word" seq
  let old : SeqLock.Payload :=
    ⟨← parseU64 "TT save word" oldPlayer, ← parseU64 "TT save word" oldOpponent,
      ← parseU64 "TT save word" oldData⟩
  let key := (← parseU64 "TT save word" newPlayer, ← parseU64 "TT save word" newOpponent)
  let data := Packing.unpack bigEndian (← parseU64 "TT save word" newData)
  let readMetadata (entry : EntryData) :=
    let square := ((Packing.decodeBestMove entry.bestMove).map Fin.val).getD 64
    let endgame := boolBit (isEndgame entry)
    let pv := boolBit (Access.isPv entry)
    s!"{entry.score.toInt}:{entry.bound.toNat}:{entry.depth.toNat}:{selectivityRank entry.selectivity}:{entry.generation.toNat}:{endgame}:{pv}:{square}"
  let expectedAccess := s!"{readMetadata (Packing.unpack bigEndian old.metadata)}|{readMetadata data}"
  unless access == expectedAccess do
    throw s!"TT metadata access differs (Lean {expectedAccess}): {line}"
  let expectedSnapshot := (Snapshot.read bigEndian seq seq old).map fun snap =>
    s!"{snap.player.toNat}:{snap.opponent.toNat}:{snap.sequence.toNat}:{snap.metadata.toNat}"
  unless snapshot == expectedSnapshot.getD "-" do
    throw s!"TT snapshot differs (Lean {(expectedSnapshot.getD "-")}): {line}"
  let expectedLookup := (LookupCut.readOccupied bigEndian seq seq old key.1 key.2).map fun entry =>
    toString (Packing.pack bigEndian entry).toNat
  unless lookup == expectedLookup.getD "-" do
    throw s!"TT lookup differs (Lean {(expectedLookup.getD "-")}): {line}"
  let obs : Save.Observation := ⟨seq, seq, old, seq⟩
  let chosen := Save.run bigEndian key data (List.replicate 4 obs)
  let expected := match chosen.map (Save.attempt bigEndian key data) with
    | some (.written next entry) => (next, key.1, key.2, Packing.pack bigEndian entry)
    | _ => (seq, old.player, old.opponent, old.metadata)
  let actual := (← parseU64 "TT save word" afterSeq, ← parseU64 "TT save word" afterPlayer,
    ← parseU64 "TT save word" afterOpponent, ← parseU64 "TT save word" afterData)
  unless actual == expected do throw s!"TT saved payload differs (Lean {repr expected}): {line}"

private def checkEndgameProbCutCase (line : String) : Except String Unit := do
  let [beta, verificationBits, screenBits, upperDelta, lowerDelta, pcBeta, screenBeta] :=
    line.splitOn " " | throw s!"malformed ProbCut window case: {line}"
  let beta ← parseInt "ProbCut beta" beta
  let verification ← parseU64 "verification margin" verificationBits
  let screen ← parseU64 "screen margin" screenBits
  unless some (EndgameProbCut.roundedMargin true verification) == upperDelta.toInt? do
    throw s!"verification ceil and cast disagree with the model: {line}"
  unless some (EndgameProbCut.roundedMargin false screen) == lowerDelta.toInt? do
    throw s!"screen floor and cast disagree with the model: {line}"
  let expected := EndgameProbCut.nativeThresholds beta verification screen
  unless some expected.verification == pcBeta.toInt? && some expected.screen == screenBeta.toInt? do
    throw s!"ProbCut thresholds disagree with the model: {line}"

private def checkMidgameProbCutCase (line : String) : Except String Unit := do
  let [depth, shallow, beta, t, cutNode, mean, sigma, mean0, sigma0, pcBeta, screen] :=
    line.splitOn " " | throw s!"malformed midgame ProbCut case: {line}"
  let depth ← parseNat "deep depth" depth
  unless depth < 2 ^ 32 && some (MidgameProbCut.shallowDepth depth) == shallow.toNat? do
    throw s!"midgame ProbCut shallow depth disagrees with the model: {line}"
  let beta ← parseInt "midgame ProbCut beta" beta
  let cutNode ← parseFlag "midgame ProbCut node type" cutNode
  let parseFloat (word : String) : Except String Int := do
    let bits ← parseU64 "binary64 bits" word
    let some units := Binary64Arithmetic.finiteUnits bits
      | throw s!"expected a finite coefficient: {word}"
    return units
  let coefficients : MidgameProbCut.Coefficients :=
    ⟨← parseFloat mean, ← parseFloat sigma, ← parseFloat mean0, ← parseFloat sigma0⟩
  let some (verification, evaluation) := MidgameProbCut.thresholdsFromUnits beta
    (← parseFloat t) coefficients cutNode | throw s!"nonfinite modeled intermediate: {line}"
  unless some (MidgameProbCut.publicThreshold verification) == pcBeta.toInt? do
    throw s!"midgame ProbCut verification threshold disagrees with the model: {line}"
  unless some (MidgameProbCut.publicThreshold evaluation) == screen.toInt? do
    throw s!"midgame ProbCut screen threshold disagrees with the model: {line}"

open PrincipalVariation in
private def checkPvStackCase (line : String) : Except String Unit := do
  let [op, ply, square, parent, child, nextParent, nextChild] :=
    line.splitOn " " | throw s!"malformed PV stack case: {line}"
  let ply ← parseNat "PV stack ply" ply
  unless ply < capacity do throw s!"PV ply out of range: {line}"
  let square ← parseNat "PV square" square
  let parseRow (row : String) : Except String Buffer := do
    let entries := row.splitOn ","
    unless entries.length = capacity do throw s!"wrong PV row length: {row}"
    entries.mapM fun entry => do
      let index ← parseNat "PV entry" entry
      if fits : index < 64 then pure (some ⟨index, fits⟩)
      else if index = 64 then pure none
      else throw s!"invalid PV entry: {entry}"
  let parent ← parseRow parent
  let child ← parseRow child
  let actualParent ← parseRow nextParent
  let actualChild ← parseRow nextChild
  let stack := (StackAccess.initial.set ply parent).set (ply + 1) child
  let next ← match op with
    | "u" =>
      if fits : square < 64 then do
        let some next := StackAccess.promote stack ply ⟨square, fits⟩ | throw "cannot prepend without a child row"
        pure next
      else throw "invalid placement square"
    | "p" => do
      let some next := StackAccess.prepareAt stack ply | throw "cannot prepare an absent parent row"
      pure next
    | "b" =>
      if fits : square < 64 then do
        let some first := StackAccess.promote stack ply 0 | throw "cannot publish first sibling"
        let some cleared := StackAccess.clearAt first (ply + 1) | throw "cannot clear sibling row"
        let some next := StackAccess.promote cleared ply ⟨square, fits⟩ | throw "cannot publish second sibling"
        pure next
      else throw "invalid placement square"
    | _ => throw s!"unknown PV operation: {op}"
  unless next[ply]? == some actualParent do throw s!"parent PV buffer disagrees with the model: {line}"
  unless next[ply+1]?.getD child == actualChild do throw s!"child PV buffer disagrees with the model: {line}"

open MidgameAspiration in
private def midgameAspirationTrial (style abortAt : Nat) : Aspiration.Trials Unit := fun i a b =>
  if i = abortAt then ⟨32767, (), true⟩
  else
    let score := match style with
      | 0 => 0
      | 1 => max a (-32766)
      | 2 => min b 32766
      | _ => if i % 2 = 0 then min b 32766 else max a (-32766)
    ⟨score, (), false⟩

open MidgameAspiration in
private def midgameAttemptWindows (search : Aspiration.Trials Unit) :
    Nat → Nat → Attempt → List String
  | 0, _, _ => []
  | fuel + 1, i, current =>
    let trial := search i current.alpha current.beta
    let rest := if trial.aborted then [] else
      match current.retry trial.score with
      | none => []
      | some next => midgameAttemptWindows search fuel (i + 1) next
    s!"{current.alpha},{current.beta}" :: rest

open MidgameAspiration in
private def checkMidgameAspirationCase (line : String) : Except String Unit := do
  match line.splitOn " " with
  | ["w", depth, helper, raw, a, b, d] =>
    let history ← if raw = "none" then pure none else do
      let [avg, mss] := raw.splitOn "," | throw s!"invalid root score history: {raw}"
      pure (some ⟨← parseInt "aspiration integer" avg, ← parseInt "aspiration integer" mss⟩)
    let start := initial (← parseNat "aspiration natural" depth) history
      (← parseNat "aspiration natural" helper)
    unless start == ⟨← parseInt "aspiration integer" a, ← parseInt "aspiration integer" b,
        ← parseInt "aspiration integer" d⟩ do
      throw s!"initial aspiration window disagrees with the model: {line}"
  | ["r", lo, hi, width, style, abortAt, a, b, score, clean, windows] =>
    let start : Attempt :=
      ⟨← parseInt "aspiration integer" lo, ← parseInt "aspiration integer" hi,
        ← parseInt "aspiration integer" width⟩
    let trials := midgameAspirationTrial (← parseNat "aspiration natural" style)
      (← parseNat "aspiration natural" abortAt)
    let some result := run trials 16 0 start | throw s!"aspiration retries exhausted: {line}"
    unless result.attempt.alpha == (← parseInt "aspiration integer" a) &&
        result.attempt.beta == (← parseInt "aspiration integer" b) &&
        result.trial.score == (← parseInt "aspiration integer" score) &&
        boolBit result.clean == clean do
      throw s!"final aspiration result disagrees with the model: {line}"
    unless ";".intercalate (midgameAttemptWindows trials 16 0 start) == windows do
      throw s!"aspiration attempt windows disagree with the model: {line}"
  | _ => throw s!"invalid midgame aspiration case: {line}"

private def checkEndgameAspirationCase (line : String) : Except String Unit := do
  let lo :: hi :: finalLo :: finalHi :: result :: records := line.splitOn " " |
    throw s!"invalid endgame aspiration case: {line}"
  let initial := (← parseInt "endgame aspiration integer" lo, ← parseInt "endgame aspiration integer" hi)
  let final := (← parseInt "endgame aspiration integer" finalLo, ← parseInt "endgame aspiration integer" finalHi)
  let result ← parseInt "endgame aspiration integer" result
  let trials ← records.toArray.mapM fun record => do
    let [a, b, s, flag] := record.splitOn "," | throw s!"invalid aspiration trial: {record}"
    let aborted ← parseFlag "stop flag" flag
    pure ((← parseInt "endgame aspiration integer" a, ← parseInt "endgame aspiration integer" b),
      (⟨← parseInt "endgame aspiration integer" s, (), aborted⟩ : Aspiration.Trial Unit))
  let some (last, trial) := trials.back? | throw s!"empty aspiration trace: {line}"
  let mut window := initial
  let mut delta := 256
  for i in [:trials.size] do
    let some (observed, current) := trials[i]? | throw s!"missing aspiration trial {i}: {line}"
    unless observed == window do throw s!"aspiration window mismatch at trial {i}: {line}"
    let next := if current.aborted then none else Aspiration.widen current.score window.1 window.2 delta
    match next with
    | none =>
      unless i + 1 == trials.size && observed == final && current.score == result do
        throw s!"aspiration return mismatch at trial {i}: {line}"
    | some widened =>
      unless i + 1 < trials.size do throw s!"missing aspiration retry: {line}"
      window := widened
      delta := delta + delta
  let search i _ _ := (trials[i]?.map Prod.snd).getD ⟨0, (), true⟩
  let accepted := Aspiration.runWith search 11 0 initial.1 initial.2 256
  match accepted with
  | none => unless trial.aborted do throw s!"completed aspiration modeled as aborted: {line}"
  | some chosen =>
    unless !trial.aborted && chosen.index + 1 == trials.size &&
        (chosen.alpha, chosen.beta) == last && chosen.trial.score == result do
      throw s!"accepted aspiration trial disagrees with the model: {line}"

open LazySmp in
private def checkLazySmpCase (line : String) : Except String Unit := do
  match line.splitOn " " with
  | ["s", helper, depth, skipped] =>
    let helper ← parseNat "helper index" helper
    let depth ← parseNat "helper depth plus ply" depth
    unless boolBit (skipsDepth helper depth) == skipped do
      throw s!"Lazy SMP schedule disagrees with the model: {line}"
  | "v" :: picked :: entries =>
    let ballots ← entries.mapM fun entry => do
      let [square, score, depth] := entry.splitOn "," | throw s!"invalid ballot: {entry}"
      pure (⟨← parseSquare "vote square" square, ← parseInt "vote score" score,
        ← parseNat "vote depth" depth⟩ : Ballot)
    let expected := (winner ballots).map (fun result => toString result.1)
    unless expected.getD "-" == picked do throw s!"Lazy SMP winner disagrees with the model: {line}"
  | _ => throw s!"invalid Lazy SMP case: {line}"

private def checkFiniteScoreCase (line : String) : Except String Unit := do
  let [score, actual] := line.splitOn " " | throw s!"invalid finite score case: {line}"
  let score ← parseInt "i32 score" score
  unless -(2^31 : Int) ≤ score && score < (2^31 : Int) do
    throw s!"score outside i32 domain: {line}"
  unless boolBit (MidgameScoreRange.isFinite (BitVec.ofInt 32 score)) == actual do
    throw s!"finite score admission disagrees with the model: {line}"

open EvalCache in
private def checkEvalCacheCase (line : String) : Except String Unit := do
  match line.splitOn " " with
  | ["g", pointerBits, size, key, count, index, way] =>
    let size ← parseNat "cache integer" size
    let key ← parseU64 "cache word" key
    unless checkedCount (← parseNat "cache integer" pointerBits) size ==
          some (← parseNat "cache integer" count) &&
        bucket size key == (← parseNat "cache integer" index) &&
        victim key == (← parseNat "cache integer" way) do
      throw s!"evaluation cache address disagrees with the model: {line}"
  | ["e", key, signed, payload] =>
    let score ← parseInt "cache score" signed
    unless encode (← parseU64 "cache word" key) (BitVec.ofInt 32 score) ==
        some (← parseU64 "cache word" payload) do
      throw s!"evaluation cache encoding disagrees with the model: {line}"
  | ["p", key, replacement, a, b, c, d, way, score] =>
    let replacement ← parseNat "cache integer" replacement
    let layout : Vector Word 4 := #v[← parseU64 "cache word" a, ← parseU64 "cache word" b,
      ← parseU64 "cache word" c, ← parseU64 "cache word" d]
    if fits : replacement < 4 then
      let (slot, signed) := match select (← parseU64 "cache word" key) (fun i => layout[i]) ⟨replacement, fits⟩ with
        | .hit i value => (i.val, toString value)
        | .miss i => (i.val, "-")
      unless slot == (← parseNat "cache integer" way) && signed == score do
        throw s!"evaluation cache probe disagrees with the model: {line}"
    else throw s!"cache victim way is out of range: {line}"
  | _ => throw s!"invalid evaluation cache case: {line}"

open AlignedBuffer in
private def checkAlignedBufferCase (line : String) : Except String Unit := do
  let indices (field : String) := (parseCsv field).mapM (parseNat "buffer integer")
  match line.splitOn " " with
  | ["l", pointerBits, count, itemSize, result, bytes] =>
    let model := checkedLayout (← parseNat "buffer integer" pointerBits)
      (← parseNat "buffer integer" count) (← parseNat "buffer integer" itemSize)
    let (expected, size) := match model with
      | .multiplyOverflow => ("overflow", 0)
      | .noBytes => ("empty", 0)
      | .invalidLayout => ("layout", 0)
      | .allocated value => ("alloc", value)
    unless expected == result && size == (← parseNat "buffer integer" bytes) do
      throw s!"aligned buffer layout disagrees with the model: {line}"
  | ["f", large, count, ending, stop, failingDrop, ready, panic, allocs, freed, pages, values, drops] =>
    let status ← match ending with
      | "all" | "drop-panic" => pure FillStatus.complete
      | "short" => pure FillStatus.short
      | "panic" => pure FillStatus.panic
      | _ => throw s!"invalid buffer iterator outcome: {ending}"
    let actual : Observation :=
      ⟨← parseFlag "buffer flag" ready, ← parseFlag "buffer flag" panic,
        ← parseNat "buffer integer" allocs, ← parseNat "buffer integer" freed,
        ← parseNat "buffer integer" pages, ← indices values, ← indices drops⟩
    let expected := fixture (← parseFlag "buffer flag" large) (← parseNat "buffer integer" count)
      status (← parseNat "buffer integer" stop) (← parseNat "buffer integer" failingDrop)
      (ending == "drop-panic")
    unless expected == actual do throw s!"aligned buffer cleanup disagrees with the model: {line}"
  | _ => throw s!"invalid aligned buffer case: {line}"

private def patternLanes (text : String) : Except String (Array Nat) := do
  let lanes ← (text.splitOn ",").toArray.mapM (parseNat "pattern feature value")
  unless lanes.size == 32 && lanes.all (· < 65536) do throw "invalid u16 pattern vector"
  return lanes

private def patternLocation (sq flip side ply : String) :
    Except String (Square.Square × Bits × Bool) := do
  let placed ← parseSquare "pattern update square" sq
  let bits ← parseU64 "pattern bitboard" flip
  let opponentMoves ← parseFlag "pattern update side" side
  let ply ← parseNat "pattern feature value" ply
  unless Eval.PatternUpdate.updateAllowed ply bits do throw "invalid pattern update"
  return (placed, bits, opponentMoves)

open Eval.PatternUpdate in
private def checkPatternFeatureCase (line : String) : Except String Unit := do
  match (line.splitOn " ").filter (· != "") with
  | ["geometry", index, coords] =>
    let index ← parseNat "pattern feature value" index
    let actual ← (coords.splitOn ",").mapM (parseNat "pattern feature value")
    if hi : index < 32 then
      unless pattern ⟨index, hi⟩ == actual do throw s!"pattern geometry disagrees: {line}"
    else throw "invalid pattern index"
  | ["board", player, opponent, sq, flip, side, ply, p₀, o₀, p₁, o₁] =>
    let b : Board := ⟨← parseU64 "pattern bitboard" player, ← parseU64 "pattern bitboard" opponent⟩
    let (placed, flipped, opponentMoves) ← patternLocation sq flip side ply
    let actor := moverBoard opponentMoves b
    if (actor.player &&& actor.opponent) != 0 || (actor.player ||| actor.opponent).getLsbD placed.val ||
        (flipped &&& ~~~actor.opponent) != 0 then
      throw s!"invalid board delta: {line}"
    let after := rootAfter opponentMoves b placed flipped
    let codes (pos : Board) : Array Nat × Array Nat :=
      let pairs := (List.finRange 32).toArray.map fun i => pairedCode (pattern i) pos
      (pairs.map (·.1.toNat), pairs.map (·.2.toNat))
    let beforeActual := (← patternLanes p₀, ← patternLanes o₀)
    let afterActual := (← patternLanes p₁, ← patternLanes o₁)
    unless beforeActual == codes b && afterActual == codes after do
      throw s!"incremental features disagree with board encoding: {line}"
  | ["raw", sq, flip, side, ply, p₀, o₀, p₁, o₁] =>
    let (placed, flipped, opponentMoves) ← patternLocation sq flip side ply
    let p ← patternLanes p₀
    let o ← patternLanes o₀
    let afterP ← patternLanes p₁
    let afterO ← patternLanes o₁
    for i in List.finRange 32 do
      let squares := pattern i
      let f := lookupWeight placed.val squares
      let s := flipCode squares flipped
      let want := wordPair (updatePair opponentMoves p[i.val]! o[i.val]! f s)
      unless want.1.toNat == afterP[i.val]! && want.2.toNat == afterO[i.val]! do
        throw s!"wrapping pattern update disagrees at lane {i.val}: {line}"
  | _ => throw s!"invalid pattern feature case: {line}"

private def checkStabilityCase (table : Stability.Edge.EdgeTable) (line : String) :
    Except String Unit := do
  let tag :: fields := line.splitOn " " | throw s!"missing stability case tag: {line}"
  let word := BitVec.ofNat 64
  let byte := BitVec.ofNat 8
  if tag == "cut" then
    let [p, o, n, alpha, margin, cutoff] := fields | throw s!"invalid cutoff case: {line}"
    let p := word (← parseNat "stability value" p)
    let o := word (← parseNat "stability value" o)
    let value ← parseInt "disc score" alpha
    let alpha : Stability.Cutoff.DiscScore := ⟨value⟩
    let expected := Stability.Cutoff.cutoffAtEmptyCount p o (← parseNat "stability value" n) alpha
    let enough := Stability.Cutoff.hasEnoughOpponentDiscsForCutoff
      (Stability.Cutoff.popcount o) alpha
    let expectedMargin := if alpha.value < 6 then "-" else boolBit enough
    unless expectedMargin == margin && ((expected.map (toString ·.value)).getD "-") == cutoff do
      throw s!"stability cutoff disagrees: {line}"
    return
  let values ← fields.mapM (parseNat "stability value")
  let agrees : Bool :=
    match tag, values with
    | "table", [p, o, index, stable] =>
      decide (p < 256 ∧ o < 256) &&
        Stability.Edge.edgeIndex (byte p) (byte o) == index &&
        (Stability.Edge.edgeStabilityOcc table (byte p) (byte o)).toNat == stable
    | "move", [p, o, sq, bit, nextP, nextO] =>
      if valid : sq < 8 then
        let (p', o') := Stability.Edge.applyEdgeMove ⟨sq, valid⟩ (byte p) (byte o)
        (Stability.Edge.bit8 sq).toNat == bit && p'.toNat == nextP && o'.toNat == nextO
      else false
    | "pack", [p, a, h] =>
      (Stability.Semantics.packA1A8 (word p)).toNat == a &&
        (Stability.Semantics.packH1H8 (word p)).toNat == h
    | "unpack", [x, a, h] =>
      (Stability.Semantics.unpackA2A7 (byte x)).toNat == a &&
        (Stability.Semantics.unpackH2H7 (byte x)).toNat == h
    | "board", [p, o, edge, h, v, d7, d9, intersection, stable] =>
      let p := word p
      let o := word o
      let full := Stability.getFullLines (p ||| o)
      (Stability.Semantics.getStableEdge table p o).toNat == edge &&
        full == ⟨word h, word v, word d7, word d9⟩ && full.intersection.toNat == intersection &&
        (Stability.Semantics.getStableDiscs p o).toNat == stable
    | "contact", [central, seed, h, v, d7, d9, stable] =>
      (Stability.getStableByContact (word central) (word seed)
        ⟨word h, word v, word d7, word d9⟩).toNat == stable
    | _, _ => false
  unless agrees do throw s!"stability case disagrees: {line}"

private def runStabilityCases (path : String) : IO UInt32 := do
  let cases ← IO.FS.lines path
  let table := Stability.Edge.initEdgeStability
  let mut tables := 0
  let mut moves := 0
  for line in cases do
    IO.ofExcept (checkStabilityCase table line)
    if line.startsWith "table " then tables := tables + 1
    if line.startsWith "move " then moves := moves + 1
  unless tables == 65536 && moves == 17496 && cases.size == 105280 do
    throw (IO.userError s!"incomplete stability fixture: {tables} entries, {moves} moves, {cases.size} cases")
  (← IO.getStdout).putStrLn s!"{cases.size} stability cases, 0 mismatches"
  return 0

private def checkSyncPrimitiveCase (line : String) : Except String Unit := do
  let input :: observed := line.splitOn "|" | throw s!"missing synchronization input: {line}"
  match input.splitOn " " with
  | ["mask", raw, first, second] =>
    let start ← parseU64 "mask" raw
    let a ← parseSquare "mask index" first
    let b ← parseSquare "mask index" second
    let first := (Threading.AtomicBitSet.Update.set a).apply start
    let booked := (Threading.AtomicBitSet.Update.set b).apply first
    let afterFirst := (Threading.AtomicBitSet.Update.reset a).apply booked
    let finished := (Threading.AtomicBitSet.Update.reset b).apply afterFirst
    let cleared := Threading.AtomicBitSet.Update.clear.apply finished
    let expected := [0, booked, afterFirst, finished, cleared].map fun bits =>
      let count := Threading.AtomicBitSet.count bits
      let empty := boolBit (Threading.AtomicBitSet.none bits)
      s!"{bits.toNat} {count} {empty}"
    unless observed == expected do
      throw s!"atomic bitset disagrees for {input}: {observed} vs {expected}"
  | ["guard", transfer, published, read] =>
    unless (transfer == "0" || transfer == "1") && observed.isEmpty && published == read do
      throw s!"guard handoff disagrees: {line}"
  | _ => throw s!"invalid synchronization row: {line}"

private def runSyncPrimitiveCases (path : String) : IO UInt32 := do
  let cases ← IO.FS.lines path
  let masks := cases.countP (·.startsWith "mask ")
  let guards := cases.countP (·.startsWith "guard ")
  unless masks == 4096 && guards == 8 && cases.size == masks + guards do
    throw (IO.userError "missing synchronization primitive cases")
  for line in cases do
    IO.ofExcept (checkSyncPrimitiveCase line)
  (← IO.getStdout).putStrLn s!"{masks} atomic bitset traces and {guards} lock handoffs, 0 mismatches"
  return 0

private def caseCommands : List (String × (String → IO UInt32)) := [
  ("activation-cases", runHeaderCaseFile "activation backend" activationHeader
    (fun (fused, scRelu) => checkActivationCase fused scRelu)),
  ("move-cases", runHeaderCaseFile "move generation backend" moveCasesHeader checkMoveCase),
  ("endgame-timing-cases", runCaseFile "endgame timing" checkEndgameTimingCase),
  ("time-settings-cases", runCaseFile "time settings" checkTimeSettingsCase),
  ("fixed-endgame-cases", runHeaderCaseFile "fixed endgame ladder" fixedEndgameHeader
    (fun _ => checkFixedEndgameCase)),
  ("game-analysis-cases", runCaseFile "game analysis" checkGameAnalysisCase),
  ("obf-cases", runCaseFile "OBF" checkObfCase),
  ("empty-list-cases", runCaseFile "empty list" checkEmptyListCase),
  ("search-counter-cases", runCaseFile "search counter" checkCounterCase),
  ("board-hash-cases", runCaseFile "board hash" checkBoardHashCase),
  ("move-storage-cases", runCaseFile "move storage" checkMoveStorageCase),
  ("move-generation-cases", runHeaderCaseFile "move generation" moveGenerationHeader
    checkMoveGenerationCase),
  ("tt-save-cases", runCaseFileWith "TT save" checkTtSaveCase (some 1008)
    (some "TT snapshot, lookup and save cases")),
  ("stability-cases", runStabilityCases),
  ("sync-primitive-cases", runSyncPrimitiveCases),
  ("stop-token-cases", runCaseFile "stop token" checkStopTokenCase),
  ("in-flight-cases", runCaseFileWith "in-flight" checkInFlightCase none
    (some "in-flight slot cases")),
  ("endgame-probcut-cases", runCaseFile "endgame ProbCut" checkEndgameProbCutCase),
  ("midgame-probcut-cases", runCaseFile "midgame ProbCut" checkMidgameProbCutCase),
  ("pv-stack-cases", runCaseFile "PV stack" checkPvStackCase),
  ("midgame-aspiration-cases", runCaseFile "midgame aspiration" checkMidgameAspirationCase),
  ("endgame-aspiration-cases", runCaseFile "endgame aspiration" checkEndgameAspirationCase),
  ("lazy-smp-cases", runCaseFile "Lazy SMP" checkLazySmpCase),
  ("lmr-cases", runCaseFile "LMR" checkLmrCase),
  ("split-report-cases", runCaseFileWith "split report" checkSplitReportCase (some 4704) none),
  ("finite-score-cases", runCaseFile "finite score" checkFiniteScoreCase),
  ("eval-cache-cases", runCaseFile "evaluation cache" checkEvalCacheCase),
  ("aligned-buffer-cases", runCaseFile "aligned buffer" checkAlignedBufferCase),
  ("large-page-cases", runCaseFileWith "Windows large-page" checkLargePageCase none
    (some "Windows large-page cases with modeled Win32 replies")),
  ("weight-stream-cases", runCaseFile "weight stream" checkWeightStreamCase),
  ("pattern-feature-cases", runCaseFileWith "pattern feature" checkPatternFeatureCase
    (some 2592) none),
  ("flip-scalar-cases", runFlipScalarCases)
]

def main (args : List String) : IO UInt32 := do
  let stdout ← IO.getStdout
  match args with
  | ["registry"] =>
    stdout.putStrLn (toJson entries).compress
    return 0
  | ["move-bound-cnf", boundText] =>
    if let some bound := boundText.toNat? then
      if bound < 64 then
        stdout.putStr (LegalMoves.MoveBound.dimacsFormula bound).dimacs
        return 0
    (← IO.getStderr).putStrLn "usage: revmodel move-bound-cnf <bound 0..63> or move-bound-check <bound 0..63> <proof.lrat>"
    return 2
  | ["move-bound-check", boundText, path] =>
    if let some bound := boundText.toNat? then
      if bound < 64 then
        let formula := LegalMoves.MoveBound.dimacsFormula bound
        let proof ← Std.Tactic.BVDecide.LRAT.loadLRATProof path
        if Std.Tactic.BVDecide.LRAT.check proof formula then
          stdout.putStrLn s!"LRAT verified: at most {bound} legal moves"
          return 0
        (← IO.getStderr).putStrLn "LRAT rejected for this move bound"
        return 1
    (← IO.getStderr).putStrLn "usage: revmodel move-bound-cnf <bound 0..63> or move-bound-check <bound 0..63> <proof.lrat>"
    return 2
  | "move-bound-cnf" :: _ | "move-bound-check" :: _ =>
    (← IO.getStderr).putStrLn "usage: revmodel move-bound-cnf <bound 0..63> or move-bound-check <bound 0..63> <proof.lrat>"
    return 2
  | ["perft-root", depth] =>
    if let some n := depth.toNat? then
      if n ≤ 9 then
        stdout.putStrLn (toString (Perft.count ⟨false, false⟩ ⟨false, false, false⟩ Perft.root n))
        return 0
    (← IO.getStderr).putStrLn "usage: revmodel perft-root <depth 0..9>"
    return 2
  | "perft-root" :: _ =>
    (← IO.getStderr).putStrLn "usage: revmodel perft-root <depth 0..9>"
    return 2
  | ["pvs-trace-check", path] =>
    return ← PvsParallel.TraceCheck.runFile path
  | ["pvs-trace-check", path, depth] =>
    if let some n := depth.toNat? then
      if n ≤ 64 then return ← PvsParallel.TraceCheck.runFile path n
    IO.eprintln "pvs-trace-check: expected a threshold depth from 0 to 64"
    return 1
  | "pvs-trace-check" :: _ =>
    (← IO.getStderr).putStrLn "usage: revmodel pvs-trace-check <path>"
    return 2
  | [name, path] =>
    if let some run := caseCommands.lookup name then
      return ← run path
    (← IO.getStderr).putStrLn s!"unknown command: {name}"
    return 2
  | name :: _ =>
    if (caseCommands.lookup name).isSome then
      (← IO.getStderr).putStrLn s!"usage: revmodel {name} <path>"
      return 2
    (← IO.getStderr).putStrLn s!"unknown command: {name}"
    return 2
  | [] =>
    (← IO.getStderr).putStrLn "usage: revmodel <command> [args...]"
    return 2

import NeuralReversi
import NeuralReversi.Board.ConstructionFixture
import NeuralReversi.Board.TransformFixture
import NeuralReversi.Bitboard.MovesModel
import NeuralReversi.Bitboard.PotentialModel
import NeuralReversi.Bitboard.Avx2Model
import NeuralReversi.Bitboard.Avx512Model
import NeuralReversi.Bitboard.Avx2PotentialModel
import NeuralReversi.Simd.Fixture
import NeuralReversi.Simd.ZmmFixture
import NeuralReversi.Bitboard.Avx512PotentialModel
import NeuralReversi.Bitboard.NeonMoves
import NeuralReversi.LegalMoves.Encoding
import Std.Tactic.BVDecide.LRAT.Parser
import NeuralReversi.Eval.Activations
import NeuralReversi.Eval.PatternUpdate.Storage
import NeuralReversi.EndgameProbCut
import NeuralReversi.EndgameTiming
import NeuralReversi.MidgameProbCut
import NeuralReversi.MidgameAspiration.Model
import NeuralReversi.LazySmp.Schedule
import NeuralReversi.LazySmp.Voting
import NeuralReversi.MidgameScoreRange.Domain
import NeuralReversi.EvalCache.Probe
import NeuralReversi.AlignedBuffer.Cleanup
import NeuralReversi.PrincipalVariation.Buffer
import NeuralReversi.Move.Model
import NeuralReversi.Flip.ScalarModel
import NeuralReversi.Flip.Model
import NeuralReversi.Flip.Avx2.Fixture
import NeuralReversi.Flip.Avx512.Fixture
import NeuralReversi.Board.MovesFixture
import NeuralReversi.Board.Move.Fixture
import NeuralReversi.Game.History.Fixture
import NeuralReversi.Game.Terminal.Fixture
import NeuralReversi.Game.Initialization.Fixture
import NeuralReversi.Game.Queries.Fixture
import NeuralReversi.ScaledScore.DisplayControl.Fixture

open Lean NeuralReversi NeuralReversi.Bitboard NeuralReversi.Board

def coordinateOutput : Except Square.Panic Nat → String
  | .ok value => toString value
  | .error reason => s!"error:{reason.message}"

def maskOutput : Except String Bits → String
  | .ok value => s!"ok:{value.toNat}"
  | .error reason => s!"error:{reason}"

def membershipOutput : Except String Bool → String
  | .ok value => s!"ok:{if value then 1 else 0}"
  | .error reason => s!"error:{reason}"

def scoreOutput : Except String Score → String
  | .ok value => s!"ok:{value.toInt}"
  | .error reason => s!"error:{reason}"

def shiftAssignmentOutput (outcome : Except String Unit × Bits) : String :=
  let status := match outcome.1 with
    | .ok () => "ok"
    | .error reason => s!"error:{reason}"
  s!"{status}|{outcome.2.toNat}"

def iteratorTraceOutput (checked : Bool) (state : Bits) : Except String String :=
  (iteratorTrace 66 checked state).map fun values =>
    String.intercalate " " (values.map fun value => match value with
      | some index => toString index
      | none => "none")

def boardGeometryOutput (b : Board) : String :=
  String.intercalate "|" ([Board.flipVertical b, Board.flipHorizontal b,
    Board.flipDiagA1H8 b, Board.flipDiagA8H1 b, Board.rotate90 b, Board.rotate180 b,
    Board.rotate270 b, Board.unique b].map fun value =>
      s!"{value.player.toNat} {value.opponent.toNat}")

def uncheckedOutput : Square.UncheckedResult → Except String String
  | .value square => .ok s!"ok:{square.val}"
  | .panic input => .ok s!"error:{Square.UncheckedResult.message input}"
  | .invalidDiscriminant _ => .error "invalid enum transmutation is outside executable comparisons"

def hexNibble (c : Char) : Except String Nat :=
  let n := c.toNat
  if 48 ≤ n ∧ n ≤ 57 then .ok (n - 48)
  else if 97 ≤ n ∧ n ≤ 102 then .ok (n - 87)
  else if 65 ≤ n ∧ n ≤ 70 then .ok (n - 55)
  else .error "invalid hex digit"

def decodeHexBytes : List Char → Except String (List UInt8)
  | [] => .ok []
  | high :: low :: rest => do
    let high ← hexNibble high
    let low ← hexNibble low
    let tail ← decodeHexBytes rest
    pure (UInt8.ofNat (16 * high + low) :: tail)
  | [_] => .error "odd hex input length"

def decodeText (encoded : String) : Except String String := do
  if encoded == "-" then return ""
  let bytes ← decodeHexBytes encoded.toList
  match String.fromUTF8? (ByteArray.mk bytes.toArray) with
  | some text => pure text
  | none => throw "invalid UTF-8 protocol input"

def squareParseOutput : Except Square.ParseFailure Square.Square → String
  | .ok square => s!"ok:{square.val}"
  | .error (.error .invalidFormat) => "error:format"
  | .error (.error (.invalidFile byte)) => s!"error:file:{byte.toNat}"
  | .error (.error (.invalidRank byte)) => s!"error:rank:{byte.toNat}"
  | .error (.panic reason) => s!"panic:{reason.message}"

def sequenceParseOutput : Except Square.SeqFailure (List Square.Square) → String
  | .ok moves => s!"ok:{String.intercalate "," (moves.map fun sq => toString sq.val)}"
  | .error (.error reason) =>
    let tag := match reason.source with
      | .invalidFormat => "format"
      | .invalidFile byte => s!"file:{byte.toNat}"
      | .invalidRank byte => s!"rank:{byte.toNat}"
    s!"error:{reason.index.toNat}:{tag}"
  | .error (.panic reason) => s!"panic:{reason.message}"

def encodeText (text : String) : String :=
  String.ofList (text.toUTF8.data.toList.flatMap fun byte =>
    let digit (n : Nat) := Char.ofNat (if n < 10 then 48 + n else 87 + n)
    [digit (byte.toNat / 16), digit (byte.toNat % 16)])

def utf8ChunkDiagnostic (bytes : List Square.Byte) : String :=
  let validity := match Square.decodeByteList bytes with
    | none => "invalid"
    | some text => s!"ok:{if text.isEmpty then "-" else encodeText text}"
  s!"{validity}|{squareParseOutput (Square.parseChunk bytes)}"

def discResultOutput : Except String Disc.Disc → String
  | .ok .empty => "ok:0"
  | .ok .black => "ok:1"
  | .ok .white => "ok:2"
  | .error reason => s!"error:{reason}"

def renderedBoardOutput : Except String String → String
  | .ok text => s!"ok:{encodeText text}"
  | .error reason => s!"error:{reason}"

def valueObservation (value : Bitboard.Value) : String :=
  let cells := String.ofList ((List.finRange 64).map fun i =>
    if value.storage.getLsbD i.val then '1' else '0')
  s!"{(Bitboard.bits value).toNat} {(Bitboard.intoU64 value).toNat} {(Bitboard.valueCount value).toNat} {cells}"

def squareValueOutput (checked : Bool) (square : Square.Square) : String :=
  match Bitboard.fromSquareValue checked square with
  | .ok value => s!"ok:{valueObservation value}"
  | .error message => s!"error:{message}"

def boardObservationOutput (b : Board) : String :=
  s!"{valueObservation (Board.getPlayer b)}|{valueObservation (Board.getOpponent b)}"

def initialStateOutput (checked : Bool) : String :=
  let observe (b : Board) :=
    s!"{boardObservationOutput b}@{renderedBoardOutput (Board.toStringAsBoard checked b .black)}"
  s!"{observe Board.defaultBoard}#{observe Board.newBoard}"

def boardParseOutput : Except Board.TextFailure Board → String
  | .ok b => s!"ok:{b.player.toNat} {b.opponent.toNat}"
  | .error (.error (.tooShort expected actual)) => s!"error:short:{expected.toNat}:{actual.toNat}"
  | .error (.error (.tooLong expected actual)) => s!"error:long:{expected.toNat}:{actual.toNat}"
  | .error (.error (.invalidChar c position)) => s!"error:char:{c.toNat}:{position.toNat}"
  | .error (.panic message) => s!"panic:{message}"

def scaledObservationOutput (score : ScaledScore.ScaledScore) : String :=
  s!"{(ScaledScore.value score).toInt} {(ScaledScore.toDiscDiff score).toInt}"

def scaledConstructorOutput : Except String ScaledScore.ScaledScore → String
  | .ok score => s!"ok:{scaledObservationOutput score}"
  | .error reason => s!"error:{reason}"

def scaledAssignmentOutput (outcome : Except String Unit × ScaledScore.ScaledScore) : String :=
  let status := match outcome.1 with
    | .ok () => "ok"
    | .error reason => s!"error:{reason}"
  s!"{status}|{(ScaledScore.value outcome.2).toInt}"

def scaledArithmeticOutput (checked : Bool) (a b : ScaledScore.ScaledScore) : String :=
  let output (result : Except String ScaledScore.ScaledScore) :=
    scoreOutput (result.map ScaledScore.value)
  String.intercalate "|" [
    output (ScaledScore.addScore checked a b), output (ScaledScore.subScore checked a b),
    output (ScaledScore.negate checked a), output (ScaledScore.addInt checked a b.raw),
    output (ScaledScore.subInt checked a b.raw),
    output (ScaledScore.divInt checked a b.raw),
    scaledAssignmentOutput (ScaledScore.addAssign checked a b),
    scaledAssignmentOutput (ScaledScore.subAssign checked a b)]

structure DisplayTrace where
  calls : Nat := 0
  bytes : String := ""
  chunks : List String := []

def traceWrite (stop behavior : Nat) (state : DisplayTrace) (chunk : String) :
    ScaledScore.WriteResult DisplayTrace String :=
  let stopping := state.calls = stop
  let added := if stopping ∧ (behavior = 3 ∨ behavior = 4) then
    String.ofList (chunk.toList.take 1) else chunk
  let next : DisplayTrace := ⟨state.calls + 1, state.bytes ++ added, state.chunks ++ [chunk]⟩
  if stopping ∧ (behavior = 1 ∨ behavior = 3) then .error next
  else if stopping ∧ (behavior = 2 ∨ behavior = 4) then .panic "display writer panic" next
  else .ok next

def displayTraceOutput (raw : ScaledScore.Word) (stop behavior : Nat) : String :=
  let outcome := ScaledScore.display raw {} (traceWrite stop behavior) {}
  let (status, state) := match outcome.1 with
    | .ok state => ("ok", state)
    | .error state => ("error", state)
    | .panic _ state => ("panic", state)
  s!"{status}|{if outcome.2 = ({} : ScaledScore.FormatOptions) then 1 else 0}|{state.bytes}|{String.intercalate "~" state.chunks}"

def runCase (line : String) : Except String String := do
  let fields := (line.splitOn " ").filter (· != "")
  if fields.head? == some "game-terminal" then
    return ← Game.Terminal.Fixture.request fields
  if fields.head? == some "game-init" then
    return ← Game.Initialization.Fixture.request fields
  if fields.head? == some "queries-action" then
    return ← Game.Queries.Fixture.actionRequest fields
  if fields.head? == some "queries-value" then
    return ← Game.Queries.Fixture.valueRequest fields
  if fields.head? == some "history-value" then
    return ← Game.History.Fixture.valueRequest fields
  if fields.head? == some "history-vec-model" then
    return ← Game.History.Fixture.vecRequest fields
  if fields.head? == some "display-control-replay" then
    return ← ScaledScore.DisplayControl.Fixture.request fields
  if fields.head? == some "board-move-release" || fields.head? == some "board-move-checked" then
    return ← Board.MoveFixture.request fields
  if fields.head? == some "board-wrappers-release" || fields.head? == some "board-wrappers-checked" then
    return ← Board.wrapperRequest fields
  if fields.head? == some "board-moves-release" || fields.head? == some "board-moves-checked" then
    return ← Board.movesRequest fields
  if fields.head? == some "flip-avx512-release" || fields.head? == some "flip-avx512-checked" then
    return ← Flip.Avx512.publicRequest fields
  if fields.head? == some "flip-avx512-index-release" || fields.head? == some "flip-avx512-index-checked" then
    return ← Flip.Avx512.indexRequest fields
  if fields.head? == some "avx512-flip-primitives" then
    return ← Flip.Avx512.primitiveRequest fields
  if fields.head? == some "flip-avx2-mm-release" || fields.head? == some "flip-avx2-mm-checked" then
    return ← Flip.Avx2.preparedRequest fields
  if fields.head? == some "avx2-flip-primitives" then
    return ← Flip.Avx2.flipPrimitiveRequest fields
  if fields.head? == some "avx512-zmm-primitives" then
    return ← Simd.X86.zmmPrimitiveRequest fields
  if fields.head? == some "avx512-zmm-ternary" then
    return ← Simd.X86.zmmTernaryRequest fields
  if fields.head? == some "avx512-ternary" then
    return ← Simd.X86.ternaryRequest fields
  if fields.head? == some "avx2-primitives" || fields.head? == some "avx2-pair-reduction" then
    return ← Simd.X86.primitiveRequest fields
  let [op, p, o, f, s] := fields | throw "expected: op player opponent flipped square"
  if op == "square-text-release" || op == "square-text-checked" then
    return squareParseOutput (Square.fromStr (← decodeText p))
  if op == "square-sequence-release" || op == "square-sequence-checked" then
    return sequenceParseOutput (Square.parseSequence (← decodeText p))
  if op == "board-parse-release" || op == "board-parse-checked" then
    let side : Disc.PlayerColor ← match o with
      | "1" => pure .black
      | "2" => pure .white
      | _ => throw "invalid player color"
    return boardParseOutput (Board.fromString (op == "board-parse-checked") (← decodeText p) side)
  let parse (text : String) : Except String Nat :=
    match text.toNat? with
    | some value => .ok value
    | none => .error "invalid unsigned decimal"
  let p ← parse p
  let o ← parse o
  let f ← parse f
  let s ← parse s
  if p >= 2^64 || o >= 2^64 || f >= 2^64 then
    throw "input outside model domain"
  let b : Board := ⟨BitVec.ofNat 64 p, BitVec.ofNat 64 o⟩
  let flipped := BitVec.ofNat 64 f
  if op == "flip-avx2-release" || op == "flip-avx2-checked" then
    if f != 0 then throw "AVX2 flip reserves the fourth field as zero"
    if h : s < 64 then
      return s!"ok:{(Flip.Avx2.flip ⟨s,h⟩ b.player b.opponent).toNat}"
    else throw "invalid square discriminant"
  if op == "flip-scalar-release" || op == "flip-scalar-checked" then
    if f != 0 then throw "scalar flip reserves the fourth field as zero"
    if h : s < 64 then
      return maskOutput (Flip.flip (op == "flip-scalar-checked") ⟨s,h⟩ b.player b.opponent)
    else throw "invalid square discriminant"
  if op == "scalar-bits-release" || op == "scalar-bits-checked" then
    if o != 0 || f != 0 || s != 0 then throw "expected one unsigned-operation word"
    return s!"{(b.player &&& -b.player).toNat}|{(Flip.Scalar.leadingZeros32 b.player).toNat}"
  if op == "scalar-flip-left-release" || op == "scalar-flip-left-checked" then
    if s != 0 then throw "expected three scalar flip words"
    return s!"ok:{(Flip.Scalar.flipLeft b.player b.opponent flipped).toNat}"
  if op == "scalar-flip-right-release" || op == "scalar-flip-right-checked" then
    if s != 0 then throw "expected three scalar flip words"
    match Flip.Scalar.flipRight (op == "scalar-flip-right-checked") b.player b.opponent flipped with
    | .ok value => return s!"ok:{value.toNat}"
    | .error reason => return s!"error:{reason}"
  if op == "move-constructor-release" || op == "move-constructor-checked" then
    if h : s < 64 then
      match Move.create (op == "move-constructor-checked") ⟨s,h⟩ flipped with
      | .ok value => return s!"ok:{value.sq.val}|{value.flipped.toNat}|{value.value.toInt}"
      | .error reason => return s!"error:{reason.message}"
    else throw "invalid square discriminant"
  if op == "transform-release" || op == "transform-checked" then
    if f >= 196 || s >= 2^64 then throw "transform fixture outside its domain"
    return Transform.Fixture.output b.player b.opponent f (BitVec.ofNat 64 s)
  if op == "board-constructor-release" || op == "board-constructor-checked" then
    if f >= 1024 || s >= 2^64 then throw "constructor fixture outside its domain"
    return Construction.Fixture.output b.player b.opponent f (BitVec.ofNat 64 s)
  let sq : Square ← if h : s < 64 then pure ⟨s, h⟩ else throw "invalid square"
  match op with
  | "representation-release" | "representation-checked" =>
    let raw := BitVec.ofNat 64 p
    pure s!"{valueObservation (Bitboard.new raw)}|{valueObservation (Bitboard.fromU64 raw)}"
  | "square-value-release" | "square-value-checked" =>
    if h : f < 64 then pure (squareValueOutput (op == "square-value-checked") ⟨f, h⟩)
    else throw "invalid square discriminant"
  | "board-observe-release" | "board-observe-checked" =>
    if b.player &&& b.opponent != 0 then throw "board constructor requires disjointness"
    pure (boardObservationOutput b)
  | "board-observe-raw-release" => pure (boardObservationOutput b)
  | "initial-state-release" | "initial-state-checked" =>
    pure (initialStateOutput (op == "initial-state-checked"))
  | "board-disc-at-release" | "board-disc-at-checked" | "board-disc-at-raw-release"
  | "board-render-release" | "board-render-checked" | "board-render-raw-release"
  | "board-roundtrip-release" | "board-roundtrip-checked" | "board-roundtrip-raw-release" =>
    if op == "board-disc-at-release" || op == "board-disc-at-checked" ||
        op == "board-render-release" || op == "board-render-checked" ||
        op == "board-roundtrip-release" || op == "board-roundtrip-checked" then
      if b.player &&& b.opponent != 0 then throw "board constructor requires disjointness"
    let side : Disc.PlayerColor ← match s with
      | 1 => pure .black
      | 2 => pure .white
      | _ => throw "invalid player color"
    let checked := op == "board-disc-at-checked" || op == "board-render-checked" ||
      op == "board-roundtrip-checked"
    if op == "board-render-release" || op == "board-render-checked" ||
        op == "board-render-raw-release" then
      pure (renderedBoardOutput (Board.toStringAsBoard checked b side))
    else if op == "board-roundtrip-release" || op == "board-roundtrip-checked" ||
        op == "board-roundtrip-raw-release" then
      match Board.toStringAsBoard checked b side with
      | .error message => pure s!"panic:{message}"
      | .ok text => pure (boardParseOutput (Board.fromString checked (Board.stripRowSeparators text) side))
    else if h : f < 64 then
      pure (discResultOutput (Board.discAt checked b ⟨f, h⟩ side))
    else throw "invalid square discriminant"
  | "scaled-display-release" | "scaled-display-checked" =>
    if p >= 2^32 then throw "scaled display input outside i32 bit domain"
    pure (displayTraceOutput (BitVec.ofNat 32 p) f s)
  | "scaled-f32-release" | "scaled-f32-checked" =>
    if p >= 2^32 then throw "scaled float input outside i32 bit domain"
    let input := BitVec.ofNat 32 p
    pure s!"{(ScaledScore.toDiscDiffF32 ⟨input⟩).toNat} {(ScaledScore.integerCastBits input).toNat}"
  | "scaled-arithmetic-release" | "scaled-arithmetic-checked" =>
    if p >= 2^32 || o >= 2^32 then throw "scaled operand outside i32 bit domain"
    pure (scaledArithmeticOutput (op == "scaled-arithmetic-checked")
      ⟨BitVec.ofNat 32 p⟩ ⟨BitVec.ofNat 32 o⟩)
  | "scaled-conversions-release" | "scaled-conversions-checked" =>
    if p >= 2^32 then throw "scaled input outside i32 bit domain"
    let input := BitVec.ofNat 32 p
    let checked := op == "scaled-conversions-checked"
    pure s!"{scaledConstructorOutput (ScaledScore.fromRaw checked input)}|{scaledConstructorOutput (ScaledScore.fromDiscDiff checked input)}|{scaledObservationOutput ⟨input⟩}"
  | "unchecked-u8-release" | "unchecked-u8-checked" =>
    if p >= 2^8 then throw "unchecked input outside u8 domain"
    uncheckedOutput (Square.fromU8Unchecked (op == "unchecked-u8-checked") (BitVec.ofNat 8 p))
  | "unchecked-u32-release" | "unchecked-u32-checked" =>
    if p >= 2^32 then throw "unchecked input outside u32 domain"
    uncheckedOutput (Square.fromU32Unchecked (op == "unchecked-u32-checked") (BitVec.ofNat 32 p))
  | "unchecked-usize-release" | "unchecked-usize-checked" =>
    uncheckedOutput (Square.fromUsizeUnchecked64 (op == "unchecked-usize-checked") (BitVec.ofNat 64 p))
  | "board-geometry-release" | "board-geometry-checked" =>
    if b.player &&& b.opponent != 0 then throw "board constructor requires disjointness"
    pure (boardGeometryOutput b)
  | "board-geometry-raw-release" => pure (boardGeometryOutput b)
  | "iterator-release" | "iterator-checked" =>
    let checked := op == "iterator-checked"
    let direct ← iteratorTraceOutput checked (bitboardIter b.player)
    let consumed ← iteratorTraceOutput checked (intoIterator b.player)
    let constructed ← iteratorTraceOutput checked (iteratorNew b.player)
    pure s!"{direct}|{consumed}|{constructed}"
  | "adjacent-release" | "adjacent-checked" =>
    if h : f < 64 then
      pure (if hasAdjacentBit b.player ⟨f, h⟩ then "1" else "0")
    else throw "invalid square discriminant"
  | "corners-release" | "corners-checked" =>
    pure s!"{(corners b.player).toNat} {(nonCorners b.player).toNat} {(cornerWeightedCount b.player).toNat} {(cornerStability b.player).toNat}"
  | "logic-release" | "logic-checked" =>
    pure s!"{(bitAnd b.player b.opponent).toNat} {(bitOr b.player b.opponent).toNat} {(bitXor b.player b.opponent).toNat} {(bitNot b.player).toNat} {(bitAndAssign b.player b.opponent).2.toNat} {(bitOrAssign b.player b.opponent).2.toNat} {(bitXorAssign b.player b.opponent).2.toNat}"
  | "shifts-release" | "shifts-checked" =>
    if f >= 2^32 then throw "shift count outside u32 domain"
    let count := BitVec.ofNat 32 f
    let checks := op == "shifts-checked"
    pure s!"{maskOutput (shiftLeft checks b.player count)}|{maskOutput (shiftRight checks b.player count)}|{shiftAssignmentOutput (shiftLeftAssign checks b.player count)}|{shiftAssignmentOutput (shiftRightAssign checks b.player count)}"
  | "delta-release" | "delta-checked" =>
    if f >= 2^32 then throw "delta outside u32 domain"
    pure (maskOutput (deltaSwap (op == "delta-checked") b.player b.opponent (BitVec.ofNat 32 f)))
  | "geometry-release" | "geometry-checked" =>
    pure s!"{(Bitboard.flipVertical b.player).toNat} {(Bitboard.flipHorizontal b.player).toNat} {(Bitboard.flipDiagA1H8 b.player).toNat} {(Bitboard.flipDiagA8H1 b.player).toNat} {(Bitboard.rotate90 b.player).toNat} {(Bitboard.rotate180 b.player).toNat} {(Bitboard.rotate270 b.player).toNat}"
  | "score-release" | "score-checked" =>
    if f >= 2^32 then throw "empty count outside u32 domain"
    let checks := op == "score-checked"
    pure s!"{scoreOutput (finalScore checks b)}|{scoreOutput (solve checks b (BitVec.ofNat 32 f))}"
  | "board-scaled-release" | "board-scaled-checked" | "board-scaled-raw-release" =>
    if f >= 2^32 then throw "empty count outside u32 domain"
    if op != "board-scaled-raw-release" && b.player &&& b.opponent != 0 then
      throw "board constructor requires disjointness"
    let checked := op == "board-scaled-checked"
    pure s!"{scaledConstructorOutput (finalScoreScaled checked b)}|{scaledConstructorOutput (solveScaled checked b (BitVec.ofNat 32 f))}"
  | "masks-release" | "masks-checked" =>
    if h : f < 64 then
      let anySq : AnySquare := ⟨f, h⟩
      let checks := op == "masks-checked"
      pure s!"ok:{(fromSquare anySq).toNat}|{maskOutput (squareBitboard checks checks anySq)}|{maskOutput (setSquare checks checks b.player anySq)}|{membershipOutput (containsSquare checks checks b.player anySq)}|{if isEmpty b.player then 1 else 0}"
    else throw "invalid square discriminant"
  | "get-moves-portable" => pure s!"{(Moves.getMovesPortable b.player b.opponent).toNat}"
  | "get-moves-avx2" => pure s!"{(Moves.getMovesAvx2 b.player b.opponent).toNat}"
  | "get-moves-avx512" => pure s!"{(Moves.getMovesAvx512 b.player b.opponent).toNat}"
  | "get-moves-and-potential-avx2" =>
    let (moves, potential) := Moves.getMovesAndPotentialAvx2 b.player b.opponent
    pure s!"{moves.toNat}|{potential.toNat}"
  | "get-moves-and-potential-avx512" =>
    let (moves, potential) := Moves.getMovesAndPotentialAvx512 b.player b.opponent
    pure s!"{moves.toNat}|{potential.toNat}"
  | "get-moves-and-potential-portable" =>
    let (moves, potential) := Moves.getMovesAndPotentialPortable b.player b.opponent
    pure s!"{moves.toNat}|{potential.toNat}"
  | "lowbits" => pure s!"{(clearLsb b.player).toNat} {if hasSingleBitNonzero b.player then 1 else 0} {(count b.player).toNat}"
  | "least-release" | "least-checked" =>
    match lsbSquareUnchecked (op == "least-checked") b.player with
    | .ok result => pure s!"ok:{result.val}"
    | .error reason => pure s!"error:{reason}"
  | "pop-release" | "pop-checked" =>
    match popLsb (op == "pop-checked") b.player with
    | .ok (result, rest) => pure s!"ok:{result.val} {rest.toNat}"
    | .error reason => pure s!"error:{reason}"
  | "place-release" | "place-checked" =>
    match makeMoveWithFlipped (op == "place-checked") b flipped sq with
    | .ok result => pure s!"ok:{result.player.toNat} {result.opponent.toNat}"
    | .error reason => pure s!"error:{reason}"
  | "square-iter" => pure (String.intercalate " " (Square.iter.map fun sq => toString (Square.index sq)))
  | "square-from-u8" =>
    if p >= 256 then throw "input outside u8 domain"
    match Square.fromU8 (BitVec.ofNat 8 p) with
    | some value => pure s!"some:{Square.index value}"
    | none => pure "none"
  | "square-coordinates" =>
    if h : p < 64 then
      let value : Square.Square := ⟨p, h⟩
      pure s!"{Square.index value}|{Square.file value}|{Square.rank value}"
    else throw "invalid square discriminant"
  | "square-from-coordinates" =>
    if p >= 256 || o >= 256 then throw "input outside u8 domain"
    match Square.fromFileRank (BitVec.ofNat 8 p) (BitVec.ofNat 8 o) with
    | .ok value => pure s!"ok:{Square.index value}"
    | .error reason => pure s!"error:{reason.message}"
  | "disc" =>
    let disc : Disc.Disc ← match p with
      | 0 => pure .empty
      | 1 => pure .black
      | 2 => pure .white
      | _ => throw "invalid disc"
    pure s!"{(Disc.toChar disc).toNat}"
  | "flip" => pure s!"{(applyFlip b.player flipped).toNat}"
  | "move" => pure s!"{(applyMove b.player flipped sq).toNat}"
  | "empty" => pure s!"{(emptySquares b).toNat} {(playerCount b).toNat} {(opponentCount b).toNat} {(emptyCount b).toNat}"
  | "switch" =>
    let result := switchPlayers b
    pure s!"{result.player.toNat} {result.opponent.toNat}"
  | _ => throw s!"unknown operation: {op}"

structure FlipScalarCase where
  sq : Square.Square
  masks : Fin 8 → Flip.Word
  player : Flip.Word
  opponent : Flip.Word
  expected : Flip.Word

def parseFlipScalarCase (line : String) : Except String FlipScalarCase := do
  let normalized := line.trimAscii.toString.replace "\t" " "
  let fields := (normalized.splitOn " ").filter (· != "")
  let ns ← fields.mapM fun field =>
    match field.toNat? with
    | some n => .ok n
    | none => .error s!"unparsable line: {line}"
  let [sqNat, m0, m1, m2, m3, m4, m5, m6, m7, p, o, exp] := ns
    | throw s!"unparsable line: {line}"
  if hsq : sqNat < 64 then
    let words := [m0, m1, m2, m3, m4, m5, m6, m7, p, o, exp]
    if words.any (· >= 2^64) then
      throw s!"input outside u64 domain: {line}"
    let maskVec : Vector Flip.Word 8 :=
      ⟨#[BitVec.ofNat 64 m0, BitVec.ofNat 64 m1, BitVec.ofNat 64 m2, BitVec.ofNat 64 m3,
         BitVec.ofNat 64 m4, BitVec.ofNat 64 m5, BitVec.ofNat 64 m6, BitVec.ofNat 64 m7], rfl⟩
    pure {
      sq := ⟨sqNat, by omega⟩
      masks := fun d => maskVec.get d
      player := BitVec.ofNat 64 p
      opponent := BitVec.ofNat 64 o
      expected := BitVec.ofNat 64 exp
    }
  else
    throw s!"square out of range (expected < 64, got {sqNat}): {line}"

def formatFlipOutcome : Except String Flip.Word → String
  | .ok w => toString w.toNat
  | .error msg => s!"error:{msg}"

def runFlipScalarCases (path : String) : IO UInt32 := do
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

private def moveCasesConfig (backend : String) : Except String Flip.Dispatch.TargetConfig :=
  match backend with
  | "neon-sha3" => .ok (.aarch64 true true)
  | "neon" => .ok (.aarch64 true false)
  | "avx512" => .ok (.x86_64 ⟨false, true, false⟩)
  | "avx2" => .ok (.x86_64 ⟨false, false, true⟩)
  | "portable" => .ok .other
  | _ => .error s!"unknown move generation backend: {backend}"

private def checkMoveCase (cfg : Flip.Dispatch.TargetConfig) (line : String) :
    Except String Unit := do
  let ws ← (line.trimAscii.toString.splitOn " ").mapM fun field => do
    let some n := field.toNat? | throw s!"invalid move generation case: {line}"
    if n ≥ 2^64 then throw s!"word outside u64 domain: {line}"
    pure (BitVec.ofNat 64 n)
  let [p, o, rustLegal, rustCombined, rustPotential] := ws
    | throw s!"invalid move generation case: {line}"
  let legal := Moves.legalMovesTarget cfg p o
  let (combined, potential) := Moves.legalMovesAndPotentialTarget cfg p o
  unless (legal, combined, potential) == (rustLegal, rustCombined, rustPotential) do
    throw s!"move generation mismatch: {line}; model {legal.toNat} {combined.toNat} {potential.toNat}"

private def runMoveCases (path : String) : IO UInt32 := do
  let lines ← IO.FS.lines path
  let some header := lines[0]? | throw (IO.userError "missing backend header")
  let backend := header.trimAscii.toString
  let cfg ← IO.ofExcept (moveCasesConfig backend)
  let cases := lines.extract 1 lines.size
  if cases.isEmpty then throw (IO.userError "missing move generation cases")
  for line in cases do
    IO.ofExcept (checkMoveCase cfg line)
  (← IO.getStdout).putStrLn s!"{backend}: {cases.size} move generation cases, 0 mismatches"
  return 0

open NeuralReversi.Eval.Activations in
private def activationBackend (name : String) : Except String Backend :=
  match name with
  | "scalar" => .ok .scalar
  | "avx2" => .ok .avx2
  | "avx512" => .ok .avx512
  | "neon" => .ok .neon
  | _ => .error s!"unknown activation backend: {name}"

private def parseActivationValues (text : String) : Except String (Array Int) :=
  ((text.splitOn " ").filter (· != "")).toArray.mapM fun field =>
    match field.toInt? with
    | some n => .ok n
    | none => .error s!"invalid activation value: {field}"

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

private def runActivationCases (path : String) : IO UInt32 := do
  let lines ← IO.FS.lines path
  let some header := lines[0]? | throw (IO.userError "missing activation backend header")
  let [fusedName, scReluName] := header.trimAscii.toString.splitOn " "
    | throw (IO.userError "expected fused and SCReLU backend names")
  let fused ← IO.ofExcept (activationBackend fusedName)
  let scRelu ← IO.ofExcept (activationBackend scReluName)
  let cases := lines.extract 1 lines.size
  if cases.isEmpty then throw (IO.userError "missing activation cases")
  for line in cases do
    IO.ofExcept (checkActivationCase fused scRelu line)
  (← IO.getStdout).putStrLn s!"{fusedName}/{scReluName}: {cases.size} activation cases, 0 mismatches"
  return 0

open NeuralReversi.EndgameTiming in
private def checkEndgameTimingCase (line : String) : Except String Unit := do
  let values ← (line.trimAscii.toString.splitOn " ").mapM fun field =>
    match field.toNat? with
    | some n => if n < 2 ^ 64 then .ok n else .error s!"outside u64 range: {field}"
    | none => .error s!"invalid timing value: {field}"
  let [elapsed, budget, factorBits, elapsedBits, budgetBits, productBits, permits] := values
    | throw s!"invalid endgame timing case: {line}"
  let some factor := decode (BitVec.ofNat 64 factorBits)
    | throw s!"unsupported negative factor: {line}"
  unless decode (BitVec.ofNat 64 elapsedBits) = some (.finite (castUnits elapsed)) do
    throw s!"elapsed cast disagrees with the binary64 model: {line}"
  unless decode (BitVec.ofNat 64 budgetBits) = some (.finite (castUnits budget)) do
    throw s!"budget cast disagrees with the binary64 model: {line}"
  unless decode (BitVec.ofNat 64 productBits) = some (elapsedProduct elapsed factor) do
    throw s!"projected time disagrees with the binary64 model: {line}"
  unless permits = (if underBudget budget elapsed factor then 1 else 0) do
    throw s!"budget decision disagrees with the binary64 model: {line}"

private def runEndgameTimingCases (path : String) : IO UInt32 := do
  let cases ← IO.FS.lines path
  if cases.isEmpty then throw (IO.userError "missing endgame timing cases")
  for line in cases do
    IO.ofExcept (checkEndgameTimingCase line)
  (← IO.getStdout).putStrLn s!"{cases.size} endgame timing cases, 0 mismatches"
  return 0

private def checkEndgameProbCutCase (line : String) : Except String Unit := do
  let [beta, verificationBits, screenBits, upperDelta, lowerDelta, pcBeta, screenBeta] :=
    line.splitOn " " | throw s!"malformed ProbCut window case: {line}"
  let some beta := beta.toInt? | throw s!"invalid ProbCut beta: {line}"
  let some verificationBits := verificationBits.toNat? | throw s!"invalid verification margin: {line}"
  let some screenBits := screenBits.toNat? | throw s!"invalid screen margin: {line}"
  unless verificationBits < 2 ^ 64 && screenBits < 2 ^ 64 do
    throw s!"ProbCut margin bits exceed binary64: {line}"
  let verification := BitVec.ofNat 64 verificationBits
  let screen := BitVec.ofNat 64 screenBits
  unless some (EndgameProbCut.roundedMargin true verification) == upperDelta.toInt? do
    throw s!"verification ceil and cast disagree with the model: {line}"
  unless some (EndgameProbCut.roundedMargin false screen) == lowerDelta.toInt? do
    throw s!"screen floor and cast disagree with the model: {line}"
  let expected := EndgameProbCut.nativeThresholds beta verification screen
  unless some expected.verification == pcBeta.toInt? && some expected.screen == screenBeta.toInt? do
    throw s!"ProbCut thresholds disagree with the model: {line}"

private def runEndgameProbCutCases (path : String) : IO UInt32 := do
  let cases ← IO.FS.lines path
  if cases.isEmpty then throw (IO.userError "missing endgame ProbCut cases")
  for line in cases do
    IO.ofExcept (checkEndgameProbCutCase line)
  (← IO.getStdout).putStrLn s!"{cases.size} endgame ProbCut cases, 0 mismatches"
  return 0

private def checkMidgameProbCutCase (line : String) : Except String Unit := do
  let [depth, shallow, beta, t, cutNode, mean, sigma, mean0, sigma0, pcBeta, screen] :=
    line.splitOn " " | throw s!"malformed midgame ProbCut case: {line}"
  let some depth := depth.toNat? | throw s!"invalid deep depth: {line}"
  unless depth < 2 ^ 32 && some (MidgameProbCut.shallowDepth depth) == shallow.toNat? do
    throw s!"midgame ProbCut shallow depth disagrees with the model: {line}"
  let some beta := beta.toInt? | throw s!"invalid midgame ProbCut beta: {line}"
  let cutNode ← match cutNode with
    | "0" => pure false
    | "1" => pure true
    | _ => throw s!"invalid midgame ProbCut node type: {line}"
  let parseFloat (word : String) : Except String Int := do
    let some bits := word.toNat? | throw s!"invalid binary64 bits: {word}"
    unless bits < 2 ^ 64 do throw s!"binary64 word out of range: {word}"
    let some units := Binary64Arithmetic.finiteUnits (BitVec.ofNat 64 bits)
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

private def runMidgameProbCutCases (path : String) : IO UInt32 := do
  let cases ← IO.FS.lines path
  if cases.isEmpty then throw (IO.userError "missing midgame ProbCut cases")
  for line in cases do
    IO.ofExcept (checkMidgameProbCutCase line)
  (← IO.getStdout).putStrLn s!"{cases.size} midgame ProbCut cases, 0 mismatches"
  return 0

open PrincipalVariation in
private def checkPvStackCase (line : String) : Except String Unit := do
  let [op, ply, square, parent, child, nextParent, nextChild] :=
    line.splitOn " " | throw s!"malformed PV stack case: {line}"
  let some ply := ply.toNat? | throw s!"invalid PV stack ply: {line}"
  unless ply < capacity do throw s!"PV ply out of range: {line}"
  let some square := square.toNat? | throw s!"invalid PV square: {line}"
  let parse (row : String) : Except String Buffer := do
    let entries := row.splitOn ","
    unless entries.length = capacity do throw s!"wrong PV row length: {row}"
    entries.mapM fun entry =>
      match entry.toNat? with
      | some index =>
        if fits : index < 64 then pure (some ⟨index, fits⟩)
        else if index = 64 then pure none else throw s!"invalid PV entry: {entry}"
      | none => throw s!"invalid PV entry: {entry}"
  let parent ← parse parent
  let child ← parse child
  let actualParent ← parse nextParent
  let actualChild ← parse nextChild
  let expected ← match op with
    | "u" =>
      if fits : square < 64 then
        if ply + 1 < capacity then pure (prepend ⟨square, fits⟩ child parent, child)
        else throw "cannot prepend without a child row"
      else throw "invalid placement square"
    | "p" =>
      let stack := ((List.replicate capacity []).set ply parent).set (ply + 1) child
      let prepared := prepare stack ply
      pure (prepared[ply]?.getD [], prepared[ply + 1]?.getD child)
    | _ => throw s!"unknown PV operation: {op}"
  unless expected.1 == actualParent do throw s!"parent PV buffer disagrees with the model: {line}"
  unless expected.2 == actualChild do throw s!"child PV buffer disagrees with the model: {line}"

private def runPvStackCases (path : String) : IO UInt32 := do
  let cases ← IO.FS.lines path
  if cases.isEmpty then throw (IO.userError "missing PV stack cases")
  for line in cases do
    IO.ofExcept (checkPvStackCase line)
  (← IO.getStdout).putStrLn s!"{cases.size} PV stack cases, 0 mismatches"
  return 0

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
  let parse (field : String) : Except String Int := do
    let some value := field.toInt? | throw s!"invalid aspiration integer: {field}"
    pure value
  let natural (field : String) : Except String Nat := do
    let some value := field.toNat? | throw s!"invalid aspiration natural: {field}"
    pure value
  match line.splitOn " " with
  | ["w", depth, helper, raw, a, b, d] =>
    let history ← if raw = "none" then pure none else do
      let [avg, mss] := raw.splitOn "," | throw s!"invalid root score history: {raw}"
      pure (some ⟨← parse avg, ← parse mss⟩)
    let start := initial (← natural depth) history (← natural helper)
    unless start == ⟨← parse a, ← parse b, ← parse d⟩ do
      throw s!"initial aspiration window disagrees with the model: {line}"
  | ["r", lo, hi, width, style, abortAt, a, b, score, clean, windows] =>
    let start : Attempt := ⟨← parse lo, ← parse hi, ← parse width⟩
    let trials := midgameAspirationTrial (← natural style) (← natural abortAt)
    let some result := run trials 16 0 start | throw s!"aspiration retries exhausted: {line}"
    unless result.attempt.alpha == (← parse a) && result.attempt.beta == (← parse b) &&
        result.trial.score == (← parse score) && (if result.clean then "1" else "0") == clean do
      throw s!"final aspiration result disagrees with the model: {line}"
    unless ";".intercalate (midgameAttemptWindows trials 16 0 start) == windows do
      throw s!"aspiration attempt windows disagree with the model: {line}"
  | _ => throw s!"invalid midgame aspiration case: {line}"

private def runMidgameAspirationCases (path : String) : IO UInt32 := do
  let cases ← IO.FS.lines path
  if cases.isEmpty then throw (IO.userError "missing midgame aspiration cases")
  for line in cases do
    IO.ofExcept (checkMidgameAspirationCase line)
  (← IO.getStdout).putStrLn s!"{cases.size} midgame aspiration cases, 0 mismatches"
  return 0

open LazySmp in
private def checkLazySmpCase (line : String) : Except String Unit := do
  match line.splitOn " " with
  | ["s", helper, depth, skipped] =>
    let some helper := helper.toNat? | throw s!"invalid helper index: {line}"
    let some depth := depth.toNat? | throw s!"invalid helper depth plus ply: {line}"
    let expected := if skipsDepth helper depth then "1" else "0"
    unless expected == skipped do throw s!"Lazy SMP schedule disagrees with the model: {line}"
  | "v" :: picked :: entries =>
    let ballots ← entries.mapM fun entry => do
      let [square, score, depth] := entry.splitOn "," | throw s!"invalid ballot: {entry}"
      let some square := square.toNat? | throw s!"invalid vote square: {entry}"
      let some score := score.toInt? | throw s!"invalid vote score: {entry}"
      let some depth := depth.toNat? | throw s!"invalid vote depth: {entry}"
      if fits : square < 64 then pure (⟨⟨square, fits⟩, score, depth⟩ : Ballot)
      else throw s!"vote square is out of range: {entry}"
    let expected := (winner ballots).map (fun result => toString result.1)
    unless expected.getD "-" == picked do throw s!"Lazy SMP winner disagrees with the model: {line}"
  | _ => throw s!"invalid Lazy SMP case: {line}"

private def runLazySmpCases (path : String) : IO UInt32 := do
  let cases ← IO.FS.lines path
  if cases.isEmpty then throw (IO.userError "missing Lazy SMP cases")
  for line in cases do
    IO.ofExcept (checkLazySmpCase line)
  (← IO.getStdout).putStrLn s!"{cases.size} Lazy SMP cases, 0 mismatches"
  return 0

private def checkFiniteScoreCase (line : String) : Except String Unit := do
  let [score, actual] := line.splitOn " " | throw s!"invalid finite score case: {line}"
  let some score := score.toInt? | throw s!"invalid i32 score: {line}"
  unless -(2^31 : Int) ≤ score && score < (2^31 : Int) do
    throw s!"score outside i32 domain: {line}"
  let expected := if MidgameScoreRange.isFinite (BitVec.ofInt 32 score) then "1" else "0"
  unless expected == actual do throw s!"finite score admission disagrees with the model: {line}"

private def runFiniteScoreCases (path : String) : IO UInt32 := do
  let cases ← IO.FS.lines path
  if cases.isEmpty then throw (IO.userError "missing finite score cases")
  for line in cases do
    IO.ofExcept (checkFiniteScoreCase line)
  (← IO.getStdout).putStrLn s!"{cases.size} finite score cases, 0 mismatches"
  return 0

open EvalCache in
private def checkEvalCacheCase (line : String) : Except String Unit := do
  let natural (field : String) : Except String Nat := do
    let some value := field.toNat? | throw s!"invalid cache integer: {field}"
    pure value
  let word (field : String) : Except String Word := do
    let value ← natural field
    if value < 2 ^ 64 then pure (BitVec.ofNat 64 value)
    else throw s!"cache word exceeds u64: {field}"
  match line.splitOn " " with
  | ["g", pointerBits, size, key, count, index, way] =>
    let size ← natural size
    let key ← word key
    unless checkedCount (← natural pointerBits) size == some (← natural count) &&
        bucket size key == (← natural index) && victim key == (← natural way) do
      throw s!"evaluation cache address disagrees with the model: {line}"
  | ["e", key, signed, payload] =>
    let some score := signed.toInt? | throw s!"invalid cache score: {signed}"
    unless encode (← word key) (BitVec.ofInt 32 score) == some (← word payload) do
      throw s!"evaluation cache encoding disagrees with the model: {line}"
  | ["p", key, replacement, a, b, c, d, way, score] =>
    let replacement ← natural replacement
    let layout : Vector Word 4 := #v[← word a, ← word b, ← word c, ← word d]
    if fits : replacement < 4 then
      let (slot, signed) := match select (← word key) (fun i => layout[i]) ⟨replacement, fits⟩ with
        | .hit i value => (i.val, toString value)
        | .miss i => (i.val, "-")
      unless slot == (← natural way) && signed == score do
        throw s!"evaluation cache probe disagrees with the model: {line}"
    else throw s!"cache victim way is out of range: {line}"
  | _ => throw s!"invalid evaluation cache case: {line}"

private def runEvalCacheCases (path : String) : IO UInt32 := do
  let cases ← IO.FS.lines path
  if cases.isEmpty then throw (IO.userError "missing evaluation cache cases")
  for line in cases do
    IO.ofExcept (checkEvalCacheCase line)
  (← IO.getStdout).putStrLn s!"{cases.size} evaluation cache cases, 0 mismatches"
  return 0

open AlignedBuffer in
private def checkAlignedBufferCase (line : String) : Except String Unit := do
  let natural (field : String) : Except String Nat := do
    let some value := field.toNat? | throw s!"invalid buffer integer: {field}"
    pure value
  let flag (field : String) : Except String Bool :=
    match field with
    | "0" => pure false
    | "1" => pure true
    | _ => throw s!"invalid buffer flag: {field}"
  let indices (field : String) : Except String (List Nat) :=
    if field = "-" then pure [] else (field.splitOn ",").mapM natural
  match line.splitOn " " with
  | ["l", pointerBits, count, itemSize, result, bytes] =>
    let model := checkedLayout (← natural pointerBits) (← natural count) (← natural itemSize)
    let (expected, size) := match model with
      | .multiplyOverflow => ("overflow", 0)
      | .noBytes => ("empty", 0)
      | .invalidLayout => ("layout", 0)
      | .allocated value => ("alloc", value)
    unless expected == result && size == (← natural bytes) do
      throw s!"aligned buffer layout disagrees with the model: {line}"
  | ["f", large, count, ending, stop, failingDrop, ready, panic, allocs, freed, pages, values, drops] =>
    let status ← match ending with
      | "all" | "drop-panic" => pure FillStatus.complete
      | "short" => pure FillStatus.short
      | "panic" => pure FillStatus.panic
      | _ => throw s!"invalid buffer iterator outcome: {ending}"
    let actual : Observation :=
      ⟨← flag ready, ← flag panic, ← natural allocs, ← natural freed, ← natural pages,
        ← indices values, ← indices drops⟩
    let expected := fixture (← flag large) (← natural count) status (← natural stop)
      (← natural failingDrop) (ending == "drop-panic")
    unless expected == actual do throw s!"aligned buffer cleanup disagrees with the model: {line}"
  | _ => throw s!"invalid aligned buffer case: {line}"

private def runAlignedBufferCases (path : String) : IO UInt32 := do
  let cases ← IO.FS.lines path
  if cases.isEmpty then throw (IO.userError "missing aligned buffer cases")
  for line in cases do
    IO.ofExcept (checkAlignedBufferCase line)
  (← IO.getStdout).putStrLn s!"{cases.size} aligned buffer cases, 0 mismatches"
  return 0

private def patternNatural (field : String) : Except String Nat :=
  match field.toNat? with
  | some n => .ok n
  | none => .error s!"invalid pattern feature value: {field}"

private def patternLanes (text : String) : Except String (Array Nat) := do
  let lanes ← (text.splitOn ",").toArray.mapM patternNatural
  unless lanes.size == 32 && lanes.all (· < 65536) do throw "invalid u16 pattern vector"
  return lanes

private def patternBits (text : String) : Except String Bits := do
  let n ← patternNatural text
  if n ≥ 2^64 then throw "pattern bitboard exceeds u64"
  return BitVec.ofNat 64 n

private def patternLocation (sq flip side ply : String) :
    Except String (Square.Square × Bits × Bool) := do
  let index ← patternNatural sq
  let bits ← patternBits flip
  let side ← patternNatural side
  let ply ← patternNatural ply
  unless side < 2 && Eval.PatternUpdate.updateAllowed ply bits do throw "invalid pattern update"
  if h : index < 64 then return (⟨index, h⟩, bits, side == 1)
  else throw "invalid pattern update square"

open Eval.PatternUpdate in
private def checkPatternFeatureCase (line : String) : Except String Unit := do
  match (line.splitOn " ").filter (· != "") with
  | ["geometry", index, coords] =>
    let index ← patternNatural index
    let actual ← (coords.splitOn ",").mapM patternNatural
    if hi : index < 32 then
      unless pattern ⟨index, hi⟩ == actual do throw s!"pattern geometry disagrees: {line}"
    else throw "invalid pattern index"
  | ["board", player, opponent, sq, flip, side, ply, p₀, o₀, p₁, o₁] =>
    let b : Board := ⟨← patternBits player, ← patternBits opponent⟩
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

private def runPatternFeatureCases (path : String) : IO UInt32 := do
  let cases ← IO.FS.lines path
  unless cases.size == 2592 do throw (IO.userError "expected 2592 pattern feature cases")
  for line in cases do
    IO.ofExcept (checkPatternFeatureCase line)
  (← IO.getStdout).putStrLn s!"{cases.size} pattern feature cases, 0 mismatches"
  return 0

def main (args : List String) : IO UInt32 := do
  let stdout ← IO.getStdout
  if let cmd :: rest := args then
    if cmd == "move-bound-cnf" || cmd == "move-bound-check" then
      let index := rest.head?.bind (·.toNat?)
      if let some bound := index then
        if bound < 64 then
          let formula := LegalMoves.MoveBound.dimacsFormula bound
          match cmd, rest with
          | "move-bound-cnf", [_] =>
            stdout.putStr formula.dimacs
            return 0
          | "move-bound-check", [_, path] =>
            let proof ← Std.Tactic.BVDecide.LRAT.loadLRATProof path
            if Std.Tactic.BVDecide.LRAT.check proof formula then
              stdout.putStrLn s!"LRAT verified: at most {bound} legal moves"
              return 0
            (← IO.getStderr).putStrLn "LRAT rejected for this move bound"
            return 1
          | _, _ => pure ()
      (← IO.getStderr).putStrLn "usage: revmodel move-bound-cnf <bound 0..63> or move-bound-check <bound 0..63> <proof.lrat>"
      return 2
  if args == ["unicode-whitespace"] then
    stdout.putStrLn (String.intercalate " " (Square.unicodeWhitespaceScalars.map toString))
    return 0
  if args == ["registry"] then
    stdout.putStrLn (toJson entries).compress
    return 0
  if args == ["utf8-square-chunks"] then
    stdout.putStrLn (utf8ChunkDiagnostic [])
    for first in [:256] do
      stdout.putStrLn (utf8ChunkDiagnostic [BitVec.ofNat 8 first])
    for first in [:256] do
      for second in [:256] do
        stdout.putStrLn (utf8ChunkDiagnostic [BitVec.ofNat 8 first, BitVec.ofNat 8 second])
    return 0
  if let ["activation-cases", path] := args then
    return ← runActivationCases path
  if let ["move-cases", path] := args then
    return ← runMoveCases path
  if let ["endgame-timing-cases", path] := args then
    return ← runEndgameTimingCases path
  if let ["endgame-probcut-cases", path] := args then
    return ← runEndgameProbCutCases path
  if let ["midgame-probcut-cases", path] := args then
    return ← runMidgameProbCutCases path
  if let ["pv-stack-cases", path] := args then
    return ← runPvStackCases path
  if let ["midgame-aspiration-cases", path] := args then
    return ← runMidgameAspirationCases path
  if let ["lazy-smp-cases", path] := args then
    return ← runLazySmpCases path
  if let ["finite-score-cases", path] := args then
    return ← runFiniteScoreCases path
  if let ["eval-cache-cases", path] := args then
    return ← runEvalCacheCases path
  if let ["aligned-buffer-cases", path] := args then
    return ← runAlignedBufferCases path
  if let ["pattern-feature-cases", path] := args then
    return ← runPatternFeatureCases path
  if let ["flip-scalar-cases", path] := args then
    return ← runFlipScalarCases path
  if let some name := args.head? then
    if ["flip-scalar-cases", "move-cases", "activation-cases", "endgame-timing-cases",
        "endgame-probcut-cases", "midgame-probcut-cases", "pv-stack-cases",
        "midgame-aspiration-cases", "lazy-smp-cases", "finite-score-cases",
        "eval-cache-cases", "aligned-buffer-cases", "pattern-feature-cases"].contains name then
      (← IO.getStderr).putStrLn s!"usage: revmodel {name} <path>"
      return 2
  let stdin ← IO.getStdin
  repeat
    let line ← stdin.getLine
    if line.isEmpty then break
    match runCase line.trimAscii.toString with
    | .ok output => stdout.putStrLn output
    | .error message =>
      (← IO.getStderr).putStrLn message
      return 1
  return 0

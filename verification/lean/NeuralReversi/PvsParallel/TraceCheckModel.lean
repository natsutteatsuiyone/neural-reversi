module
public import Std.Data.HashMap.Basic
public import Std.Data.HashSet.Basic
public import NeuralReversi.JsonLite
public import NeuralReversi.Board.MovesModel
public import NeuralReversi.Board.Counting
public import NeuralReversi.Board.ScoreProperties
public import NeuralReversi.Flip.Dispatch.Model
public import NeuralReversi.PvsParallel.Prefix
public import NeuralReversi.TranspositionTable.LookupCut
public import NeuralReversi.TranspositionTable.Snapshot
public import NeuralReversi.TranspositionTable.Packing

@[expose] public section

namespace NeuralReversi.PvsParallel.TraceCheck

open JsonLite (FromJson fromJson? expectObj getField getFieldOpt)
open PvsPostpone (Queue step)
open PvsMoveLoop (ChildSearch moveScore scoreInf scoreMax)
open Prefix.Serial (Outcome)
open Threading.SplitWindow (State choosesMove marksCutoff publish reSearch)

def moveConfig : Bitboard.Moves.X86Config := ⟨false, false⟩
def flipConfig : Flip.Dispatch.X86Config := ⟨false, false, false⟩
def exactSelectivity : TranspositionTable.Selectivity := ⟨5, by decide⟩

structure ChildReturn where
  frame : Nat
  square : Nat
  pv : Bool
  alpha : Int
  beta : Int
  score : Int

instance : FromJson ChildReturn where
  fromJson? v := do
    let obj ← expectObj v
    pure {
      frame := ← getField obj "frame"
      square := ← getField obj "square"
      pv := ← getField obj "pv"
      alpha := ← getField obj "alpha"
      beta := ← getField obj "beta"
      score := ← getField obj "score"
    }

def findChild (calls : List ChildReturn) (move : Nat) (pv : Bool) (alpha beta : Int) : Option Int :=
  (calls.find? fun child =>
    child.square == move && child.pv == pv && child.alpha == alpha && child.beta == beta).map (·.score)

def observe (calls : List ChildReturn) : ChildSearch Nat :=
  fun move pv alpha beta => (findChild calls move pv alpha beta).getD 0

def checkedCalls (calls : List ChildReturn) (pv first : Bool) (move : Nat) (alpha beta : Int) : Bool :=
  let scout := findChild calls move false (-(alpha + 1)) (-alpha)
  let wide := (findChild calls move true (-beta) (-alpha)).isSome
  if pv && first then wide
  else scout.isSome && (if pv = true ∧ alpha < -scout.getD 0 then wide else true)

structure Selection where
  index : Nat
  freshEnd : Nat
  square : Nat
  score : Option Int

instance : FromJson Selection where
  fromJson? v := do
    let obj ← expectObj v
    pure {
      index := ← getField obj "index"
      freshEnd := ← getField obj "freshEnd"
      square := ← getField obj "square"
      score := ← getFieldOpt obj "score"
    }

structure SerialResult where
  visited : List Nat
  remaining : Queue Nat
  outcome : Outcome

def replay (calls : List ChildReturn) (pv : Bool) (beta : Int) (handoff : Bool) :
    List Selection → Queue Nat → State → Nat → Option SerialResult
  | [], queue, state, _ =>
    if handoff then some ⟨[], queue, .handoff state⟩
    else if queue.pending.isEmpty then some ⟨[], queue, .returned state.best⟩
    else none
  | choice :: rest, queue, state, freshEnd => do
    if choice.index != queue.idx || choice.freshEnd != freshEnd ||
        queue.fresh.length != freshEnd - queue.idx then none
    else match choice.score with
    | none =>
      if queue.fresh.head? != some choice.square then none
      else match step true queue with
        | some (none, next) => replay calls pv beta handoff rest next state (freshEnd - 1)
        | _ => none
    | some score =>
      let first := queue.idx == 0
      if score != moveScore pv first (observe calls) choice.square state.alpha beta ||
          !checkedCalls calls pv first choice.square state.alpha beta then none
      else match step false queue with
        | some (some move, next) =>
          if move != choice.square then none
          else if marksCutoff pv beta state score then
            if rest.isEmpty && !handoff then some ⟨[move], next, .returned score⟩ else none
          else (replay calls pv beta handoff rest next (publish pv beta state score) freshEnd).map fun after =>
            { after with visited := move :: after.visited }
        | _ => none

structure WideReturn where
  frame : Nat
  alpha : Int
  score : Int

instance : FromJson WideReturn where
  fromJson? v := do
    let obj ← expectObj v
    pure {
      frame := ← getField obj "frame"
      alpha := ← getField obj "alpha"
      score := ← getField obj "score"
    }

structure ReportRow where
  index : Nat
  square : Nat
  worker : Nat
  scoutFrame : Nat
  scoutAlpha : Int
  scout : Int
  wide : Option WideReturn
  score : Int
  bestAfter : Int
  alphaAfter : Int
  moveAfter : Option Nat

instance : FromJson ReportRow where
  fromJson? v := do
    let obj ← expectObj v
    pure {
      index := ← getField obj "index"
      square := ← getField obj "square"
      worker := ← getField obj "worker"
      scoutFrame := ← getField obj "scoutFrame"
      scoutAlpha := ← getField obj "scoutAlpha"
      scout := ← getField obj "scout"
      wide := ← getFieldOpt obj "wide"
      score := ← getField obj "score"
      bestAfter := ← getField obj "bestAfter"
      alphaAfter := ← getField obj "alphaAfter"
      moveAfter := ← getFieldOpt obj "moveAfter"
    }

def ReportRow.fresh (row : ReportRow) : Int := (row.wide.map (·.alpha)).getD row.scoutAlpha
def ReportRow.wideScore (row : ReportRow) : Int := (row.wide.map (·.score)).getD 0

def ReportRow.selected (row : ReportRow) (pv : Bool) : Threading.SplitResult.Report :=
  Threading.SplitWindow.selectReport pv row.index row.scoutAlpha row.fresh row.scout row.wideScore

def ReportRow.scoutCall (row : ReportRow) : ChildReturn :=
  ⟨row.scoutFrame, row.square, false, -(row.scoutAlpha + 1), -row.scoutAlpha, row.scout⟩

def ReportRow.wideCall (row : ReportRow) (beta : Int) (wide : WideReturn) : ChildReturn :=
  ⟨wide.frame, row.square, true, -beta, -wide.alpha, wide.score⟩

def reportAccepted (row : ReportRow) (moves : List Nat) (start : Nat) (seen : List Nat)
    (alphas : List Int) (pv : Bool) (beta : Int) (locked : State) (chosen : Option Nat) : Bool :=
  (((decide (start ≤ row.index) && !seen.contains row.index) &&
    (alphas.contains row.scoutAlpha && row.wide.isSome == reSearch pv row.scoutAlpha row.scout &&
      (row.wide.all fun wide => alphas.contains wide.alpha))) &&
    (row.score == (row.selected pv).score && row.alphaAfter == (publish pv beta locked row.score).alpha &&
      row.bestAfter == (publish pv beta locked row.score).best)) &&
  (moves[row.index]? == some row.square &&
    row.moveAfter == (if choosesMove locked row.score then some row.square else chosen))

def replayReports (moves : List Nat) (start : Nat) (pv : Bool) (beta : Int) :
    List ReportRow → List Nat → List Int → State → Option Nat → Option (State × Option Nat)
  | [], _, _, locked, chosen => some (locked, chosen)
  | row :: tail, seen, alphas, locked, chosen =>
    if reportAccepted row moves start seen alphas pv beta locked chosen then
      replayReports moves start pv beta tail (row.index :: seen) (row.alphaAfter :: alphas)
        (publish pv beta locked row.score) row.moveAfter
    else none

structure Handoff where
  start : Nat
  moves : List Nat
  alpha : Int
  best : Int
  seed : Option Nat
  reports : List ReportRow

instance : FromJson Handoff where
  fromJson? v := do
    let obj ← expectObj v
    pure {
      start := ← getField obj "start"
      moves := ← getField obj "moves"
      alpha := ← getField obj "alpha"
      best := ← getField obj "best"
      seed := ← getFieldOpt obj "seed"
      reports := ← getField obj "reports"
    }

structure SlotStamp where
  scope : Nat
  slot : Nat
  sequence : Nat

instance : FromJson SlotStamp where
  fromJson? v := do
    let obj ← expectObj v
    pure {
      scope := ← getField obj "scope"
      slot := ← getField obj "slot"
      sequence := ← getField obj "sequence"
    }

def SlotStamp.key (stamp : SlotStamp) : Nat × Nat × Nat :=
  (stamp.scope, stamp.slot, stamp.sequence)

structure Node where
  player : Nat
  opponent : Nat
  root : Bool
  pv : Bool
  alpha : Int
  beta : Int
  generated : List Nat
  moves : List Nat
  calls : List ChildReturn
  selections : List Selection
  handoff : Option Handoff
  result : Int
  move : Option Nat
  publication : Option SlotStamp

instance : FromJson Node where
  fromJson? v := do
    let obj ← expectObj v
    pure {
      player := ← getField obj "player"
      opponent := ← getField obj "opponent"
      root := ← getField obj "root"
      pv := ← getField obj "pv"
      alpha := ← getField obj "alpha"
      beta := ← getField obj "beta"
      generated := ← getField obj "generated"
      moves := ← getField obj "moves"
      calls := ← getField obj "calls"
      selections := ← getField obj "selections"
      handoff := ← getFieldOpt obj "handoff"
      result := ← getField obj "result"
      move := ← getFieldOpt obj "move"
      publication := ← getFieldOpt obj "publication"
    }

def Node.initial (node : Node) : State := ⟨node.alpha, -scoreInf⟩
def Node.queue (node : Node) : Queue Nat := ⟨0, node.moves, []⟩
def Node.board (node : Node) : Board.Board :=
  ⟨BitVec.ofNat 64 node.player, BitVec.ofNat 64 node.opponent⟩
def Node.legalMask (node : Node) : Bitboard.Bits := Board.getMoves moveConfig node.board
def Node.coversAll (node : Node) : Bool :=
  node.moves.length == ((List.range 64).filter node.legalMask.getLsbD).length

def candidatesAgree (node : Node) : Bool :=
  decide (node.player < 2 ^ 64 ∧ node.opponent < 2 ^ 64 ∧ node.moves.Nodup) &&
    (node.board.player &&& node.board.opponent) == 0 && !node.moves.isEmpty &&
    (node.root || node.coversAll) &&
    node.generated.mergeSort == node.moves.mergeSort

def covered (split : Handoff) (beta best : Int) : Bool :=
  decide (beta ≤ best ∨ PvsMoveLoop.scoreMax ≤ best) ||
    (Threading.SplitResult.slots split.start split.moves.length).all fun index =>
      split.reports.any (·.index == index)

def matchEnd (node : Node) (visited : List Nat) (remaining : Queue Nat)
    (picked : Option Nat) : Outcome → Option Handoff → Bool
  | .returned score, none => score == node.result && picked == node.move
  | .handoff state, some split =>
    ((split.start == remaining.idx && split.alpha == state.alpha && split.best == state.best &&
        decide (split.start + 2 ≤ split.moves.length) && !(node.pv && split.start == 0)) &&
        split.moves == visited ++ remaining.pending && split.seed == picked) &&
      match replayReports split.moves split.start node.pv node.beta split.reports [] [split.alpha] state split.seed with
      | some (final, chosen) => final.best == node.result && chosen == node.move && covered split node.beta final.best
      | none => false
  | _, _ => false

def Node.check (node : Node) : Bool :=
  decide (-scoreInf ≤ node.alpha ∧ node.alpha < node.beta ∧ node.beta ≤ scoreInf) &&
    (node.pv || node.beta == node.alpha + 1) &&
    candidatesAgree node && node.moves.all node.legalMask.getLsbD &&
    match replay node.calls node.pv node.beta node.handoff.isSome node.selections node.queue node.initial node.moves.length with
    | none => false
    | some serial =>
      let picked := Prefix.Serial.bestMove node.pv (observe node.calls) node.beta serial.visited true node.initial none
      matchEnd node serial.visited serial.remaining picked serial.outcome node.handoff

structure FrameReturn where
  parent : Nat
  worker : Nat
  depth : Nat
  alpha : Int
  beta : Int
  pv : Bool
  player : Nat
  opponent : Nat
  score : Int

instance : FromJson FrameReturn where
  fromJson? v := do
    let obj ← expectObj v
    pure {
      parent := ← getField obj "parent"
      worker := ← getField obj "worker"
      depth := ← getField obj "depth"
      alpha := ← getField obj "alpha"
      beta := ← getField obj "beta"
      pv := ← getField obj "pv"
      player := ← getField obj "player"
      opponent := ← getField obj "opponent"
      score := ← getField obj "score"
    }

def FrameReturn.board (frame : FrameReturn) : Board.Board :=
  ⟨BitVec.ofNat 64 frame.player, BitVec.ofNat 64 frame.opponent⟩

def FrameReturn.sane (frame : FrameReturn) : Bool :=
  decide (frame.player < 2 ^ 64 ∧ frame.opponent < 2 ^ 64 ∧
    frame.board.player &&& frame.board.opponent = 0) &&
    frame.depth == (Board.emptyCount frame.board).toNat &&
    decide (-scoreInf ≤ frame.alpha ∧ frame.alpha < frame.beta ∧ frame.beta ≤ scoreInf) &&
    (frame.pv || frame.beta == frame.alpha + 1) &&
    decide (-scoreInf < frame.score ∧ frame.score < scoreInf)

def FrameReturn.onMove (child : FrameReturn) (fc : Flip.Dispatch.X86Config) (board : Board.Board)
    (square : Nat) : Bool :=
  if inside : square < 64 then
    let sq : Bitboard.Square := ⟨square, inside⟩
    let flipped := Flip.Dispatch.resultValue fc sq board.player board.opponent
    (Board.getMoves moveConfig board).getLsbD square &&
      decide (child.board = Board.place board flipped sq)
  else false

def FrameReturn.windowMatches (frame : FrameReturn) (pv : Bool) (alpha beta score : Int) : Bool :=
  frame.pv == pv && frame.alpha == alpha && frame.beta == beta && frame.score == score

def FrameReturn.checkWith (frame : FrameReturn) (exceeds : Int → Bool) : Bool :=
  decide (-scoreInf < frame.score ∧ frame.score < scoreInf) &&
    (if frame.score ≤ frame.alpha then exceeds frame.score == false else true) &&
    (if frame.beta ≤ frame.score then exceeds (frame.score - 1) else true) &&
    (if frame.alpha < frame.score ∧ frame.score < frame.beta then
      exceeds (frame.score - 1) && (exceeds frame.score == false) else true)

def FrameReturn.checkBound (frame : FrameReturn) (value : Int) : Bool :=
  frame.checkWith (fun cut => decide (cut < value))

def FrameReturn.isTerminal (frame : FrameReturn) : Bool :=
  (Board.hasLegalMoves moveConfig frame.board == false) &&
    (Board.hasLegalMoves moveConfig (Board.switchPlayers frame.board) == false)

def FrameReturn.terminalValue (frame : FrameReturn) : Int := 256 * Board.AwardedScore frame.board

def exceedsRaw (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config)
    (board : Board.Board) (cut : Int) (fuel : Nat) (canPass : Bool := true) : Bool :=
  if cut < -scoreMax then true
  else if scoreMax ≤ cut then false
  else
    match fuel with
    | 0 => decide (cut < 256 * Board.AwardedScore board)
    | nextFuel + 1 =>
      let moves := (List.finRange 64).filter fun sq => (Board.getMoves mc board).getLsbD sq.val
      match moves with
      | first :: rest =>
        let child (sq : Bitboard.Square) :=
          let flipped := Flip.Dispatch.resultValue fc sq board.player board.opponent
          Board.place board flipped sq
        !exceedsRaw mc fc (child first) (-cut - 1) nextFuel true ||
          rest.any (fun sq => !exceedsRaw mc fc (child sq) (-cut - 1) nextFuel true)
      | [] =>
        if canPass then
          let next := Board.switchPlayers board
          if Board.hasLegalMoves mc next then !exceedsRaw mc fc next (-cut - 1) (nextFuel + 1) false
          else decide (cut < 256 * Board.AwardedScore board)
        else decide (cut < 256 * Board.AwardedScore board)
termination_by (fuel, if canPass then 1 else 0)

def FrameReturn.checkThresholdRaw (frame : FrameReturn) (fc : Flip.Dispatch.X86Config) : Bool :=
  frame.checkWith (fun cut => exceedsRaw moveConfig fc frame.board cut frame.depth true)

def FrameReturn.followsPass (child parent : FrameReturn) (parentId : Nat) : Bool :=
  (Board.getMoves moveConfig parent.board == 0 &&
    Board.hasLegalMoves moveConfig child.board &&
    decide (child.board = Board.switchPlayers parent.board)) &&
  (child.parent == parentId && child.worker == parent.worker && child.depth == parent.depth) &&
    child.windowMatches parent.pv (-parent.beta) (-parent.alpha) (-parent.score)

def checkChild (known : Nat → Option FrameReturn) (fc : Flip.Dispatch.X86Config)
    (board : Board.Board) (depth worker : Nat) (requiredParent : Option Nat) (call : ChildReturn) : Bool :=
  match known call.frame with
  | none => false
  | some child =>
    (child.worker == worker && child.depth + 1 == depth && requiredParent.all (· == child.parent)) &&
      child.onMove fc board call.square && child.windowMatches call.pv call.alpha call.beta call.score

def Node.matchesReturn (node : Node) (parent : FrameReturn) : Bool :=
  parent.player == node.player && parent.opponent == node.opponent &&
    parent.windowMatches node.pv node.alpha node.beta node.result

def Node.checkLinks (node : Node) (known : Nat → Option FrameReturn) (fc : Flip.Dispatch.X86Config)
    (frameId : Nat) (parent : FrameReturn) : Bool :=
  node.matchesReturn parent &&
  node.calls.all (checkChild known fc node.board parent.depth parent.worker (some frameId)) &&
  node.handoff.all (fun split => split.reports.all fun row =>
    let check := checkChild known fc node.board parent.depth row.worker none
    check row.scoutCall && row.wide.all (fun wide => check (row.wideCall node.beta wide)))

structure Sample where
  worker : Nat
  scope : Nat
  slot : Nat
  first : Nat
  last : Nat
  player : Nat
  opponent : Nat
  metadata : Nat

instance : FromJson Sample where
  fromJson? v := do
    let obj ← expectObj v
    pure {
      worker := ← getField obj "worker"
      scope := ← getField obj "scope"
      slot := ← getField obj "slot"
      first := ← getField obj "first"
      last := ← getField obj "last"
      player := ← getField obj "player"
      opponent := ← getField obj "opponent"
      metadata := ← getField obj "metadata"
    }

namespace Sample

open TranspositionTable

def payload (sample : Sample) : SeqLock.Payload :=
  ⟨BitVec.ofNat 64 sample.player, BitVec.ofNat 64 sample.opponent, BitVec.ofNat 64 sample.metadata⟩

def board (sample : Sample) : Board.Board := ⟨sample.payload.player, sample.payload.opponent⟩

def sane (sample : Sample) : Bool :=
  decide (sample.first < 2 ^ 64 ∧ sample.last < 2 ^ 64 ∧
    sample.player < 2 ^ 64 ∧ sample.opponent < 2 ^ 64 ∧ sample.metadata < 2 ^ 64)

def accepted (lookup : Bool) (sample : Sample) : Bool :=
  let first := BitVec.ofNat 64 sample.first
  let last := BitVec.ofNat 64 sample.last
  if lookup then
    (LookupCut.readOccupied false first last sample.payload sample.payload.player sample.payload.opponent).isSome
  else (Snapshot.read false first last sample.payload).isSome &&
    (Packing.unpack false sample.payload.metadata).bound != 0

def samePublication (sample publication : Sample) : Bool :=
  sample.sane && publication.sane &&
    sample.scope == publication.scope && sample.slot == publication.slot && sample.first == publication.first &&
    sample.last == sample.first && publication.last == publication.first &&
    decide (publication.first % 2 = 0) &&
    (sample.player == publication.player && sample.opponent == publication.opponent &&
      sample.metadata == publication.metadata)

def ofNode (sample : Sample) (node : Node) : Bool :=
  let data := Packing.unpack false sample.payload.metadata
  let score := BitVec.ofInt 32 node.result
  let alpha := BitVec.ofInt 32 node.alpha
  let beta := BitVec.ofInt 32 node.beta
  let hint := (Packing.decodeBestMove data.bestMove).map Fin.val
  sample.player == node.player && sample.opponent == node.opponent &&
    data.score.toInt == node.result &&
    data.bound == Packing.boundByte (TranspositionTable.classify node.pv ⟨score⟩ ⟨alpha⟩ ⟨beta⟩) &&
    (match node.move with
      | some square => hint == some square
      | none => hint.all node.legalMask.getLsbD)

def cutMatches (sample : Sample) (frame : FrameReturn) : Bool :=
  let data := Packing.unpack false sample.payload.metadata
  (sample.worker == frame.worker && sample.player == frame.player && sample.opponent == frame.opponent) &&
    (frame.pv == false && data.score.toInt == frame.score) &&
    (match canCut false data ⟨BitVec.ofInt 32 frame.beta⟩ (BitVec.ofNat 32 frame.depth)
        exactSelectivity true with
      | .ok true => true
      | _ => false)

end Sample

def parseRow [FromJson α] (json : String) : Except String α := do
  fromJson? (← JsonLite.parse json)

def frameBefore {α : Type} (frames : Std.HashMap Nat (Nat × α)) (limit id : Nat) : Option α := do
  let (line, frame) ← frames[id]?
  if line < limit then some frame else none

structure RunState where
  nodes : Nat := 0
  splits : Nat := 0
  reads : Nat := 0
  external : Nat := 0
  stores : Nat := 0
  frames : Std.HashMap Nat (Nat × FrameReturn) := {}
  checked : Std.HashMap Nat (Nat × FrameReturn) := {}
  terminals : Nat := 0
  thresholds : Nat := 0
  provedPasses : Nat := 0
  provedNodes : Nat := 0
  passes : Nat := 0
  childLinks : Nat := 0
  checkedPublications : Std.HashSet (Nat × Nat × Nat) := {}
  provedPublications : Std.HashMap (Nat × Nat × Nat) Sample := {}
  lastProbe : Std.HashMap Nat (Nat × Sample) := {}
  cutoffs : Nat := 0
  provedCuts : Nat := 0
  provedSamples : Nat := 0
  readCounts : Std.HashMap (Nat × Nat × Nat) Nat := {}

def stepFrame (thresholdDepth : Nat) (st : RunState) (line : Nat) (idText fields : String) :
    Except String RunState :=
  match idText.toNat? with
  | none => .error s!"line {line}: malformed returned frame number"
  | some id =>
    match parseRow fields with
    | .error err => .error s!"line {line}, frame {id}: {err}"
    | .ok (frame : FrameReturn) =>
      if id == 0 || st.frames.contains id || !frame.sane then
        .error s!"line {line}, frame {id}: invalid or duplicate return"
      else if frame.isTerminal then
        if frame.checkBound frame.terminalValue then
          .ok { st with
            checked := st.checked.insert id (line, frame)
            terminals := st.terminals + 1
            frames := st.frames.insert id (line, frame) }
        else
          .error s!"line {line}, frame {id}: returned score disagrees with terminal value"
      else if frame.depth ≤ thresholdDepth then
        if frame.checkThresholdRaw flipConfig then
          .ok { st with
            checked := st.checked.insert id (line, frame)
            thresholds := st.thresholds + 1
            frames := st.frames.insert id (line, frame) }
        else
          .error s!"line {line}, frame {id}: returned score disagrees with the minimax threshold"
      else
        .ok { st with frames := st.frames.insert id (line, frame) }

def stepPass (st : RunState) (line : Nat) (frameText childText : String) :
    Except String RunState :=
  match frameText.toNat? with
  | none => .error s!"line {line}: malformed pass frame number"
  | some id =>
    match childText.toNat? with
    | none => .error s!"line {line}, frame {frameText}: malformed pass child number"
    | some childId =>
      match st.frames[id]? with
      | none => .error s!"line {line}, frame {id}: pass return missing"
      | some (atReturn, parent) =>
        match frameBefore st.frames atReturn childId with
        | none => .error s!"line {line}, frame {id}: earlier pass reply missing"
        | some child =>
          if !child.followsPass parent id then
            .error s!"line {line}, frame {id}: pass child disagrees with reflected position or result"
          else
            let next :=
              match frameBefore st.checked atReturn childId with
              | some known =>
                if known.followsPass parent id && parent.sane && !st.checked.contains id then
                  { st with
                    checked := st.checked.insert id (atReturn, parent)
                    provedPasses := st.provedPasses + 1 }
                else st
              | none => st
            .ok { next with passes := next.passes + 1 }

def stepCut (st : RunState) (line : Nat) (frameText fields : String) :
    Except String RunState :=
  match frameText.toNat? with
  | none => .error s!"line {line}: malformed cutoff frame number"
  | some id =>
    match st.frames[id]? with
    | none => .error s!"line {line}, frame {id}: cutoff return missing"
    | some (atReturn, parent) =>
      match parseRow fields with
      | .error err => .error s!"line {line}, frame {frameText}: {err}"
      | .ok (stamp : SlotStamp) =>
        match st.lastProbe[parent.worker]? with
        | none => .error s!"line {line}, frame {id}: cutoff probe missing"
        | some (observedAt, sample) =>
          if !(observedAt < atReturn && stamp.key == (sample.scope, sample.slot, sample.first)) then
            .error s!"line {line}, frame {id}: cutoff names a different probe"
          else if !sample.cutMatches parent then
            .error s!"line {line}, frame {id}: probe cannot justify the recorded cutoff"
          else
            let next :=
              match st.provedPublications[stamp.key]? with
              | some publication =>
                if sample.samePublication publication && parent.sane && !st.checked.contains id then
                  { st with
                    checked := st.checked.insert id (atReturn, parent)
                    provedCuts := st.provedCuts + 1 }
                else st
              | none => st
            .ok { next with cutoffs := next.cutoffs + 1 }

def stepNode (publications : Std.HashMap (Nat × Nat × Nat) (Nat × Sample))
    (st : RunState) (line : Nat) (frameText fields : String) :
    Except String RunState :=
  match frameText.toNat? with
  | none => .error s!"line {line}: malformed search frame number"
  | some id =>
    match st.frames[id]? with
    | none => .error s!"line {line}, frame {id}: search return missing"
    | some (atReturn, parent) =>
      match parseRow fields with
      | .error err => .error s!"line {line}, frame {frameText}: {err}"
      | .ok (node : Node) =>
        if !node.check then
          .error s!"line {line}, frame {frameText}: search replay disagrees"
        else if !node.checkLinks (frameBefore st.frames atReturn) flipConfig id parent then
          .error s!"line {line}, frame {frameText}: returned child or parent position disagrees with search"
        else
          let (checked, provedNodes) :=
            if node.coversAll && !st.checked.contains id &&
                node.checkLinks (frameBefore st.checked atReturn) flipConfig id parent then
              (st.checked.insert id (atReturn, parent), st.provedNodes + 1)
            else
              (st.checked, st.provedNodes)
          let childLinks := st.childLinks + node.calls.length + (node.handoff.map fun split =>
            split.reports.foldl (fun count row => count + 1 + row.wide.toList.length) 0).getD 0
          let splits := st.splits + if node.handoff.isSome then 1 else 0
          match node.publication with
          | none =>
            .ok { st with
              checked, provedNodes, nodes := st.nodes + 1, splits, childLinks }
          | some stamp =>
            match publications[stamp.key]? with
            | none => .error s!"line {line}, frame {frameText}: publication omitted"
            | some (prior, publication) =>
              if !(decide (prior < line)) || !publication.ofNode node then
                .error s!"line {line}, frame {frameText}: saved table entry disagrees with search"
              else
                let provedPublications :=
                  match checked[id]? with
                  | some (_, certified) =>
                    if node.matchesReturn certified then
                      st.provedPublications.insert stamp.key publication
                    else st.provedPublications
                  | none => st.provedPublications
                .ok { st with
                  checked, provedNodes, provedPublications
                  checkedPublications := st.checkedPublications.insert stamp.key
                  stores := st.stores + 1
                  nodes := st.nodes + 1, splits, childLinks }

def stepSample (publications : Std.HashMap (Nat × Nat × Nat) (Nat × Sample))
    (st : RunState) (line : Nat) (kind fields : String) :
    Except String RunState :=
  if kind == "publish" then .ok st
  else if kind != "probe" && kind != "lookup" then
    .error s!"line {line}: unknown record {kind}"
  else
    match parseRow fields with
    | .error err => .error s!"line {line}: {err}"
    | .ok (sample : Sample) =>
      let key := (sample.scope, sample.slot, sample.first)
      match publications[key]? with
      | none => .error s!"line {line}: missing slot publication {key}"
      | some (prior, publication) =>
        if !sample.samePublication publication || !sample.accepted (kind == "lookup") ||
            (sample.worker == publication.worker && !(decide (prior < line))) then
          .error s!"line {line}: slot sample disagrees with publication {key}"
        else
          let provedSamples :=
            match st.provedPublications[key]? with
            | some origin => if sample.samePublication origin then st.provedSamples + 1 else st.provedSamples
            | none => st.provedSamples
          .ok { st with
            reads := st.reads + 1
            external := st.external + if sample.worker != publication.worker then 1 else 0
            readCounts := st.readCounts.insert key (st.readCounts[key]?.getD 0 + 1)
            lastProbe := if kind == "probe" then st.lastProbe.insert sample.worker (line, sample) else st.lastProbe
            provedSamples }

def stepRow (thresholdDepth : Nat) (publications : Std.HashMap (Nat × Nat × Nat) (Nat × Sample))
    (st : RunState) (line : Nat) (inputRow : String) : Except String RunState :=
  match inputRow.trimAscii.toString.splitOn "|" with
  | ["frame", id, fields] => stepFrame thresholdDepth st line id fields
  | ["pass", frame, childId] => stepPass st line frame childId
  | ["cut", frame, fields] => stepCut st line frame fields
  | ["node", frame, fields] => stepNode publications st line frame fields
  | [kind, fields] => stepSample publications st line kind fields
  | _ => .error s!"line {line}: malformed search trace record"

def runFile (path : String) (thresholdDepth : Nat := 8) : IO UInt32 := do
  let input ← IO.FS.Handle.mk path .read
  let mut publications : Std.HashMap (Nat × Nat × Nat) (Nat × Sample) := {}
  let mut line := 0
  repeat
    let row ← input.getLine
    if row.isEmpty then break
    line := line + 1
    if let ["publish", fields] := row.trimAscii.toString.splitOn "|" then
      match parseRow fields with
      | .error err => IO.eprintln s!"line {line}: {err}"; return 1
      | .ok (sample : Sample) =>
        let key := (sample.scope, sample.slot, sample.first)
        if !sample.samePublication sample || publications.contains key then
          IO.eprintln s!"line {line}: invalid or duplicate slot publication {key}"
          return 1
        publications := publications.insert key (line, sample)
  let input ← IO.FS.Handle.mk path .read
  let mut st : RunState := {}
  line := 0
  repeat
    let inputRow ← input.getLine
    if inputRow.isEmpty then break
    line := line + 1
    match stepRow thresholdDepth publications st line inputRow with
    | .error err => IO.eprintln err; return 1
    | .ok next => st := next
  let linked := st.checkedPublications.fold (fun acc stamp => acc + st.readCounts[stamp]?.getD 0) 0
  if st.nodes == 0 || st.splits == 0 || st.reads == 0 || linked == 0 || st.childLinks == 0 then
    IO.eprintln "trace lacks a completed serial frame, split handoff or sampled search publication"
    return 1
  IO.println s!"{st.nodes} frames, {st.splits} split handoffs, {st.stores} search publications, {st.reads} occupied table samples ({st.external} cross-thread)"
  IO.println s!"{st.childLinks} move children and {st.passes} forced pass children matched among {st.frames.size} returned frames"
  IO.println s!"{st.checked.size} of {st.frames.size} returned bounds certified ({st.terminals} terminal, {st.thresholds} threshold up to {thresholdDepth} empties, {st.provedPasses} through pass, {st.provedNodes} through replay)"
  IO.println s!"{linked} table samples agree with a checked search publication"
  IO.println s!"{st.provedPublications.size} table publications and {st.provedSamples} reads have certified bounds"
  IO.println s!"{st.cutoffs} probe cutoffs matched, {st.provedCuts} new returns certified through the table"
  return 0

end NeuralReversi.PvsParallel.TraceCheck

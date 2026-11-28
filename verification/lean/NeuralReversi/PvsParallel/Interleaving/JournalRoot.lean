module
public import NeuralReversi.PvsParallel.Interleaving.CutoffPollReplay
public import NeuralReversi.PvsParallel.Interleaving.JournalEntry
public import NeuralReversi.PvsParallel.Interleaving.SolveEntry
public import NeuralReversi.PvsParallel.Interleaving.JournalHelpers
public import NeuralReversi.PvsParallel.Interleaving.JointReads

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.JournalRoot

open PvsBoardTree (Position GeneratedSquare)
open CallJournal (Choice)
open CallReturns (Row Scope Owner)
open TranspositionTable

/-- Registry rows include a boundary solver which can publish a score without a
    recursive PVS return. Such a row has no parent and cannot satisfy a child request. -/
def unreturned (e : JournalEntry.Execution mc fc) (state : CallReturns.State) (c : Choice) : Row where
  id := state.calls.size
  parent := none
  worker := c.worker
  site := 0
  entry := ⟨⟨c.board,c.depth,c.pv,c.cut,c.alpha.raw.toInt,c.beta.raw.toInt⟩,c.endgame,c.level.val,c.rootLine⟩
  chosen := c.stamp
  returned := CallJournal.returnOf e.state.closed c
  score := c.word.raw.toInt

def row (e : JournalEntry.Execution mc fc) (s : CallReturns.State) (c : Choice) : Row :=
  let missing := unreturned e s c
  let actual := s.calls.toList.zipIdx.findSome? fun (a,i) =>
    if a.worker = c.worker ∧ a.first = c.opening then a.row i else none
  match actual with
  | none => missing
  | some r => if r.entry = missing.entry ∧ r.chosen = c.stamp ∧ r.returned = missing.returned ∧ r.score = missing.score
      then r else missing

private theorem projected (e : JournalEntry.Execution mc fc) (s : CallReturns.State) (c : Choice) :
    (row e s c).frame = c.frame (CallJournal.returnOf e.state.closed) := by
  unfold row
  dsimp only
  split <;> try rfl
  split <;> try rfl
  rename_i matched
  obtain ⟨request,chosen,returned,score⟩ := matched
  simp only [Row.frame,request,chosen,returned,score]
  rfl

def eligible (s : CallJournal.State Op) (level : Nat) (f : CausalTable.Frame) :=
  if level = 5 then s.level f == Cutoffs.exactSelectivity
  else s.level f != Cutoffs.exactSelectivity && (s.choices.find? (fun d => d.stamp == f.chosen)).any (·.rootLine.isNone)

/-- The writer registry and the physical call stack use different orders. Join them
    by worker, opening, decision event and return operands. Ordinals come from the
    original array, including an entry that stopped without delivering a score. -/
def scope (e : JournalEntry.Execution mc fc) (state : CallReturns.State) (owner : CallReturns.Trial) (level : Nat) : Scope :=
  let represented := (e.state.choices.filter Choice.whole).map (row e state)
  let registered := represented.filter (fun r => eligible e.state level r.frame)
  ⟨registered,owner.id,owner.children.consumer,state.calls⟩

theorem registry (e : JournalEntry.Execution mc fc) (state : CallReturns.State) (owner : CallReturns.Trial) (level : Nat) :
    RootCausal.frames e.capture.history.weak e.capture.history.exact level = (scope e state owner level).rows.map Row.frame := by
  change RootCausal.frames e.state.weakChildren e.state.exactFrames level = _
  have mapFilter (rows : List Row) (test : CausalTable.Frame → Bool) :
      (rows.filter (fun r => test r.frame)).map Row.frame = (rows.map Row.frame).filter test := by
    induction rows with
    | nil => rfl
    | cons r rows ih => cases h : test r.frame <;> simp [h,ih]
  simp only [scope,mapFilter,List.map_map,Function.comp_def,projected]
  by_cases full : level = 5 <;>
    simp [RootCausal.frames,full,CallJournal.State.weakChildren,CallJournal.State.exactFrames,CallJournal.State.frames] <;>
    congr 1 <;> funext f <;> simp [eligible,full]

/-- A returning owner can be replayed against exactly the physical frame list
    supplied to the public result theorem, including postponement and retries. -/
theorem root_owner (e : JournalEntry.Execution mc fc) (state : CallReturns.State) (owner : CallReturns.Trial) (level : Nat) (b : Position)
    (schedule : Nat → Bool × Bool) (limit : Nat) (alpha beta : Int) (moves : List (GeneratedSquare mc b.val))
    (distinct : moves.Nodup) (out : Owner (GeneratedSquare mc b.val))
    (ran : (scope e state owner level).rootRun mc fc b schedule limit alpha beta moves = some out) :
    RootOwnerMir.fromMoves Subtype.val
      (((scope e state owner level).planSites (CallReturns.boardPlan mc fc b out.plan)).reply mc fc b
        e.capture.history.order e.capture.history.weak e.capture.history.exact level)
      schedule (Board.emptyCount b.val).toNat limit alpha beta moves = some out.result :=
  (scope e state owner level).rootRun_fromMoves mc fc b schedule limit alpha beta moves distinct (registry e state owner level) ran

open AspirationMir
open AspirationMir.SerialTable
open RootOwnerMir (Result)
open PublicationMir (SearchResult SliceSort)

/-- A raw scheduler run may only contribute to this root once the same
    invocation has supplied the recursive scout and any wide retry of each
    live packet. Stopped packets never provide a score to the parent. -/
inductive SplitExit (e : JournalEntry.Execution mc fc) (b : Position) (s : CallReturns.State)
    (owner : CallReturns.Trial) (level : Nat) (r : Result (GeneratedSquare mc b.val)) :
    (t : Schedule.Task) → (len : Nat) → {w : Nat} → BitVec w → Type where
  | native (tickets : NativeSplit.Tickets rc pv beta) (root cut : Bool) (depth : Nat)
      (raw : AlphaOrigin.Capture tickets root cut depth (fun _ _ _ _ _ => True))
      (exited : JournalHelpers.ownerExit e.events (scope e s owner level) b level raw = true)
      (receipts : JournalHelpers.checkPackets e.capture.history e.events (scope e s owner level).rows owner.id
        raw.bankIndex level (raw.mask.get raw.generation.mark).event tickets b r (registry e s owner level) rfl = true) :
      SplitExit e b s owner level r (ScoreOrigin.task tickets root cut depth) rc.length tickets.start

/-- Rebuild the scheduler from the same event tape as the calling PVS activation.
    The worker's recursive returns and the owner's load must pass their own checks. -/
def SplitExit.fromTape (e : JournalEntry.Execution mc fc) (b : Position) (s : CallReturns.State)
    (owner : CallReturns.Trial) (level : Nat) (r : Result (GeneratedSquare mc b.val))
    (tickets : NativeSplit.Tickets rc pv beta) (root cut : Bool) (depth : Nat)
    (data : AlphaOrigin.SetupOperands tickets) (stops : CutoffPollReplay.Context data e.events)
    (before : Option (Threading.AtomicWord.Event → Threading.AtomicWord.Event → Bool) := none) :
    Option (SplitExit e b s owner level r (ScoreOrigin.task tickets root cut depth) rc.length tickets.start) := do
  let raw ← CutoffPollReplay.fromTape data root cut depth e.events stops before
  if exit : JournalHelpers.ownerExit e.events (scope e s owner level) b level raw then
    if children : JournalHelpers.checkPackets e.capture.history e.events (scope e s owner level).rows owner.id
        raw.bankIndex level (raw.mask.get raw.generation.mark).event tickets b r (registry e s owner level) rfl then
      return .native tickets root cut depth raw exit children
  none

theorem SplitExit.fromTape_dispatches
    (ok : SplitExit.fromTape e b s owner level r tickets root cut depth data stops before = some out) :
    ∀ d ∈ tickets.dispatches, ∃ sd : ActivePointer.SplitDispatch data.memory.history
        (CutoffPollReplay.slots data) e.events (CutoffPollReplay.generation data) d.entry,
      ActivePointer.splitDispatch stops.chronology (CutoffPollReplay.slots data) e.events
        (CutoffPollReplay.generation data) d.entry = some sd ∧
      sd.origin.generation = CutoffPollReplay.generation data ∧
      Threading.AtomicWord.SequencedBefore sd.origin.drop d.entry := by
  obtain ⟨cap,replayed,_⟩ := Option.bind_eq_some_iff.mp ok
  intro d mem
  obtain ⟨sd,found,gen,_,after⟩ := CutoffPollReplay.fromTape_dispatches tickets data root cut depth e.events stops before cap replayed d mem
  exact ⟨sd,found,gen,after⟩

def SplitExit.trace {start : BitVec w} : SplitExit e b s owner level r t len start →
    ObservedJoin.Trace t (RootSchedule.indexed (e.capture.history.completed b level) r) len start
  | .native _ _ _ _ raw _ receipts => .physical (raw.withReports (JournalHelpers.packetReports receipts)).score.frame.physical

/-- Search and scheduler operands at a single native root entry. The return
    decoder supplies the owner callback and the position of the next attempt. -/
structure Operands (e : JournalEntry.Execution mc fc) (b : Position) (s : CallReturns.State)
    (owner : CallReturns.Trial) (lvl : Nat) (beta : Int) where
  memory : MoveList.Initialization.Memory Move.Move
  paired : Bool
  moves : SerialRecursion.RootInput
  schedule : Nat → Bool × Bool
  minimumSplitDepth : Nat
  width : Nat
  supported : 7 ≤ width
  split : (r : Result (GeneratedSquare mc b.val)) →
    Option (SplitExit e b s owner lvl r
      (RootSchedule.task r (Board.emptyCount b.val).toNat beta false) (RootTrial.labels r).length
      (BitVec.ofNat width r.remaining.idx))

abbrev Feed (e : JournalEntry.Execution mc fc) (b : Position) :=
  (s : CallReturns.State) → (owner : CallReturns.Trial) → (lvl : Nat) → (beta : Int) → Option (Operands e b s owner lvl beta)

def calls (e : JournalEntry.Execution mc fc) (b : Position) (state : CallReturns.State) (solve : Nat) (feed : Feed e b) :
    RootNative.CallsFor Nat e.capture.history b := fun idx lvl s ordinal => do
  let lo := s.regs.alpha.toInt
  let hi := s.regs.beta.toInt
  let trial ← state.rootAt solve ordinal b idx lvl lo hi
  let op ← feed state trial lvl hi
  let frame := scope e state trial lvl
  let roots := s.rows.map RootMoveStorage.Row.square
  let generation := MoveList.Generation.fill false op.paired fc b.val (Root.remainingMask mc b.val roots idx) op.memory
  let plan ← match ← SerialRootCalls.entry generation.len generation.wipeout with
    | some _ => some []
    | none => do
      let moves := SmallRoot.candidates mc fc b roots idx op.moves.key op.moves.sort
      let owner ← frame.rootRun mc fc b op.schedule op.minimumSplitDepth lo hi moves
      pure (CallReturns.boardPlan mc fc b owner.plan)
  let call : RootNative.Call Nat e.capture.history b lvl hi :=
    ⟨op.memory,op.paired,op.moves,frame.planSites plan,op.schedule,op.minimumSplitDepth,op.width,op.supported,
      fun r => (op.split r).map SplitExit.trace,trial.returned,ordinal+1⟩
  let answer ← RootProgram.run fc idx s call.input.code
  guard (answer.value = trial.score.raw.toInt)
  return call

def run (e : JournalEntry.Execution mc fc) (b : Position) (seed : JournalEntry.Seed e b) (eval : Position → Int)
    (feed : Feed e b) (multi : Bool) (clock : MultiPublic.Clock) (cap : Nat) (paired : Bool)
    (scratch : MoveList.Initialization.Memory Move.Move) (sort : SliceSort) (nodes : Nat) (fallback : Unit → SearchResult) := do
  let state ← CallReturns.captureTimed (e.events.map Joint.Event.frameSample)
  let base := SeedTable.estimate mc fc eval b seed.word
  let solve ← SolveEntry.direct e.events state seed.use.event ⟨b,base,NativeRootBirth.initial mc b,none,multi,1,cap⟩
  RootNative.direct e.capture.history b (calls e b state solve feed) multi clock cap base paired scratch 0 sort nodes fallback

def Contract : Prop := ∀ (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (e : JournalEntry.Execution mc fc)
    (b : Position) (seed : JournalEntry.Seed e b) (eval : Position → Int) (feed : Feed e b)
    (multi : Bool) (clock : MultiPublic.Clock) (cap : Nat) (paired : Bool)
    (scratch : MoveList.Initialization.Memory Move.Move) (sort : SliceSort) (nodes : Nat) (fallback : Unit → SearchResult)
    (report : SearchFallback.Report) (next : Nat), 1 ≤ cap →
    clock.ordered ((if multi then (NativeRootBirth.initial mc b).length else 1)-1) →
    run e b seed eval feed multi clock cap paired scratch sort nodes fallback = some (.bestMove report,next) →
    report.endgame = true → report.stamp.selectivity = 5 →
    report.best.score = 256 * EndgameTreeCache.discValue mc fc b.val ∧ (multi = true → MultiRows.ReportExact mc fc b report)

theorem from_returns : Contract := by
  intro mc fc e b seed eval feed multi clock cap paired scratch sort nodes fallback report next enabled time returned
  obtain ⟨state,_,rest⟩ := Option.bind_eq_some_iff.mp returned
  obtain ⟨solve,_,normal⟩ := Option.bind_eq_some_iff.mp rest
  exact JournalEntry.from_journal Nat mc fc e b seed eval (calls e b state solve feed) multi clock cap paired scratch 0
    sort nodes fallback report next enabled time normal

end NeuralReversi.PvsParallel.Interleaving.JournalRoot

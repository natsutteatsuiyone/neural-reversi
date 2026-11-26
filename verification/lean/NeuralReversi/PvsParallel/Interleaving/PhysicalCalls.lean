import NeuralReversi.PvsParallel.Interleaving.JournalCoverage
import NeuralReversi.PvsParallel.Interleaving.JournalRootStore
import NeuralReversi.PvsParallel.Interleaving.JournalExit

namespace NeuralReversi.PvsParallel.Interleaving.PhysicalCalls

open FrameTableReads.Operations.Execution (Method)
open TranspositionTable
open PvsBoardTree (Position)
open ReaderWindow (Args)

deriving instance DecidableEq for Method

variable {x : NativeTableWrites.Journal.Execution c endian rs} {es : Joint.Tape Op}

def raw (es : Joint.Tape Op) (endian : Bool) (finish consumer : Nat) :=
  ReaderWindow.check (Joint.worker es) (Joint.marker es) (Joint.opcode es) (fun _ => Joint.word es)
    (fun _ i => (Joint.word es i).setWidth 8) endian finish consumer

/-- Only the first and last sequence loads on executed TT attempts are constrained. -/
def NoWrap (m : Joint.Atomics x es) : Prop :=
  ∀ finish consumer opening args, (use : ReaderWindow.Use (Joint.bus es m) finish) →
    use.event = consumer → raw es endian finish consumer = some (opening,args) →
    let call := (Joint.stream es m).call opening finish use
    call.NoWrap (Joint.bus es m) (args.ops ((call.reader (Joint.bus es m)).supply x) endian)

/-- Replay the actual closing event and recover its method, address and registers. -/
def read (m : Joint.Atomics x es) (nowrap : NoWrap m) (finish consumer : Nat) (method : Method) :
    Option { r : JointReads.Call m consumer method // r.finish = finish } := do
  if connected : finish ≤ consumer ∧ Joint.worker es finish = Joint.worker es consumer then
    match observed : raw es endian finish consumer with
    | none => none
    | some (opening,⟨actual,allocation,clusters,key⟩) =>
      guard (key == ReaderWindow.engineKey endian actual)
      if methodEq : actual = method then
        return ⟨⟨allocation,clusters,key,opening,finish,connected.1,connected.2,methodEq ▸ observed,
          nowrap finish consumer opening _ ⟨consumer,connected.1,connected.2⟩ rfl (methodEq ▸ observed)⟩,rfl⟩
      else none
  else none

/-- For a scored frame, the native parent probe starts before any other TT
    call in that frame. Never skip a different invocation looking for a matching board. -/
def parentWindow (es : Joint.Tape Op) (state : CallJournal.State Op) (consumer : Nat) : Option (Nat × Nat) := do
  let owner ← state.choices.find? (fun f => f.stamp == consumer)
  let (_,start) ← es.zipIdx.find? fun (ev,_) => match ev.engineEvent with
    | .localCall w (.entry t _ _ ..) _ => w == owner.worker && t == owner.opening
    | _ => false
  let frames := (es.zipIdx.drop (start+1)).takeWhile (fun row => row.2 < consumer)
  let bounds := frames.filter fun (ev,_) => ev.worker == owner.worker && match ev with
    | .enter .. | .leave .. => true | _ => false
  let (.enter ..,opening) :: (.leave ..,finish) :: _ := bounds | none
  return (opening,finish)

def probes (m : Joint.Atomics x es) (nowrap : NoWrap m) (state : CallJournal.State Op) : JournalBodies.Probes m :=
  fun consumer board => do
    let (opening,finish) ← parentWindow es state consumer
    let ⟨r,_⟩ ← read m nowrap finish consumer (.probe (Cutoffs.key board))
    guard (r.opening = opening)
    return r

/-- Consume the first reader boundary after this child's lookup entry on its
    worker. An intervening entry invalidates the lookup instead of advancing to
    another invocation with the same board or a later successful read. -/
def lookupWindow (es : Joint.Tape Op) (opening consumer : Nat) : Option Nat := do
  let .enter w _ ← es[opening]? | none
  let (ev,finish) ← (es.zipIdx.drop (opening+1)).find? fun (ev,t) =>
    t < consumer && ev.worker == w && match ev with
      | .enter .. | .leave .. => true | _ => false
  let .leave .. := ev | none
  return finish

def children (m : Joint.Atomics x es) (nowrap : NoWrap m) : JournalScout.ChildReads m :=
  fun consumer opening b => do
    let finish ← lookupWindow es opening consumer
    let ⟨r,_⟩ ← read m nowrap finish consumer (.lookup b)
    guard (r.opening = opening)
    return r

private def sample (middle : Joint.Event Unit) : Joint.Tape Unit :=
  [.enter 0 ⟨.lookup ⟨⟨0,0⟩,rfl⟩,0,16,0⟩,middle,.leave 0 (.lookup none),.engine (.other 0)]

example : lookupWindow (sample (.leave 1 (.lookup none))) 0 3 = some 2 := by decide
example : lookupWindow (sample (.enter 0 ⟨.lookup ⟨⟨0,0⟩,rfl⟩,0,16,0⟩)) 0 3 = none := by decide
example : lookupWindow (sample (.engine (.other 0))) 0 2 = none := by decide

private def board : Position := ⟨⟨0x100000001,0x200000002⟩,rfl⟩
private def lookupArgs : Args := ⟨.lookup board,12,64,Board.Hash.hash false board.val⟩
example : lookupArgs.inTable false 12 64 = true := by decide
example : lookupArgs.inTable false 13 64 = false := by decide
example : lookupArgs.inTable false 12 32 = false := by decide
example : ({lookupArgs with key := lookupArgs.key + 1}).inTable false 12 64 = false := by decide
example : lookupArgs.inTable true 12 64 = false := by decide

def seed (e : JournalEntry.Execution mc fc) (nowrap : NoWrap e.atomics) (consumer : Nat) (b : Position) :
    Option (JournalEntry.Seed e b) := do
  match used : Joint.seedAt e.events consumer b with
  | none => none
  | some (finish,v) =>
    let ⟨r,closing⟩ ← read e.atomics nowrap finish consumer (.probe (Cutoffs.key b))
    return ⟨r.allocation,r.clusters,r.key,r.finish,⟨consumer,r.later,r.same⟩,r.opening,r.decoded,r.nowrap,v,closing ▸ used⟩

def fallback (e : JournalEntry.Execution mc fc) (nowrap : NoWrap e.atomics) (consumer : Nat) (b : Position) :
    Option ((data : Option EntryData) × JournalExit.CacheRead e b data) := do
  match used : Joint.receiverAt e.events consumer b with
  | some (finish,.lookup data) =>
    let ⟨r,closing⟩ ← read e.atomics nowrap finish consumer (.lookup b)
    return ⟨data,consumer,r,closing ▸ used⟩
  | _ => none

/-- Registers, observed table operations, and memory-order evidence for one
    capture. The interior frames carry no supplied score or bound proofs. -/
structure Raw (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) where
  calls : NativeTableWrites.Journal.Calls
  endian : Bool
  records : List NativeTableWrites.Journal.Record
  memory : NativeTableWrites.Journal.Execution calls endian records
  init : ∀ a, NativeTableWrites.Journal.Even (memory.memory a)
  resident : ∀ a, Entry.Resident mc fc endian ((memory.slot a (init a)).payloadAt 0)
  events : Joint.Tape (CallJournal.Operand calls endian (CallJournal.observed memory init))
  state : CallJournal.State (CallJournal.Operand calls endian (CallJournal.observed memory init))
  replay : CallJournal.decode mc fc calls endian (CallJournal.observed memory init)
    (events.map Joint.Event.engineEvent) = some state
  recorded : state.records = records
  order : ∀ a i j, (memory.memory a).hb i j → i < j
  atomics : Joint.Atomics memory events
  nowrap : NoWrap atomics
  pool : EndgameSearch.CacheHistory.Pool
  checked : Bool
  owners : JournalCoverage.Owners mc fc events state
  root : CausalTable.Frame → JournalRootStore.Options
  rootSplits : (f : CausalTable.Frame) → JournalRootStore.Splits events mc fc f.board

private def Raw.rootFrame (u : Raw mc fc) (f : CausalTable.Frame) :=
  JournalRootStore.check mc fc u.init (probes u.atomics u.nowrap u.state) u.state f (u.root f) (u.rootSplits f)

def Raw.covered (u : Raw mc fc) : Bool :=
  JournalBodies.poolHeader u.pool u.events &&
    JournalCoverage.covers mc fc u.init (probes u.atomics u.nowrap u.state) (children u.atomics u.nowrap)
      u.pool u.checked u.state u.owners u.rootFrame

def Raw.execution (u : Raw mc fc) (accepted : u.covered = true) : JournalEntry.Execution mc fc :=
  let probe := probes u.atomics u.nowrap u.state
  let frames := JournalCoverage.checkedFrames mc fc u.init probe (children u.atomics u.nowrap)
    u.pool u.checked u.state u.owners u.rootFrame (Bool.and_eq_true_iff.mp accepted).2
  JournalBodies.execution mc fc u.memory u.init u.resident u.events u.state u.replay u.recorded u.order
    u.atomics probe u.pool frames.owners.probCuts.etc.owners.passes.prefixes.bodies

open AspirationMir
open AspirationMir.SerialTable
open PublicationMir (SliceSort SearchResult)

def run (e : JournalEntry.Execution mc fc) (nowrap : NoWrap e.atomics) (b : Position) (receiver : Nat)
    (eval : Position → Int) (feed : JournalRoot.Feed e b) (multi : Bool) (clock : MultiPublic.Clock)
    (cap : Nat) (paired : Bool) (scratch : MoveList.Initialization.Memory Move.Move) (sort : SliceSort)
    (nodes : Nat) (noSaved : Unit → SearchResult) := do
  let first ← seed e nowrap receiver b
  JournalRoot.run e b first eval feed multi clock cap paired scratch sort nodes noSaved

def Contract : Prop := ∀ mc fc (e : JournalEntry.Execution mc fc) (nowrap : NoWrap e.atomics) (b : Position)
    (receiver : Nat) eval (feed : JournalRoot.Feed e b) multi clock cap paired scratch (sort : SliceSort)
    nodes noSaved report next, 1 ≤ cap →
    clock.ordered ((if multi then (NativeRootBirth.initial mc b).length else 1)-1) →
    run e nowrap b receiver eval feed multi clock cap paired scratch sort nodes noSaved = some (.bestMove report,next) →
    report.endgame = true → report.stamp.selectivity = 5 →
    report.best.score = 256 * EndgameTreeCache.discValue mc fc b.val ∧ (multi = true → MultiRows.ReportExact mc fc b report)

theorem direct : Contract := by
  intro mc fc e nowrap b receiver eval feed multi clock cap paired scratch sort nodes noSaved report next enabled time normal
  obtain ⟨first,_,ran⟩ := Option.bind_eq_some_iff.mp normal
  exact JournalRoot.from_returns mc fc e b first eval feed multi clock cap paired scratch sort nodes noSaved report next enabled time ran

/-- Each public path consumes the same checked score history. -/
def Raw.withFrames (u : Raw mc fc) (k : (ok : u.covered = true) → Option α) : Option α :=
  if ok : u.covered = true then k ok else none

private theorem Raw.accepted {u : Raw mc fc} {k : (ok : u.covered = true) → Option α}
    (normal : u.withFrames k = some result) : ∃ ok, k ok = some result := by
  unfold withFrames at normal
  split at normal
  · exact ⟨_,normal⟩
  · contradiction

/-- Run the public direct-entry replay only after every scored frame in the
    finite registries has passed the branch checks. -/
def Raw.run (u : Raw mc fc) (b : Position) (receiver : Nat) (eval : Position → Int)
    (feed : (ok : u.covered = true) → JournalRoot.Feed (u.execution ok) b)
    (multi : Bool) (clock : MultiPublic.Clock) (cap : Nat) (paired : Bool)
    (scratch : MoveList.Initialization.Memory Move.Move) (sort : SliceSort) (nodes : Nat) (noSaved : Unit → SearchResult) :=
  u.withFrames fun ok =>
    PhysicalCalls.run (u.execution ok) u.nowrap b receiver eval (feed ok) multi clock cap paired scratch sort nodes noSaved

def CheckedContract : Prop := ∀ mc fc (u : Raw mc fc) (b : Position) receiver eval feed multi clock cap paired scratch
    (sort : SliceSort) nodes noSaved report next, 1 ≤ cap →
    clock.ordered ((if multi then (NativeRootBirth.initial mc b).length else 1)-1) →
    u.run b receiver eval feed multi clock cap paired scratch sort nodes noSaved = some (.bestMove report,next) →
    report.endgame = true → report.stamp.selectivity = 5 →
    report.best.score = 256 * EndgameTreeCache.discValue mc fc b.val ∧ (multi = true → MultiRows.ReportExact mc fc b report)

theorem from_checked_frames : CheckedContract := by
  intro mc fc u b receiver eval feed multi clock cap paired scratch sort nodes noSaved report next enabled time normal
  obtain ⟨ok,ran⟩ := Raw.accepted normal
  exact direct mc fc (u.execution ok) u.nowrap b receiver eval (feed ok) multi clock cap paired scratch sort nodes noSaved
    report next enabled time ran

def unsaved (e : JournalEntry.Execution mc fc) (nowrap : NoWrap e.atomics) (b : Position)
    (consumer : Nat) (eval : Board.Board → Int) (sort : SliceSort) (nodes : Nat) := do
  let readBack ← fallback e nowrap consumer b
  PublicationMir.ExactScore.returned mc fc b eval sort none readBack.1 nodes .search

theorem cached (e : JournalEntry.Execution mc fc) (nowrap : NoWrap e.atomics) (b : Position)
    (normal : unsaved e nowrap b consumer eval sort nodes = some result) :
    PublicationMir.ExactScore.Ensures mc fc b result := by
  obtain ⟨⟨data,call⟩,_,published⟩ := Option.bind_eq_some_iff.mp normal
  exact JournalExit.fallback mc fc e b eval sort data nodes .search result published (fun _ => call)

/-- The timed continuation uses the actual solve entry after the midgame iteration,
    with the endgame frame proofs recovered by the same finite checks as direct entry. -/
theorem checked_handoff (u : Raw mc fc) (b : Position) (trial : MidgameIterate.Trial M)
    (works : MidgameIterate.Works mc b safe trial) (initial : MidgameIterate.State M)
    (valid : MidgameIterate.Valid mc b safe initial)
    (crossed : MidgameIterate.run trial midSort level depth midClock initial = some (.endgame phase))
    (feed : (ok : u.covered = true) → JournalRoot.Feed (u.execution ok) b) (multi : Bool) (clock : MultiPublic.Clock)
    (time : clock.ordered ((if multi then phase.state.rows.length else 1)-1))
    (normal : u.withFrames (fun ok => JournalExit.handoff (u.execution ok) b phase entry (feed ok)
      multi clock sort nodes noSaved) = some (.bestMove report,next))
    (tag : report.endgame = true) (full : report.stamp.selectivity = 5) :
    report.best.score = 256 * EndgameTreeCache.discValue mc fc b.val ∧ (multi = true → MultiRows.ReportExact mc fc b report) := by
  obtain ⟨ok,ran⟩ := Raw.accepted normal
  exact JournalExit.after_midgame (u.execution ok) b trial works initial valid crossed (feed ok) multi clock time ran tag full

/-- When there is no saved iteration, the actual closing lookup supplies the public
    score. The receipt uses the same memory and the already checked score registry. -/
theorem checked_cache_return (u : Raw mc fc) (b : Position)
    (normal : u.withFrames (fun ok => unsaved (u.execution ok) u.nowrap b consumer eval sort nodes) = some result) :
    PublicationMir.ExactScore.Ensures mc fc b result := by
  obtain ⟨ok,ran⟩ := Raw.accepted normal
  exact cached (u.execution ok) u.nowrap b ran

end NeuralReversi.PvsParallel.Interleaving.PhysicalCalls

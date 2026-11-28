module
public import NeuralReversi.PvsParallel.Interleaving.CutClosure
public import NeuralReversi.PvsParallel.Interleaving.MultiSessionCarry
public import NeuralReversi.PvsParallel.Interleaving.PhysicalReadback

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.ParallelOperational

open TranspositionTable
open NativeTableWrites.Journal (Calls Record Even)
open JournalAtomicGraph (Graph Rules HB)
open JournalBodies (Check)
open CausalTable (Frame)
open EndgameSearch.CacheHistory (Pool)

/-- Enumeration of the 10 operational branches of Rust's `pvs::search` / `solve_root`
    that produce a scored frame in `CallJournal.State`. Each branch matches its exact
    recognizer in `JournalCoverage` while preceding arms in Rust's dispatch order do
    not claim the frame (`.outside`). -/
inductive OperationalBranch
    (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    {c : Calls} {endian : Bool} {rs : List Record}
    {x : NativeTableWrites.Journal.Execution c endian rs}
    {Op : Type} {es : Joint.Tape Op} {m : Joint.Atomics x es}
    (init : ∀ a, Even (x.memory a)) (probes : JournalBodies.Probes m)
    (look : JournalScout.ChildReads m) (pool : Pool) (checked : Bool)
    (s : CallJournal.State Op) (op : JournalCoverage.Owners mc fc es s)
    (root : JournalCoverage.Final mc fc init s) (f : Frame) (prior : List Frame) : Prop where
  /-- Branch 1: non-PV depth ≤ 12 direct/shallow/deep leaf using worker's local endgame cache. -/
  | localLeaf {p} :
      JournalBodies.check mc fc init probes pool s f = .proved p →
      OperationalBranch mc fc init probes look pool checked s op root f prior
  /-- Branch 2: stability / terminal / wipeout / parent TT cutoff. -/
  | prefixCut {p} :
      JournalBodies.check mc fc init probes pool s f = .outside →
      JournalPrefix.check mc fc init probes checked s f = .proved p →
      OperationalBranch mc fc init probes look pool checked s op root f prior
  /-- Branch 3: forced pass when side-to-move has no legal moves and opponent does. -/
  | forcedPass {p} :
      JournalBodies.check mc fc init probes pool s f = .outside →
      JournalPrefix.check mc fc init probes checked s f = .outside →
      JournalPass.check mc fc es s prior (CausalTable.lookup x init f.chosen) f checked = .proved p →
      OperationalBranch mc fc init probes look pool checked s op root f prior
  /-- Branch 4: interior PV node completing serially. -/
  | pvSerial {p} :
      JournalBodies.check mc fc init probes pool s f = .outside →
      JournalPrefix.check mc fc init probes checked s f = .outside →
      JournalPass.check mc fc es s prior (CausalTable.lookup x init f.chosen) f checked = .outside →
      JournalPv.check mc fc init probes s f (op.pv f) = .proved p →
      OperationalBranch mc fc init probes look pool checked s op root f prior
  /-- Branch 5: interior Non-PV ETC cutoff. -/
  | scoutEtc {p} :
      JournalBodies.check mc fc init probes pool s f = .outside →
      JournalPrefix.check mc fc init probes checked s f = .outside →
      JournalPass.check mc fc es s prior (CausalTable.lookup x init f.chosen) f checked = .outside →
      JournalPv.check mc fc init probes s f (op.pv f) = .outside →
      JournalScout.check mc fc init probes look checked s f = .proved p →
      OperationalBranch mc fc init probes look pool checked s op root f prior
  /-- Branch 6: interior Non-PV ProbCut cutoff. -/
  | probCut {p} :
      JournalBodies.check mc fc init probes pool s f = .outside →
      JournalPrefix.check mc fc init probes checked s f = .outside →
      JournalPass.check mc fc es s prior (CausalTable.lookup x init f.chosen) f checked = .outside →
      JournalPv.check mc fc init probes s f (op.pv f) = .outside →
      JournalScout.check mc fc init probes look checked s f = .outside →
      JournalProbCut.check mc fc init probes look checked s f = .proved p →
      OperationalBranch mc fc init probes look pool checked s op root f prior
  /-- Branch 7: interior Non-PV node completing serially. -/
  | scoutSerial {p} :
      JournalBodies.check mc fc init probes pool s f = .outside →
      JournalPrefix.check mc fc init probes checked s f = .outside →
      JournalPass.check mc fc es s prior (CausalTable.lookup x init f.chosen) f checked = .outside →
      JournalPv.check mc fc init probes s f (op.pv f) = .outside →
      JournalScout.check mc fc init probes look checked s f = .outside →
      JournalProbCut.check mc fc init probes look checked s f = .outside →
      JournalScoutOwner.check mc fc init probes look checked s f (op.scout f) = .proved p →
      OperationalBranch mc fc init probes look pool checked s op root f prior
  /-- Branch 8: interior PV node handing off to a YBWC parallel split (Tasks 1–2). -/
  | pvSplit {p} :
      JournalBodies.check mc fc init probes pool s f = .outside →
      JournalPrefix.check mc fc init probes checked s f = .outside →
      JournalPass.check mc fc es s prior (CausalTable.lookup x init f.chosen) f checked = .outside →
      JournalPv.check mc fc init probes s f (op.pv f) = .outside →
      JournalScout.check mc fc init probes look checked s f = .outside →
      JournalProbCut.check mc fc init probes look checked s f = .outside →
      JournalScoutOwner.check mc fc init probes look checked s f (op.scout f) = .outside →
      JournalPvSplit.check mc fc init probes s f (op.pv f) op.pvSplit = .proved p →
      OperationalBranch mc fc init probes look pool checked s op root f prior
  /-- Branch 9: interior Non-PV node handing off to a YBWC parallel split (Tasks 1–2). -/
  | scoutSplit {p} :
      JournalBodies.check mc fc init probes pool s f = .outside →
      JournalPrefix.check mc fc init probes checked s f = .outside →
      JournalPass.check mc fc es s prior (CausalTable.lookup x init f.chosen) f checked = .outside →
      JournalPv.check mc fc init probes s f (op.pv f) = .outside →
      JournalScout.check mc fc init probes look checked s f = .outside →
      JournalProbCut.check mc fc init probes look checked s f = .outside →
      JournalScoutOwner.check mc fc init probes look checked s f (op.scout f) = .outside →
      JournalPvSplit.check mc fc init probes s f (op.pv f) op.pvSplit = .outside →
      JournalScoutOwner.checkWith mc fc init probes look checked s f (op.scout f) (some (op.scoutSplit f)) = .proved p →
      OperationalBranch mc fc init probes look pool checked s op root f prior
  /-- Branch 10: root line 0 serial or parallel split (Tasks 1–2). -/
  | rootLine0 {p} :
      JournalBodies.check mc fc init probes pool s f = .outside →
      JournalPrefix.check mc fc init probes checked s f = .outside →
      JournalPass.check mc fc es s prior (CausalTable.lookup x init f.chosen) f checked = .outside →
      JournalPv.check mc fc init probes s f (op.pv f) = .outside →
      JournalScout.check mc fc init probes look checked s f = .outside →
      JournalProbCut.check mc fc init probes look checked s f = .outside →
      JournalScoutOwner.check mc fc init probes look checked s f (op.scout f) = .outside →
      JournalPvSplit.check mc fc init probes s f (op.pv f) op.pvSplit = .outside →
      JournalScoutOwner.checkWith mc fc init probes look checked s f (op.scout f) (some (op.scoutSplit f)) = .outside →
      root f = .proved p →
      OperationalBranch mc fc init probes look pool checked s op root f prior

theorem OperationalBranch.recognized
    (branch : OperationalBranch mc fc init probes look pool checked s op root f prior) :
    JournalCoverage.recognized mc fc init probes look pool checked s op root f prior := by
  unfold JournalCoverage.recognized JournalPvSplit.pending JournalScoutOwner.pending
    JournalProbCut.pending JournalScout.pending JournalPv.pending JournalPass.pending
    JournalCoverage.last JournalCoverage.next
  cases branch with
  | localLeaf h1 => rw [h1]; trivial
  | prefixCut h1 h2 => rw [h1,h2]; trivial
  | forcedPass h1 h2 h3 => rw [h1,h2,h3]; trivial
  | pvSerial h1 h2 h3 h4 => rw [h1,h2,h3,h4]; trivial
  | scoutEtc h1 h2 h3 h4 h5 => rw [h1,h2,h3,h4,h5]; trivial
  | probCut h1 h2 h3 h4 h5 h6 => rw [h1,h2,h3,h4,h5,h6]; trivial
  | scoutSerial h1 h2 h3 h4 h5 h6 h7 => rw [h1,h2,h3,h4,h5,h6,h7]; trivial
  | pvSplit h1 h2 h3 h4 h5 h6 h7 h8 => rw [h1,h2,h3,h4,h5,h6,h7,h8]; trivial
  | scoutSplit h1 h2 h3 h4 h5 h6 h7 h8 h9 => rw [h1,h2,h3,h4,h5,h6,h7,h8,h9]; trivial
  | rootLine0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 => rw [h1,h2,h3,h4,h5,h6,h7,h8,h9,h10]; trivial

/-- An RF+HB-closed prefix cut of a full concurrent TT execution (Task 3),
    projected to `PhysicalReadback.Prefix` via `CutClosure.prefix_rules`. -/
structure CutPrefix (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) where
  calls : Calls
  endian : Bool
  fullTape : Joint.Tape Unit
  cut : Nat
  cutLe : cut ≤ fullTape.length
  preState : CallJournal.State Unit
  preRan : CallJournal.run mc fc calls endian ((fullTape.take cut).map Joint.Event.engineEvent) = some preState
  postRecords : List Record
  fullState : CallJournal.State Unit
  fullRan : CallJournal.run mc fc calls endian (fullTape.map Joint.Event.engineEvent) = some fullState
  fullRecEq : fullState.records = preState.records ++ postRecords
  postAfter : ∀ s, Record.writer s ∈ postRecords → cut ≤ s.event calls
  fullGraph : Graph
  fullRules : Rules fullGraph calls endian (preState.records ++ postRecords) fullTape
  hbClosed : CutClosure.HBClosed fullGraph fullTape cut
  rfClosed : CutClosure.RFClosed fullGraph calls (preState.records ++ postRecords) fullTape cut
  ordered : ∀ i j, HB (CutClosure.projectGraph fullGraph fullTape cut) (fullTape.take cut) i j → i < j

def CutPrefix.toPrefix (p : CutPrefix mc fc) : PhysicalReadback.Prefix mc fc where
  calls := p.calls
  endian := p.endian
  events := p.fullTape.take p.cut
  state := p.preState
  ran := p.preRan
  graph := CutClosure.projectGraph p.fullGraph p.fullTape p.cut
  rules := CutClosure.prefix_rules p.fullGraph p.calls p.endian p.preState.records p.postRecords
    p.fullTape p.cut p.cutLe (p.fullRecEq ▸ WriterSteps.run p.fullRan)
    (WriterSteps.run p.preRan) p.postAfter p.fullRules p.hbClosed p.rfClosed

/-- End-to-end operational parallel search run combining Tasks 1, 2, and 3:
    - Task 3 (Cut closure + multi-session carry): a concurrent TT execution cut at
      an `HBClosed ∧ RFClosed` prefix `cut`, starting from a validated resident TT
      image (`MultiSessionCarry.SlotOrigin`) and persistent worker-pool local cache
      history (`poolHeaderOk`).
    - Tasks 1 & 2 (ActivePointer + unified split tape): all worker dispatches and
      split lifecycles replayed on the same tape `events`.
    Crucially, `OperationalRun` does NOT assume `raw.covered = true` or child scores. -/
structure OperationalRun (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) where
  prefix_ : CutPrefix mc fc
  initEven : ∀ a, Even (prefix_.toPrefix.memory.memory a)
  slotOrigin : ∀ a, MultiSessionCarry.SlotOrigin mc fc prefix_.endian
    ((prefix_.toPrefix.memory.slot a (initEven a)).payloadAt 0)
  events : Joint.Tape (CallJournal.Operand prefix_.calls prefix_.endian
    (CallJournal.observed prefix_.toPrefix.memory initEven))
  erases : events.map PhysicalReadback.rawEvent = prefix_.toPrefix.events
  state : CallJournal.State (CallJournal.Operand prefix_.calls prefix_.endian
    (CallJournal.observed prefix_.toPrefix.memory initEven))
  decoded : CallJournal.decode mc fc prefix_.calls prefix_.endian
    (CallJournal.observed prefix_.toPrefix.memory initEven)
    (events.map Joint.Event.engineEvent) = some state
  nowrap : PhysicalCalls.NoWrap (prefix_.toPrefix.atomics events erases)
  pool : Pool
  checked : Bool
  poolHeaderOk : JournalBodies.poolHeader pool (prefix_.toPrefix.attach events) = true
  owners : JournalCoverage.Owners mc fc (prefix_.toPrefix.attach events) state
  rootOptions : Frame → JournalRootStore.Options
  rootSplits : (f : Frame) → JournalRootStore.Splits (prefix_.toPrefix.attach events) mc fc f.board

/-- Construct `PhysicalCalls.Raw` directly from an `OperationalRun` using Tasks 1–3. -/
def OperationalRun.toRaw (r : OperationalRun mc fc) : PhysicalCalls.Raw mc fc :=
  r.prefix_.toPrefix.toRaw
    r.initEven
    (fun a => (r.slotOrigin a).toResident)
    r.events
    r.erases
    r.state
    r.decoded
    r.prefix_.ordered
    r.nowrap
    r.pool
    r.checked
    r.owners
    r.rootOptions
    r.rootSplits

/-- Operational branch coverage for every scored frame in the three finite registries
    of `r.toRaw` — without assuming `r.toRaw.covered = true` or child score soundness. -/
structure BranchComplete (r : OperationalRun mc fc) : Prop where
  seed : ∀ f ∈ r.toRaw.state.frames,
    OperationalBranch mc fc r.toRaw.init
      (PhysicalCalls.probes r.toRaw.atomics r.toRaw.nowrap r.toRaw.state)
      (PhysicalCalls.children r.toRaw.atomics r.toRaw.nowrap)
      r.toRaw.pool r.toRaw.checked r.toRaw.state r.toRaw.owners
      (fun g => JournalRootStore.check mc fc r.toRaw.init
        (PhysicalCalls.probes r.toRaw.atomics r.toRaw.nowrap r.toRaw.state)
        r.toRaw.state g (r.toRaw.root g) (r.toRaw.rootSplits g))
      f (JournalPass.seedRows r.toRaw.state f)
  weak : ∀ f ∈ r.toRaw.state.weakChildren,
    OperationalBranch mc fc r.toRaw.init
      (PhysicalCalls.probes r.toRaw.atomics r.toRaw.nowrap r.toRaw.state)
      (PhysicalCalls.children r.toRaw.atomics r.toRaw.nowrap)
      r.toRaw.pool r.toRaw.checked r.toRaw.state r.toRaw.owners
      (fun g => JournalRootStore.check mc fc r.toRaw.init
        (PhysicalCalls.probes r.toRaw.atomics r.toRaw.nowrap r.toRaw.state)
        r.toRaw.state g (r.toRaw.root g) (r.toRaw.rootSplits g))
      f r.toRaw.state.weakChildren
  exact : ∀ f ∈ r.toRaw.state.exactFrames,
    OperationalBranch mc fc r.toRaw.init
      (PhysicalCalls.probes r.toRaw.atomics r.toRaw.nowrap r.toRaw.state)
      (PhysicalCalls.children r.toRaw.atomics r.toRaw.nowrap)
      r.toRaw.pool r.toRaw.checked r.toRaw.state r.toRaw.owners
      (fun g => JournalRootStore.check mc fc r.toRaw.init
        (PhysicalCalls.probes r.toRaw.atomics r.toRaw.nowrap r.toRaw.state)
        r.toRaw.state g (r.toRaw.root g) (r.toRaw.rootSplits g))
      f r.toRaw.state.exactFrames

/-- Operational Completeness Contract (Task 4):
    Any `OperationalRun` whose emitted frames each execute one of the 10 Rust
    `pvs::search` / `solve_root` branches AUTOMATICALLY satisfies `r.toRaw.covered = true`
    — no checker-acceptance premise or child-score premise is assumed. -/
def OperationalCompletenessContract : Prop :=
  ∀ mc fc (r : OperationalRun mc fc), BranchComplete r → r.toRaw.covered = true

theorem covers_proven : OperationalCompletenessContract := by
  intro mc fc r hBranch
  unfold PhysicalCalls.Raw.covered
  apply Bool.and_eq_true_iff.mpr
  refine ⟨r.poolHeaderOk,?_⟩
  exact JournalCoverage.covers_of_recognized mc fc r.toRaw.init
    (PhysicalCalls.probes r.toRaw.atomics r.toRaw.nowrap r.toRaw.state)
    (PhysicalCalls.children r.toRaw.atomics r.toRaw.nowrap)
    r.toRaw.pool r.toRaw.checked r.toRaw.state r.toRaw.owners _
    (fun f mem => (hBranch.seed f mem).recognized)
    (fun f mem => (hBranch.weak f mem).recognized)
    (fun f mem => (hBranch.exact f mem).recognized)

/-- Build the certified `JournalEntry.Execution` directly from an operational run
    and its branch discipline (deriving `covered = true` via `covers_proven`). -/
def OperationalRun.execution (r : OperationalRun mc fc) (hBranch : BranchComplete r) :
    JournalEntry.Execution mc fc :=
  r.toRaw.execution (covers_proven mc fc r hBranch)

end NeuralReversi.PvsParallel.Interleaving.ParallelOperational

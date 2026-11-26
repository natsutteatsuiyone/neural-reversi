import NeuralReversi.PvsParallel.Interleaving.RootNative
import NeuralReversi.PvsParallel.Interleaving.JournalResident
import NeuralReversi.PvsParallel.Interleaving.History
import NeuralReversi.AspirationMir.SerialTable.MultiSession

namespace NeuralReversi.PvsParallel.Interleaving.Entry

open PvsBoardTree (Position)
open AspirationMir
open AspirationMir.SerialTable
open TranspositionTable
open PublicationMir (SearchResult SliceSort)

/-- Starting slots come from allocation, a serial request, or the final four
    words of a previous parallel search with all its slot writers released. -/
inductive Resident (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) : Bool → SeqLock.Payload → Prop where
  | zero : Resident mc fc endian ⟨0,0,0⟩
  | warm {l : PvsParallel.SerialTable.Layout} (m : PvsParallel.SerialTable.Memory l) (i : l.Index)
      (ready : Repeated.ReadyMemory pool mc fc m) :
      Resident mc fc l.bigEndian (PvsParallel.SerialTable.payload (m.tt.image.get i))
  | parallel (previous : JournalResident.Session mc fc endian)
      (origins : ∀ a, Resident mc fc endian (previous.initial a))
      (a : FrameTableReads.Address) (drained : previous.drained) :
      Resident mc fc endian (previous.finalImage a)

theorem Resident.valid (origin : Resident mc fc endian p) :
    Table.Safe mc fc endian p ∧ FiniteTable.Words endian p ∧ SeedTable.Words endian p := by
  induction origin with
  | zero => exact ⟨History.zero_safe (a := ⟨0,0⟩),FiniteTable.zero,SeedTable.zero⟩
  | warm m i ready =>
    exact ⟨Table.warm m.tt ready.1.1.1 i,FiniteTable.warm m ready.1 i,SeedTable.warm m.tt m.supported ready.2 i⟩
  | parallel previous _ a drained ih => exact previous.final_valid ih a (previous.drained_released drained a)

/-- A finite chain needs provenance only for its first table. Each handoff
    supplies the physical equality at the same address for the next session. -/
theorem Resident.chain {first last : JournalResident.Session mc fc endian}
    (steps : JournalResident.Chain first last) (seed : ∀ a, Resident mc fc endian (first.initial a)) :
    ∀ a, Resident mc fc endian (last.initial a) := by
  induction steps with
  | start => exact seed
  | @next previous session _ link origins =>
    intro a
    rw [JournalResident.Session.initial,link.words session.init a]
    exact .parallel previous origins a link.drained

/-- The three projections share the writer journal, slot addresses and causal order.
    The numeric projection also accounts for earlier low-selectivity producers. -/
structure Capture (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) where
  writers : NativeTableWrites.Journal.Calls
  endian : Bool
  records : List NativeTableWrites.Journal.Record
  memory : NativeTableWrites.Journal.Execution writers endian records
  init : ∀ a, NativeTableWrites.Journal.Even (memory.memory a)
  slots : ∀ a, Resident mc fc endian ((memory.slot a (init a)).payloadAt 0)
  order : Nat → Nat → Bool
  frames : List CausalTable.Frame
  level : CausalTable.Frame → Selectivity
  levels : SeedHistory.Prefix mc fc memory init frames level (fun i j => order i j = true)
  weak : List CausalTable.Frame
  exactFrames : List CausalTable.Frame
  weakRun : FiniteTable.Prefix memory init weak (fun i j => order i j = true) (WeakFrame.Frame mc fc)
  exactRun : CausalTable.Prefix memory init exactFrames (fun i j => order i j = true) (Nested.Body mc fc)

def Capture.session (c : Capture mc fc) : JournalResident.Session mc fc c.endian :=
  ⟨c.writers,c.records,c.memory,c.init,(fun i j => c.order i j = true),c.frames,c.level,c.levels,c.exactFrames,c.exactRun⟩

/-- Every physical allocation/slot address is preserved across the idle boundary.
    The resulting residents may seed another parallel session in turn. -/
theorem Capture.carried (c : Capture mc fc) {next : NativeTableWrites.Journal.Execution calls c.endian rs}
    (joined : JournalResident.Handoff c.session next) (init : ∀ a, NativeTableWrites.Journal.Even (next.memory a))
    (a : FrameTableReads.Address) : Resident mc fc c.endian ((next.slot a (init a)).payloadAt 0) := by
  rw [joined.words init a]
  exact .parallel c.session c.slots a joined.drained

def Capture.history (c : Capture mc fc) : RootNative.History mc fc :=
  ⟨c.writers,c.endian,c.records,c.memory,c.init,c.levels.finiteStores,(fun a => (c.slots a).valid.1),
    (fun a => (c.slots a).valid.2.1),c.weak,c.exactFrames,c.order,c.weakRun,Nested.toPrefix c.exactRun⟩

structure Probe (h : RootNative.History mc fc) (b : Position) where
  reader : CausalReads.Reader
  owner : Nat
  allocation : Nat
  clusters : SeqLock.Word
  key : SeqLock.Word
  first : Nat
  issued : let out := FrameTableReads.Operations.probe (reader.supply h.native) allocation clusters key h.endian (Cutoffs.key b) first
    ∀ round ∈ out.rounds, CausalReads.Issued h.native reader owner round.front (FrameTableReads.snapCode h.endian) ∧
      ∀ back ∈ round.back, CausalReads.Issued h.native reader owner back (FrameTableReads.snapCode h.endian)

def Probe.value {h : RootNative.History mc fc} {b : Position} (p : Probe h b) (eval : Position → Int) :=
  let out := FrameTableReads.Operations.probe (p.reader.supply h.native) p.allocation p.clusters p.key h.endian (Cutoffs.key b) p.first
  SeedTable.estimate mc fc eval b out.result

/-- The initial estimate comes from the same physical TT as the worker calls.
    A previous selective or interrupted producer may already have published its word. -/
theorem direct {h : RootNative.History mc fc} {b : Position} (p : Probe h b) (eval : Position → Int)
    (prior : SeedHistory.Prefix mc fc h.native h.init frames level (fun i j => h.order i j = true))
    (seed : ∀ a, SeedTable.Words h.endian ((h.native.slot a (h.init a)).payloadAt 0))
    (calls : RootNative.CallsFor M h b) (multi : Bool) (clock : MultiPublic.Clock) (enabled : 1 ≤ cap)
    (time : clock.ordered ((if multi then (NativeRootBirth.initial mc b).length else 1)-1))
    (returned : RootNative.direct h b calls multi clock cap (p.value eval) paired scratch m sort nodes fallback =
      some (.bestMove report,next))
    (tag : report.endgame = true) (full : report.stamp.selectivity = 5) :
    report.best.score = 256 * EndgameTreeCache.discValue mc fc b.val ∧
    (multi = true → MultiRows.ReportExact mc fc b report) := by
  have range := SeedHistory.estimate_bounded prior h.finite seed p.reader p.owner p.allocation p.clusters p.key p.first eval b p.issued
  exact RootNative.direct_from_flags h b calls multi clock range enabled time returned tag full

/-- A direct public return from primitive worker, frame and table events; the initial
    root rows and aspiration estimate are computed by the respective native paths. -/
def Contract : Prop := ∀ (M : Type) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (c : Capture mc fc) (b : Position) (probe : Probe c.history b) (eval : Position → Int)
    (calls : RootNative.CallsFor M c.history b) (multi : Bool) (clock : MultiPublic.Clock)
    (cap : Nat) (paired : Bool) (scratch : MoveList.Initialization.Memory Move.Move) (m : M)
    (sort : SliceSort) (nodes : Nat) (fallback : Unit → SearchResult) (report : SearchFallback.Report) (next : M),
    1 ≤ cap → clock.ordered ((if multi then (NativeRootBirth.initial mc b).length else 1)-1) →
    RootNative.direct c.history b calls multi clock cap (probe.value eval) paired scratch m sort nodes fallback =
      some (.bestMove report,next) → report.endgame = true → report.stamp.selectivity = 5 →
    report.best.score = 256 * EndgameTreeCache.discValue mc fc b.val ∧
      (multi = true → MultiRows.ReportExact mc fc b report)

theorem from_events : Contract := by
  intro M mc fc c b probe eval calls multi clock cap paired scratch m sort nodes fallback report next enabled time returned
  exact direct probe eval c.levels (fun a => (c.slots a).valid.2.2) calls multi clock enabled time returned

/-- A timed handoff obtains the root rows, seed score and saved midgame stamp from
    the completed iteration. Subsequent endgame attempts can split at any level. -/
theorem handoff {h : RootNative.History mc fc} {b : Position} (calls : RootNative.CallsFor M h b)
    (phase : MidgameIterate.Completed N) (ready : MidgameIterate.Complete mc b good level depth phase)
    (multi : Bool) (clock : MultiPublic.Clock)
    (time : clock.ordered ((if multi then phase.state.rows.length else 1)-1))
    (returned : RootNative.solve h b calls multi clock 5 2 (some phase.saved) (BitVec.ofInt 32 phase.base.score)
      phase.state.rows m sort nodes fallback = some (.bestMove report,next))
    (tag : report.endgame = true) (full : report.stamp.selectivity = 5) :
    report.best.score = 256 * EndgameTreeCache.discValue mc fc b.val ∧
      (multi = true → MultiRows.ReportExact mc fc b report) :=
  RootNative.correct _ _ _ h b calls multi clock _ _ _ _ _ _ _ _ _ _ _ ready.valid.1
    (List.length_pos_iff.mp ready.valid.2.1) (MultiSession.base_range _ ready.bounds)
    (.handoff phase.saved ready.midgame) (by decide) time returned tag full

end NeuralReversi.PvsParallel.Interleaving.Entry

import NeuralReversi.PvsParallel.Interleaving.Joint

namespace NeuralReversi.PvsParallel.Interleaving.JournalEntry

open PvsBoardTree (Position)
open AspirationMir
open AspirationMir.SerialTable
open TranspositionTable
open PublicationMir (SearchResult SliceSort)
open NativeTableWrites.Journal (Even Record Calls)

/-- The single writer projection supplies all three score histories. Its event
    numbers are the ones used by the reads-from and program-order relations. -/
structure Execution (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) where
  calls : Calls
  endian : Bool
  records : List Record
  memory : NativeTableWrites.Journal.Execution calls endian records
  init : ∀ a, Even (memory.memory a)
  resident : ∀ a, Entry.Resident mc fc endian ((memory.slot a (init a)).payloadAt 0)
  events : Joint.Tape (CallJournal.Operand calls endian (CallJournal.observed memory init))
  state : CallJournal.State (CallJournal.Operand calls endian (CallJournal.observed memory init))
  replay : CallJournal.decode mc fc calls endian (CallJournal.observed memory init)
    (events.map Joint.Event.engineEvent) = some state
  recorded : state.records = records
  order : ∀ a i j, (memory.memory a).hb i j → i < j
  seeds : CallJournal.SeedBodies mc fc state
  weak : CallJournal.WeakBodies mc fc state
  exactFrames : CallJournal.ExactBodies mc fc state
  atomics : Joint.Atomics memory events

def Execution.trace (e : Execution mc fc) : CallJournal.Trace mc fc e.calls e.endian (CallJournal.observed e.memory e.init) :=
  ⟨e.events.map Joint.Event.engineEvent,e.state,e.replay⟩

abbrev Execution.bus (e : Execution mc fc) := Joint.bus e.events e.atomics
abbrev Execution.readers (e : Execution mc fc) := Joint.stream e.events e.atomics

def Execution.capture (e : Execution mc fc) : Entry.Capture mc fc :=
  e.trace.capture e.recorded e.resident e.order e.seeds e.weak e.exactFrames

/-- The board, return slot, and metadata are decoded from the completed probe
    immediately used by the initial aspiration estimate. -/
structure Seed (e : Execution mc fc) (b : Position) where
  allocation : Nat
  clusters : SeqLock.Word
  key : SeqLock.Word
  finish : Nat
  use : ReaderWindow.Use e.bus finish
  opening : Nat
  checked : ReaderWindow.check e.bus.worker e.readers.marker e.bus.instruction (fun _ => Joint.word e.events)
    e.bus.generation e.endian finish use.event = some (opening,⟨.probe (Cutoffs.key b),allocation,clusters,key⟩)
  nowrap : let call := e.readers.call opening finish use
    call.NoWrap e.bus ((ReaderWindow.Args.mk (.probe (Cutoffs.key b)) allocation clusters key).ops
      ((call.reader e.bus).supply e.memory) e.endian)
  word : ProbeRetry.Result
  consumed : Joint.seedAt e.events use.event b = some (finish,word)

def Contract : Prop := ∀ (M : Type) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (e : Execution mc fc) (b : Position) (seed : Seed e b) (eval : Position → Int)
    (calls : RootNative.CallsFor M e.capture.history b) (multi : Bool) (clock : MultiPublic.Clock)
    (cap : Nat) (paired : Bool) (scratch : MoveList.Initialization.Memory Move.Move) (m : M)
    (sort : SliceSort) (nodes : Nat) (fallback : Unit → SearchResult) (report : SearchFallback.Report) (next : M),
    1 ≤ cap → clock.ordered ((if multi then (NativeRootBirth.initial mc b).length else 1)-1) →
    RootNative.direct e.capture.history b calls multi clock cap (SeedTable.estimate mc fc eval b seed.word)
      paired scratch m sort nodes fallback = some (.bestMove report,next) →
    report.endgame = true → report.stamp.selectivity = 5 →
    report.best.score = 256 * EndgameTreeCache.discValue mc fc b.val ∧ (multi = true → MultiRows.ReportExact mc fc b report)

theorem from_journal : Contract := by
  intro M mc fc e b seed eval calls multi clock cap paired scratch m sort nodes fallback report next enabled time returned
  have checked := Joint.checked e.events e.atomics ⟨.probe (Cutoffs.key b),seed.allocation,seed.clusters,seed.key⟩
    seed.finish seed.use.event seed.opening seed.checked
  let probe := e.readers.finished seed.use checked
  let p : Entry.Probe e.capture.history b :=
    ReaderWindow.rootProbe (h := e.capture.history) e.readers seed.use b probe seed.nowrap
  have raw := ReaderWindow.probe_answer e.readers seed.use b probe (Joint.seed_returned e.events seed.consumed)
  have same : p.value eval = SeedTable.estimate mc fc eval b seed.word :=
    congrArg (SeedTable.estimate mc fc eval b) raw
  rw [←same] at returned
  exact Entry.from_events M mc fc e.capture b p eval calls multi clock _ _ _ _ _ _ _ _ _ enabled time returned

end NeuralReversi.PvsParallel.Interleaving.JournalEntry

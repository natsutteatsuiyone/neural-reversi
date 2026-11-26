import NeuralReversi.PvsParallel.Interleaving.RootCausal

namespace NeuralReversi.PvsParallel.Interleaving.RootNative

open PvsBoardTree (Position GeneratedSquare)
open AspirationMir
open AspirationMir.SerialTable
open NativeTableWrites.Journal
open RootMoveStorage (Row)
open PvsMoveLoop (scoreMax)
open PublicationMir (Iteration SliceSort SearchResult)
open RootCausal (Sites Body)
open RootProgram (Operands)
open RootOwnerMir (Result)
open Bitboard.Moves (X86Config)

structure History (mc : X86Config) (fc : Root.FlipConfig) where
  calls : Calls
  endian : Bool
  records : List Record
  native : NativeTableWrites.Journal.Execution calls endian records
  init : ∀ a, Even (native.memory a)
  stores : FiniteTable.Stores native init
  safe : ∀ a, Table.Safe mc fc endian ((native.slot a (init a)).payloadAt 0)
  finite : ∀ a, FiniteTable.Words endian ((native.slot a (init a)).payloadAt 0)
  weak : List CausalTable.Frame
  exact : List CausalTable.Frame
  order : Nat → Nat → Bool
  weakRun : FiniteTable.Prefix native init weak (fun i j => order i j = true) (WeakFrame.Frame mc fc)
  exactRun : CausalTable.Prefix native init exact (fun i j => order i j = true) (Body mc fc)

def History.completed (h : History mc fc) (b : Position) : RootLadder.Children mc b :=
  fun level move pv a z score => ∃ site : Sites, site.reply mc fc b h.order h.weak h.exact level move pv a z = some score

private theorem History.laws (h : History mc fc) (b : Position) : RootLadder.Laws mc fc b (h.completed b) := by
  have typed := RootCausal.laws (b := b) h.stores h.safe h.finite h.weakRun h.exactRun
  exact ⟨fun l notFull m pv a z v ⟨site,observed⟩ => typed.finite l notFull _ _ _ _ _ (site.hit m observed),
    fun m pv a z v window narrow ⟨site,observed⟩ => typed.exact _ _ _ _ _ window narrow (site.hit m observed)⟩

structure Call (M : Type) (h : History mc fc) (b : Position) (level : Nat) (beta : Int) where
  memory : MoveList.Initialization.Memory Move.Move
  paired : Bool
  moves : SerialRecursion.RootInput
  owner : Sites
  scheduling : Nat → Bool × Bool
  minimumSplitDepth : Nat
  width : Nat
  supported : 7 ≤ width
  split : (r : Result (GeneratedSquare mc b.val)) →
    Option (ObservedJoin.Trace (RootSchedule.task r (Board.emptyCount b.val).toNat beta false) (RootSchedule.indexed (h.completed b level) r)
      (RootTrial.labels r).length (BitVec.ofNat width r.remaining.idx))
  closing : Nat
  after : M

def Call.input {h : History mc fc} (call : Call M h b level beta) : RootProgram.Input M mc b (h.completed b level) beta :=
  ⟨⟨call.memory,call.paired,call.moves.key,call.moves.sort,call.moves.pvLine,
    call.owner.reply mc _ b h.order h.weak h.exact level,fun _ _ _ _ _ eq => ⟨call.owner,eq⟩,
    call.scheduling,call.minimumSplitDepth,call.width,call.supported,call.split⟩,call.closing,call.after⟩

abbrev CallsFor (M : Type) (h : History mc fc) (b : Position) :=
  (idx level : Nat) → (s : SerialStack.State) → M → Option (Call M h b level s.regs.beta.toInt)

def source (h : History mc fc) (b : Position) (calls : CallsFor M h b) :=
  RootProgram.source fc (fun idx level s m => ((calls idx level s m).map Call.input))

def solve (h : History mc fc) (b : Position) (calls : CallsFor M h b) (multi : Bool) (clock : MultiPublic.Clock)
    (cap minimum : Nat) (prior : Option Iteration) (base : Word) (rows : List Row) (m : M)
    (sort : SliceSort) (nodes : Nat) (fallback : Unit → SearchResult) :=
  Publication.solve (source h b calls) multi clock cap minimum prior base rows m sort nodes fallback

def Contract : Prop := ∀ (M : Type) (mc : X86Config) (fc : Root.FlipConfig) (h : History mc fc) (b : Position)
    (calls : CallsFor M h b) (multi : Bool) (clock : MultiPublic.Clock) (cap minimum : Nat) (prior : Option Iteration)
    (base : Word) (rows : List Row) (m : M) (sort : SliceSort) (nodes : Nat) (fallback : Unit → SearchResult)
    (report : SearchFallback.Report) (next : M),
    Root.Covers mc b (rows.map Row.square) → rows ≠ [] → (-scoreMax ≤ base.toInt ∧ base.toInt ≤ scoreMax) →
    MultiClock.Origin minimum prior → 1 ≤ cap → clock.ordered ((if multi then rows.length else 1)-1) →
    solve h b calls multi clock cap minimum prior base rows m sort nodes fallback = some (.bestMove report,next) →
    report.endgame = true → report.stamp.selectivity = 5 →
    report.best.score = 256 * EndgameTreeCache.discValue mc fc b.val ∧ (multi = true → MultiRows.ReportExact mc fc b report)

theorem correct : Contract := by
  intro M mc fc h b calls multi clock cap minimum prior base rows m sort nodes fallback report next cover pos range origin enabled time
  exact Publication.from_flags _ (h.laws b) multi clock cover pos range origin enabled time

def direct (h : History mc fc) (b : Position) (calls : CallsFor M h b) (multi : Bool) (clock : MultiPublic.Clock)
    (cap : Nat) (base : Word) (paired : Bool) (scratch : MoveList.Initialization.Memory Move.Move)
    (m : M) (sort : SliceSort) (nodes : Nat) (fallback : Unit → SearchResult) := do
  let rows ← NativeRootBirth.started mc fc b paired scratch
  solve h b calls multi clock cap 1 none base rows m sort nodes fallback

theorem direct_from_flags (h : History mc fc) (b : Position) (calls : CallsFor M h b) (multi : Bool) (clock : MultiPublic.Clock)
    (range : -scoreMax ≤ base.toInt ∧ base.toInt ≤ scoreMax) (enabled : 1 ≤ cap)
    (time : clock.ordered ((if multi then (NativeRootBirth.initial mc b).length else 1)-1))
    (returned : direct h b calls multi clock cap base paired scratch m sort nodes fallback = some (.bestMove report,next))
    (tag : report.endgame = true) (full : report.stamp.selectivity = 5) :
    report.best.score = 256 * EndgameTreeCache.discValue mc fc b.val ∧ (multi = true → MultiRows.ReportExact mc fc b report) := by
  obtain ⟨rows,birth,run⟩ := Option.bind_eq_some_iff.mp returned
  have ⟨names,pos⟩ := NativeRootBirth.covers birth
  have eq := ((NativeRootBirth.started_iff ..).mp birth).2.2
  exact correct M mc fc h b calls multi clock _ _ _ _ _ _ _ _ _ _ _ names pos range .direct enabled (eq ▸ time) run tag full

end NeuralReversi.PvsParallel.Interleaving.RootNative

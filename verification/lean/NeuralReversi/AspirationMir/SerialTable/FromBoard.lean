import NeuralReversi.AspirationMir.SerialTable
import NeuralReversi.PvsParallel.SerialTable.Trial

namespace NeuralReversi.AspirationMir.SerialTable

open EndgameSearch.CacheHistory (Pool)
open PvsParallel
open PvsParallel.SerialTable (Layout Memory)
open PvsBoardTree (Position)
open RootMoveStorage (Row)
open Bitboard.Moves (X86Config)
open PvsParallel.Cutoffs (FlipConfig)

structure MoveInputs (mc : X86Config) where
  scratch : Nat → SerialExecution.Frame → MoveList.Initialization.Memory Move.Move
  heuristic : Nat → SerialExecution.Frame → Bitboard.Square → EndgameOrderMir.Heuristic
  tie : Nat → (f : SerialExecution.Frame) → PvsBoardTree.GeneratedSquare mc f.request.board.val → Fin 64
  rootSort : RootFrameMir.LibraryOrder
  pvLine : Bitboard.Square → List Bitboard.Square

def MoveInputs.base (o : MoveInputs mc) : SerialExecution.Inputs mc fc := fun i f =>
  ⟨o.scratch i f,[],[],[],[],o.heuristic i f,o.tie i f⟩

def MoveInputs.root (o : MoveInputs mc) : SerialRecursion.RootInput :=
  ⟨fun _ => 0,o.rootSort,o.pvLine⟩

/-- The root rebuilds its keys from the current physical hint and the move heuristic. -/
def search (p : Pool) (c : SerialScout.Config) (b : Position) (orders : SerialStack.State → MoveInputs mc)
    (fc : FlipConfig) : SerialCache.Trial (Memory l) :=
  fun s before => PvsParallel.SerialTable.Trial.trial p c b ((orders s).base (fc := fc))
    (orders s).root s before

def recorded (trial : SerialCache.Trial M) (r : SerialCache.Receipt M) : Prop :=
  trial r.state r.before = some (r.closing,r.answer,r.after)

theorem works (p : Pool) (c : PvsParallel.SerialScout.Config) (b : Position)
    (orders : SerialStack.State → MoveInputs mc) (fc : FlipConfig) (address : c.address = p.address) :
    let trial := search (l := l) p c b orders fc
    Works mc fc b trial (Memory.Good p mc fc) (recorded trial) := by
  intro trial s before good cover positive window
  obtain ⟨k,ans,after,returned,cert,safe⟩ := PvsParallel.SerialTable.Trial.trial_returns p c b
    ((orders s).base (fc := fc)) (orders s).root s before address good cover positive window
  exact ⟨k,ans,after,returned,cert,safe,PvsParallel.SerialTable.Trial.opening_le returned,returned⟩

def FromBoard (p : Pool) (c : PvsParallel.SerialScout.Config) (b : Position)
    (orders : SerialStack.State → MoveInputs mc) (fc : FlipConfig)
    (birth : MoveList.Initialization.Memory Move.Move) (m : Machine) (before : Memory l)
    (sort : PublicationMir.SliceSort) (nodes : Nat) (fallback : Unit → PublicationMir.SearchResult) : Prop :=
  let rows := NativeRootBirth.initial mc b
  let trial := search p c b orders fc
  NativeRootBirth.started mc fc b c.paired birth = some rows ∧
    Published mc fc b trial (Memory.Good p mc fc) (recorded trial) rows m before sort nodes fallback

theorem from_warm (p : Pool) (c : PvsParallel.SerialScout.Config) (b : Position)
    (orders : SerialStack.State → MoveInputs mc) (fc : FlipConfig)
    (birth : MoveList.Initialization.Memory Move.Move) (m : Machine) (before : Memory l)
    (address : c.address = p.address) (valid : before.Good p mc fc)
    (checked : Std.Tactic.BVDecide.LRAT.check certificate (LegalMoves.MoveBound.dimacsFormula 35) = true)
    (mobile : Board.hasLegalMoves mc b.val = true) (pc : m.pc = 0)
    (window : Aspiration.Window m.regs.alpha.toInt m.regs.beta.toInt) (sort nodes fallback) :
    FromBoard p c b orders fc birth m before sort nodes fallback := by
  have started := started_of_certificate mc fc b c.paired birth checked mobile
  obtain ⟨cover,positive⟩ := NativeRootBirth.covers started
  exact ⟨started,publication _ (works p c b orders fc address) valid cover (List.length_pos_iff.mpr positive)
    m pc window sort nodes fallback⟩

def ColdContract : Prop :=
  ∀ (l : Layout) (generation : BitVec 8) (p : Pool) (mc : X86Config) (fc : FlipConfig)
    (c : PvsParallel.SerialScout.Config) (b : Position) (orders : SerialStack.State → MoveInputs mc) (birth : MoveList.Initialization.Memory Move.Move) (m : Machine) certificate sort nodes fallback,
    c.address = p.address →
    Std.Tactic.BVDecide.LRAT.check certificate (LegalMoves.MoveBound.dimacsFormula 35) = true →
    Board.hasLegalMoves mc b.val = true → m.pc = 0 → Aspiration.Window m.regs.alpha.toInt m.regs.beta.toInt →
    FromBoard p c b orders fc birth m (Memory.empty l generation p) sort nodes fallback

theorem cold : ColdContract := by
  intro l generation p mc fc c b orders birth m certificate sort nodes fallback address checked mobile pc window
  exact from_warm p c b orders fc birth m _ address (Memory.empty_good l generation p mc fc)
    checked mobile pc window sort nodes fallback

end NeuralReversi.AspirationMir.SerialTable

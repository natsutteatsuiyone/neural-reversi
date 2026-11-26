import NeuralReversi.PvsParallel.Interleaving.RootProgram
import NeuralReversi.PvsParallel.Interleaving.NativeNodes
import NeuralReversi.PvsParallel.Interleaving.WeakFrame
import NeuralReversi.PvsParallel.Interleaving.FiniteTable

namespace NeuralReversi.PvsParallel.Interleaving.RootCausal

open PvsBoardTree (Position GeneratedSquare play minimax)
open CausalTable (Values Lookups)
open RootMoveStorage (Row)
open AspirationMir
open AspirationMir.SerialTable
open Aspiration (Window SoundResult)
open PvsMoveLoop (scoreMax)
open NativeTableWrites.Journal (Even)
open Bitboard (Square)

private theorem childValue {b : Position} (m : GeneratedSquare mc b.val) :
    -(Root.squareValue mc fc b m.val) = minimax mc fc (play mc fc b m).val := by
  rw [Root.square_value_eq_child m,Root.childValue,Int.neg_neg]

/-- A root store for the full board may use only PV index zero. Other root
    attempts may still contribute ordinary child frames to this same history. -/
inductive Body (mc : Bitboard.Moves.X86Config) (fc : Root.FlipConfig) (prior : Values) (reads : Lookups)
    (b : Position) : Bool → Int → Int → Int → Prop where
  | node (frame : NativeNodes.Body mc fc prior reads b pv a z v) : Body mc fc prior reads b pv a z v
  | root (s : SerialStack.State) (op : RootProgram.Operands mc b (BoardFrame.Children mc fc b prior) s.regs.beta.toInt)
      (covers : Root.Covers mc b (s.rows.map Row.square)) (positive : 0 < s.rows.length)
      (answer : RootProgram.run fc 0 s op = some ans) :
      Body mc fc prior reads b true s.regs.alpha.toInt s.regs.beta.toInt ans.value

theorem preserves : CausalTable.Preserves mc fc (Body mc fc) := by
  intro prior reads children words b p a z v window narrow run
  cases run with
  | node frame => exact NativeNodes.preserves _ _ children words _ _ _ _ _ window narrow frame
  | root s op cover pos answer =>
    have cert := ((RootProgram.ran op answer window).properties (lvl := 5) (fun no => (no rfl).elim)
      (fun _ m cp lo hi score w n issued => childValue m ▸ children _ _ _ _ _ w n issued) cover pos window).2.2.2 rfl
    obtain ⟨_,_,_,sound⟩ := cert
    exact (Root.value_zero cover (by simpa using pos)) ▸ sound.1

def frames (weak exact : List CausalTable.Frame) (level : Nat) := if level = 5 then exact else weak

def children (mc : Bitboard.Moves.X86Config) (fc : Root.FlipConfig) (b : Position)
    (before : Nat → Nat → Bool) (weak exact : List CausalTable.Frame) : RootLadder.Children mc b :=
  fun level m pv a z v => ∃ event, CausalTable.returned (fun i j => before i j) (frames weak exact level)
    event (play mc fc b m).val pv a z v

/-- A reply lookup checks board, PV/cut kind, depth, bounds, and native return
    order before the owner can consume the selected child word. -/
structure Sites where
  depth : Nat → TranspositionTable.Depth
  cut : Nat → Bool
  event : Position → Bool → Int → Int → Nat
  select : Position → TranspositionTable.Depth → Bool → Bool → Int → Int → Option Nat

def Sites.atCall (s : Sites) (records : List CausalTable.Frame) (before : Nat → Nat → Bool)
    (b : Position) (pv : Bool) (a z : Int) : NativeNodes.Returns :=
  ⟨records,s.depth,s.cut,before,s.event b pv a z,s.select⟩

def Sites.reply (s : Sites) (mc : Bitboard.Moves.X86Config) (fc : Root.FlipConfig) (b : Position)
    (before : Nat → Nat → Bool) (weak exact : List CausalTable.Frame) (level : Nat) :
    RootOwnerMir.Replies (GeneratedSquare mc b.val) := fun m pv a z =>
  let q := SerialRootCalls.request mc fc b (Board.emptyCount b.val) ⟨m,pv,a,z⟩
  (s.atCall (frames weak exact level) before q.board q.pv q.alpha q.beta).reply q.board q.depth q.pv q.cut q.alpha q.beta

theorem Sites.hit {b : Position} (s : Sites) (move : GeneratedSquare mc b.val)
    (used : s.reply mc fc b order weak exact level move p a z = some v) :
    children mc fc b order weak exact level move p a z v :=
  ⟨_,(s.atCall (frames weak exact level) order (play mc fc b move).val p a z).hit used⟩

theorem laws {x : NativeTableWrites.Journal.Execution c endian rs} {init : ∀ a, Even (x.memory a)}
    (stores : FiniteTable.Stores x init) (safe : ∀ a, Table.Safe mc fc endian ((x.slot a (init a)).payloadAt 0))
    (finite : ∀ a, FiniteTable.Words endian ((x.slot a (init a)).payloadAt 0))
    (weakRun : FiniteTable.Prefix x init weak (fun i j => order i j = true) (WeakFrame.Frame mc fc))
    (exactRun : CausalTable.Prefix x init exact (fun i j => order i j = true) (Body mc fc)) :
    RootLadder.Laws mc fc b (children mc fc b order weak exact) := by
  have range := (FiniteTable.certified stores finite weakRun WeakFrame.preserves).1
  have sound := (CausalTable.certified x init exactRun preserves safe).1
  constructor
  · intro lvl earlier m pv a z v used
    obtain ⟨_,f,mem,_,_,_,_,_,_,_,w⟩ := used
    rw [←w]
    exact range f (by simpa [frames,earlier] using mem)
  · intro m pv a z v win narrow used
    obtain ⟨_,f,mem,_,_,_,board,kind,lo,hi,score⟩ := used
    rw [childValue m,←board,←lo,←hi,←score]
    exact sound f (by simpa [frames] using mem) (lo ▸ hi ▸ win) (by simpa only [kind,lo,hi] using narrow)

theorem from_flags {x : NativeTableWrites.Journal.Execution c endian rs} {init : ∀ a, Even (x.memory a)}
    (stores : FiniteTable.Stores x init) (safe : ∀ a, Table.Safe mc fc endian ((x.slot a (init a)).payloadAt 0))
    (finite : ∀ a, FiniteTable.Words endian ((x.slot a (init a)).payloadAt 0))
    (weakRun : FiniteTable.Prefix x init weak (fun i j => order i j = true) (WeakFrame.Frame mc fc))
    (exactRun : CausalTable.Prefix x init exact (fun i j => order i j = true) (Body mc fc))
    (ops : RootProgram.Supplies M mc b (children mc fc b order weak exact)) (multi : Bool)
    (clock : MultiPublic.Clock) (cover : Root.Covers mc b (rows.map Row.square)) (pos : rows ≠ [])
    (range : -scoreMax ≤ base.toInt ∧ base.toInt ≤ scoreMax) (origin : MultiClock.Origin minimum prior)
    (enabled : 1 ≤ cap) (time : clock.ordered ((if multi then rows.length else 1)-1))
    (returned : Publication.solve (RootProgram.source fc ops) multi clock cap minimum prior base rows m sort nodes fallback =
      some (.bestMove report,next)) (tag : report.endgame = true) (full : report.stamp.selectivity = 5) :
    report.best.score = 256 * EndgameTreeCache.discValue mc fc b.val ∧
      (multi = true → MultiRows.ReportExact mc fc b report) :=
  Publication.from_flags _ (laws stores safe finite weakRun exactRun) multi clock cover pos range origin enabled time returned tag full

end NeuralReversi.PvsParallel.Interleaving.RootCausal

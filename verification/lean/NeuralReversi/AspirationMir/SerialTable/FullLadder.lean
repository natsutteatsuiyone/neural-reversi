import NeuralReversi.AspirationMir.SerialTable.InitialEstimate
import NeuralReversi.PvsParallel.SerialTable.WeakTrial

namespace NeuralReversi.AspirationMir.SerialTable.FullLadder

open PvsParallel
open PvsBoardTree (Position)
open PvsParallel.SerialTable (Layout Memory Table)
open EndgameSearch.CacheHistory (Pool)
open TranspositionTable (Selectivity)
open RootMoveStorage (Row)
open SerialStack (State)
open SelectiveLadder (Start FiniteTrial)

structure Inputs (mc : Bitboard.Moves.X86Config) where
  order : Nat → State → MoveInputs mc
  screen : Nat → State → PvsParallel.SerialTable.WeakStack.Screen

def trial (p : Pool) (c : SerialScout.Config) (b : Position) (ops : Inputs mc) (fc : Cutoffs.FlipConfig)
    (lvl : Nat) : SerialCache.Trial (Memory l) := fun s before =>
  if h : lvl < 5 then
    let order := ops.order lvl s
    PvsParallel.SerialTable.WeakTrial.trial p c b (order.base (fc := fc)) order.root ⟨lvl,by omega⟩
      (ops.screen lvl s) s before
  else none

theorem earlier_returns (p : Pool) (c : SerialScout.Config) (b : Position) (ops : Inputs mc)
    (fc : Cutoffs.FlipConfig) (address : c.address = p.address) (lvl : Nat) (early : lvl ∈ OuterMir.ladder.take 4) :
    FiniteTrial mc b (Memory.Valid (l := l) p mc fc) (trial p c b ops fc lvl) := by
  have weak : lvl < 5 := by simp [OuterMir.ladder] at early; omega
  intro s before valid cover positive window
  let order := ops.order lvl s
  obtain ⟨k,ans,next,ran,safe,range,names⟩ := PvsParallel.SerialTable.WeakTrial.trial_returns p c address b
    (order.base (fc := fc)) order.root ⟨lvl,by omega⟩ weak (ops.screen lvl s) s before valid cover positive window
  exact ⟨k,ans,next,by simpa only [trial,weak,↓reduceDIte,order] using ran,safe,range.1,range.2,.of_eq names⟩

/-- Every root call at levels 1–4 shares the same worker caches and clustered TT. The zeroed image
    also determines the first aspiration base. The fifth level publishes the minimax score. -/
def ColdContract : Prop :=
  ∀ (l : Layout) (g : BitVec 8) (p : Pool) mc fc (c : SerialScout.Config) (b : Position)
    (ops : Inputs mc) (raw : Position → Int) (birth : MoveList.Initialization.Memory Move.Move)
    certificate sort nodes fallback,
    c.address = p.address → Std.Tactic.BVDecide.LRAT.check certificate (LegalMoves.MoveBound.dimacsFormula 35) = true →
    Board.hasLegalMoves mc b.val = true →
    let rows := NativeRootBirth.initial mc b
    let base := InitialEstimate.estimate (Table.empty l g).image mc fc raw b
    let start : Start (Memory l) := ⟨rows,SerialEntry.firstMachine base,Memory.empty l g p⟩
    let stages := trial (l := l) p c b ops fc
    let finalTrial := search (l := l) p c b (ops.order 5) fc
    NativeRootBirth.started mc fc b c.paired birth = some rows ∧
    ∃ final, (OuterMir.ladder.take 4).foldlM (SelectiveLadder.advance stages) start = some final ∧
      SelectiveLadder.Levels mc b (Memory.Valid p mc fc) stages (OuterMir.ladder.take 4) start final ∧
      Published mc fc b finalTrial (Memory.Good p mc fc) (recorded finalTrial)
        final.rows final.machine final.memory sort nodes fallback

theorem cold : ColdContract := by
  intro l g p mc fc c b ops raw birth cert sort nodes fallback address verified mobile
  have bounded := InitialEstimate.cold_bounded l g mc fc raw b
  exact SelectiveLadder.cold l g p mc fc c b (trial p c b ops fc) (ops.order 5) birth _ cert sort nodes fallback
    address verified mobile bounded.1 bounded.2 (earlier_returns p c b ops fc address)

end NeuralReversi.AspirationMir.SerialTable.FullLadder

module
public import NeuralReversi.PvsParallel.SerialTable.WeakStack.Program
public import NeuralReversi.PvsParallel.SerialTable.Trial
public import NeuralReversi.PvsParallel.SerialTable.WeakRoot

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialTable.WeakTrial

open EndgameSearch.CacheHistory (Pool)
open PvsBoardTree (Position)
open SerialExecution (Frame Inputs push)
open SerialCalls (Program)
open SerialCalls.ScoreWords (Finite)
open TranspositionTable (Selectivity)
open AspirationMir.SerialStack (State)
open AspirationMir.SmallRoot (Answer)
open RootMoveStorage (Row)

/-- The serial root retains its probe slot across all descendant writes, then publishes with
    the selectivity of this level. The next retry receives the resulting table and worker caches. -/
def trial (p : Pool) (c : SerialScout.Config) (b : Position) (base : Inputs mc fc) (root : SerialRecursion.RootInput)
    (lvl : Selectivity) (screen : WeakStack.Screen) (s : State) (prior : Memory l) : Option (Nat × Answer × Memory l) := do
  let ctx := SerialRootExecution.context b s.regs.alpha s.regs.beta
  let slot := prior.tt.image.slot b
  let code := Root.frame mc fc b ctx.request.depth c.paired (base s.opening ctx).memory s.rows 0
    (Trial.ordering prior base b root s) s.regs.alpha s.regs.beta
  let (k,out,after) ← Stack.interpret (mc := mc) (fc := fc) (WeakStack.run p c base lvl screen) s.opening ctx prior code
  return (k,out.answer,after.apply slot out.proposal lvl)

theorem trial_of_children (p : Pool) (c : SerialScout.Config) (b : Position) (base : Inputs mc fc)
    (root : SerialRecursion.RootInput) (lvl : Selectivity) (weak : lvl.val < 5) (screen : WeakStack.Screen)
    (s : State) (prior : Memory l) (valid : prior.Valid p mc fc)
    (covers : PvsParallel.Root.Covers mc b (s.rows.map Row.square)) (within : 0 < s.rows.length)
    (window : Aspiration.Window s.regs.alpha.toInt s.regs.beta.toInt)
    (descendants : ∀ i (ctx : Frame) (m : Memory l), SerialTermination.Entry ctx → m.Valid p mc fc →
      ∃ k w next, WeakStack.run p c base lvl screen i ctx m = some (k,w,next) ∧ next.Valid p mc fc ∧ Finite w) :
    ∃ k ans next, trial p c b base root lvl screen s prior = some (k,ans,next) ∧ next.Valid p mc fc ∧
      Finite ans.value ∧ ans.rows.map Row.square = s.rows.map Row.square := by
  let ctx := SerialRootExecution.context b s.regs.alpha s.regs.beta
  let order := Trial.ordering prior base b root s
  let code := Root.frame mc fc b ctx.request.depth c.paired (base s.opening ctx).memory s.rows 0 order s.regs.alpha s.regs.beta
  obtain ⟨entry,ready⟩ := SerialCacheStack.root_ready c b base order s covers within window
  change Program.Ready _ (SerialRootCalls.frame ..) at ready
  rw [← Root.erase] at ready
  obtain ⟨k,out,after,ran,good,cs,issued,finite⟩ := WeakStack.interpret_returns (code := code) (ctx := ctx)
    (Stack.ready_of_map ready) (WeakStack.run p c base lvl screen) (fun q allowed i m ok => by
      obtain ⟨child,pushed,aligned,_⟩ := SerialTermination.child_entry entry q allowed.1
      obtain ⟨j,w,next,ran,safe,range⟩ := descendants i child m aligned ok
      exact ⟨child,j,w,next,pushed,ran,safe,range⟩) s.opening prior valid
  obtain ⟨range,same⟩ := WeakRoot.result issued covers within finite
  let slot := prior.tt.image.slot b
  refine ⟨k,out.answer,after.apply slot out.proposal lvl,?_,?_,range,same⟩
  · dsimp only [trial]
    rw [ran]
    rfl
  · apply after.apply_weak good slot lvl weak
    intro w written
    obtain ⟨_,board,_,_,_,_,legal⟩ := Root.writes issued written
    exact board ▸ legal

end NeuralReversi.PvsParallel.SerialTable.WeakTrial

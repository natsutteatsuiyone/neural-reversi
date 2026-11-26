import NeuralReversi.PvsParallel.SerialTable.Finite
import NeuralReversi.PvsParallel.SerialTable.WeakPv
import NeuralReversi.PvsParallel.SerialTable.SelectiveFrame

namespace NeuralReversi.PvsParallel.SerialTable.WeakStack

open EndgameSearch.CacheHistory (Pool)
open SerialExecution (Frame Inputs push)
open SerialCalls (Program Request)
open SerialCalls.ScoreWords (Finite encode)
open SerialTermination (Entry Allowed)
open TranspositionTable (Selectivity)

abbrev Screen := Nat → Frame → SelectiveFrame.Estimates

def attempts (m : Memory l) (c : SerialScout.Config) (base : Inputs mc fc) (i : Nat) (ctx : Frame) :=
  let q := ctx.request
  if SelectiveFrame.preAction mc fc q.board q.depth (encode q.alpha) (encode q.beta) c.paired c.checked
      (base i ctx).memory = .rest then m.tt.image.attempts q.board else []

def operands (m : Memory l) (c : SerialScout.Config) (base : Inputs mc fc) (level : Selectivity)
    (i : Nat) (ctx : Frame) : SerialRecursion.Operands mc fc ctx.request.board :=
  let q := ctx.request
  let op := base i ctx
  let seen := attempts m c base i ctx
  let children := if 14 ≤ q.depth.toNat ∧ !seen.isEmpty ∧
        (SelectiveFrame.direct q.board q.depth (encode q.beta) c.checked level seen).isNone then
      m.tt.image.children m.supported mc fc q.board (SelectiveFrame.query q.depth level (encode q.alpha))
        (PvsBoardTree.legalSquares mc q.board.val)
    else []
  ⟨op.memory,m.cache.history,m.tt.history,seen,children,op.ordering,op.tie⟩

theorem reads (m : Memory l) (c : SerialScout.Config) (base : Inputs mc fc) (i : Nat) (ctx : Frame) :
    TableReads.ProbeReads m.tt.history (attempts m c base i ctx) := by
  dsimp only [attempts]
  split
  · exact m.tt.image.probe_reads m.supported ctx.request.board
  · simp [TableReads.ProbeReads,TableReads.ProbeReadsFrom]

def body (m : Memory l) (c : SerialScout.Config) (base : Inputs mc fc) (level : Selectivity) (screen : Screen)
    (i : Nat) (ctx : Frame) : Program Request Writes.Result :=
  let q := ctx.request
  if q.pv then
    let op := m.inputs c base i ctx
    Writes.pv mc fc q.board q.depth c.paired c.checked q.cut op.memory q.alpha q.beta (SerialRecursion.pvOrder q.board op)
  else
    let op := operands m c base level i ctx
    SelectiveFrame.frame mc fc q.board q.depth (encode q.alpha) (encode q.beta) c.paired c.checked q.cut op.memory
      level (screen i ctx) op.probe op.etc (EndgameOrderMir.framePriority mc fc q.board op.ordering op.tie)

def runFuel (p : Pool) (c : SerialScout.Config) (base : Inputs mc fc) (level : Selectivity) (screen : Screen) :
    Nat → Stack.Child l
  | 0, _, _, _ => none
  | n+1, i, ctx, m =>
    if ctx.request.pv = false ∧ ctx.request.depth.toNat ≤ 12 then Stack.run p c base i ctx m
    else do
      let slot := m.tt.image.slot ctx.request.board
      let (k,out,after) ← Stack.interpret (mc := mc) (fc := fc) (runFuel p c base level screen n)
        i ctx m (body m c base level screen i ctx)
      return (k,out.score,after.apply slot out.store level)

def run (p : Pool) (c : SerialScout.Config) (base : Inputs mc fc) (level : Selectivity) (screen : Screen)
    (i : Nat) (ctx : Frame) (m : Memory l) :=
  runFuel p c base level screen (Board.playRank mc ctx.request.board.val + 1) i ctx m

theorem interpret_returns {code : Program Request α} (ready : Program.Ready allowed code)
    (descend : Stack.Child l) (enters : ∀ q, allowed q → ∀ j (m : Memory l), m.Valid p mc fc →
      ∃ child k w next, push mc fc ctx q = some child ∧ descend j child m = some (k,w,next) ∧
        next.Valid p mc fc ∧ Finite w) (i : Nat) (m : Memory l) (valid : m.Valid p mc fc) :
    ∃ k v after, Stack.interpret (mc := mc) (fc := fc) descend i ctx m code = some (k,v,after) ∧
      after.Valid p mc fc ∧ ∃ cs, Program.Runs code cs v ∧ ∀ t ∈ cs, Finite t.2 := by
  obtain ⟨k,v,post,done_,good,cs,issued,sub⟩ := Stack.interpret_valid (word := fun _ w => Finite w)
    ready descend enters i m valid
  exact ⟨k,v,post,done_,good,cs,issued,fun t ht => (sub t ht).choose_spec.2⟩

private theorem represented (a z : Int) (window : Aspiration.Window a z) :
    (encode a).raw.toInt = a ∧ (encode z).raw.toInt = z := by
  obtain ⟨lo,order,hi⟩ := window
  exact ⟨BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega),
    BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)⟩

theorem body_ready (m : Memory l) (c : SerialScout.Config) (base : Inputs mc fc) (level : Selectivity) (screen : Screen)
    (i : Nat) (ctx : Frame) (good : m.Valid p mc fc) (entry : Entry ctx)
    (regular : ¬ (ctx.request.pv = false ∧ ctx.request.depth.toNat ≤ 12)) :
    Program.Ready (Allowed mc fc ctx) (body m c base level screen i ctx) := by
  dsimp only [body]
  cases role : ctx.request.pv <;> simp only [Bool.false_eq_true,ite_false,ite_true]
  · have ⟨a,z⟩ := represented _ _ entry.2.2.1
    have deep : 12 < ctx.request.depth.toNat := Nat.lt_of_not_ge (fun h => regular ⟨role,h⟩)
    apply SelectiveFrame.frame_ready entry.2.1 deep
    · rw [a,z]; exact entry.2.2.1
    · rw [a,z]; exact entry.2.2.2 role
    · exact reads m c base i ctx
    · exact good.1.1.hints
  · apply Stack.ready_of_map (f := Writes.Result.score)
    rw [Writes.pv_score]
    exact SerialPvReady.frame_ready entry.2.1 entry.2.2.1 _ _ _ _ _

end NeuralReversi.PvsParallel.SerialTable.WeakStack

module
public import NeuralReversi.AspirationMir.SerialTable.Repeated

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialTable.MultiExact

open EndgameSearch.CacheHistory (Pool)
open PvsBoardTree (Position)
open SerialExecution (Inputs)
open SerialCalls (Program)
open AspirationMir.SmallRoot (Answer)
open AspirationMir.SerialStack (State)
open AspirationMir.SerialTable.Repeated (ReadyMemory)
open RootMoveStorage (Row)

def Certified (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (idx : Nat) (s : State) (ans : Answer) : Prop :=
  ∃ snap : RootReplay.Snapshot, ans.value = snap.state.best ∧
    RootSuffix.Writes s.rows idx s.regs.alpha.toInt s.regs.beta.toInt ans.rows snap ∧
    RootOrder.SoundAttempt mc fc b (s.rows.map Row.square) idx s.regs.alpha.toInt s.regs.beta.toInt ans.value snap.scores

def trial (idx : Nat) (p : Pool) (c : SerialScout.Config) (b : Position) (base : Inputs mc fc)
    (root : SerialRecursion.RootInput) (s : State) (prior : Memory l) : Option (Nat × Answer × Memory l) := do
  let ctx := SerialRootExecution.context b s.regs.alpha s.regs.beta
  let code := Root.frame mc fc b ctx.request.depth c.paired (base s.opening ctx).memory s.rows idx
    (Trial.ordering prior base b root s) s.regs.alpha s.regs.beta
  let (k,out,after) ← Stack.interpret (mc := mc) (fc := fc) (Stack.run p c base) s.opening ctx prior code
  return (k,out.answer,after.apply (prior.tt.image.slot b) out.proposal)

theorem trial_returns (idx : Nat) (p : Pool) (c : SerialScout.Config) (b : Position) (base : Inputs mc fc)
    (root : SerialRecursion.RootInput) (s : State) (prior : Memory l) (address : c.address = p.address)
    (ready : ReadyMemory p mc fc prior) (covers : PvsParallel.Root.Covers mc b (s.rows.map Row.square))
    (within : idx < s.rows.length) (window : Aspiration.Window s.regs.alpha.toInt s.regs.beta.toInt) :
    ∃ k ans next, trial idx p c b base root s prior = some (k,ans,next) ∧
      Certified mc fc b idx s ans ∧ ReadyMemory p mc fc next := by
  by_cases first : idx = 0
  · subst idx
    obtain ⟨k,ans,next,ran,cert,good⟩ := Trial.trial_returns p c b base root s prior address ready.1.1 covers within window
    obtain ⟨finite,bounded⟩ := ExactRetention.trial ready.1.2 ready.2 ran good
    exact ⟨k,ans,next,ran,cert,⟨⟨good,finite⟩,bounded⟩⟩
  let ctx := SerialRootExecution.context b s.regs.alpha s.regs.beta
  let selected := Trial.ordering prior base b root s
  let code := Root.frame mc fc b ctx.request.depth c.paired (base s.opening ctx).memory s.rows idx
    selected s.regs.alpha s.regs.beta
  have entry : SerialTermination.Entry ctx := by
    simp only [ctx,SerialTermination.Entry,SerialRootExecution.context,ContextPathMir.start_eq]
    exact ⟨EndgameSearch.emptyState_initial b,rfl,window,by simp⟩
  have live : Program.Ready (SerialTermination.Allowed mc fc ctx) (code.map Root.Exit.answer) := by
    dsimp only [code]
    rw [Root.erase]
    exact SerialRootCalls.frame_ready covers within entry.2.1 window
  obtain ⟨k,out,after,ran,post,calls,issued,children⟩ := Stack.interpret_valid
    (valid := ReadyMemory p mc fc) (word := SerialExecution.Invariant mc fc) (Stack.ready_of_map live)
    (Stack.run p c base) (fun q allowed j m ok => by
      obtain ⟨child,pushed,aligned,_⟩ := SerialTermination.child_entry entry q allowed
      obtain ⟨last,v,next,returned,good,inv⟩ := Stack.run_returns p c address base j child m aligned ok.1.1
      obtain ⟨finite,bounded⟩ := ExactRetention.frame ok.1.2 ok.2 returned good
      exact ⟨child,last,v,next,pushed,returned,⟨⟨good,finite⟩,bounded⟩,inv⟩) s.opening prior ready
  have owner := Stack.runs_map issued Root.Exit.answer
  dsimp only [code] at owner
  rw [Root.erase] at owner
  obtain ⟨snap,best,writes,_,sound⟩ := SerialRootCalls.frame_replay owner covers within window (fun t h => by
    obtain ⟨_,_,_,_,w,narrow⟩ := SerialRootCalls.frame_edges owner window t h
    exact ⟨_,SerialExecution.child_replay entry.2.1 entry.1 (children t h) w narrow⟩)
  have absent : out.proposal = none := by
    cases saved : out.proposal with
    | none => rfl
    | some w => exact (first (Root.writes issued saved).1).elim
  refine ⟨k,out.answer,after,?_,⟨snap,best,writes,sound⟩,post⟩
  dsimp only [trial]
  rw [ran]
  simp [absent,Memory.apply]

end NeuralReversi.PvsParallel.SerialTable.MultiExact

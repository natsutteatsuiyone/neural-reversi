module
public import NeuralReversi.PvsParallel.SerialTable.Stack
public import NeuralReversi.PvsParallel.SerialTable.Root

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialTable.Trial

open EndgameSearch.CacheHistory (Pool)
open PvsBoardTree (Position minimax)
open SerialExecution (Inputs)
open SerialCalls (Program Request)
open AspirationMir.SmallRoot (Answer)
open AspirationMir.SerialStack (State Certified)

/-- Both the hint and the retained store destination come from the image at root entry.
    For four or more moves, the library supplies the order of equal keys. -/
def ordering (prior : Memory l) (base : Inputs mc fc) (b : Position) (root : SerialRecursion.RootInput)
    (s : State) : SerialRecursion.RootInput :=
  let ctx := SerialRootExecution.context b s.regs.alpha s.regs.beta
  let hint := GeneralNonPvMir.probeHint b (prior.tt.image.attempts b)
  let key := EndgameOrderMir.actualKey (EndgameOrderMir.priority mc fc b (base s.opening ctx).ordering) hint
  let library : RootFrameMir.LibraryOrder := fun records =>
    let order := root.sort records
    let priority := fun m : Move.Move => m.value.toInt
    ⟨MoveList.Ordering.sortByKeyDesc priority order.val,
      (MoveList.Ordering.sortByKeyDesc_perm priority order.val).trans order.property⟩
  ⟨fun sq => BitVec.ofInt 32 (key sq),library,root.pvLine⟩

def trial (p : Pool) (c : SerialScout.Config) (b : Position) (base : Inputs mc fc)
    (root : SerialRecursion.RootInput) (s : State) (prior : Memory l) : Option (Nat × Answer × Memory l) := do
  let ctx := SerialRootExecution.context b s.regs.alpha s.regs.beta
  let slot := prior.tt.image.slot b
  let code := Root.frame mc fc b ctx.request.depth c.paired (base s.opening ctx).memory s.rows 0
    (ordering prior base b root s) s.regs.alpha s.regs.beta
  let (k,out,after) ← Stack.interpret (mc := mc) (fc := fc) (Stack.run p c base) s.opening ctx prior code
  return (k,out.answer,after.apply slot out.proposal)

theorem trial_returns (p : Pool) (c : SerialScout.Config) (b : Position) (base : Inputs mc fc)
    (root : SerialRecursion.RootInput) (s : State) (prior : Memory l) (address : c.address = p.address)
    (good : prior.Good p mc fc) (covers : PvsParallel.Root.Covers mc b (s.rows.map RootMoveStorage.Row.square))
    (within : 0 < s.rows.length) (window : Aspiration.Window s.regs.alpha.toInt s.regs.beta.toInt) :
    ∃ k ans next, trial p c b base root s prior = some (k,ans,next) ∧ Certified mc fc b s ans ∧
      next.Good p mc fc := by
  let ctx := SerialRootExecution.context b s.regs.alpha s.regs.beta
  let selected := ordering prior base b root s
  let code := Root.frame mc fc b ctx.request.depth c.paired (base s.opening ctx).memory s.rows 0
    selected s.regs.alpha s.regs.beta
  obtain ⟨entry,ready⟩ := SerialCacheStack.root_ready c b base selected s covers within window
  change Program.Ready _ (SerialRootCalls.frame ..) at ready
  rw [← Root.erase] at ready
  obtain ⟨k,out,after,returned,valid,cs,issued,children⟩ := Stack.interpret_returns
    (code := code) (ctx := ctx) (Stack.ready_of_map ready) (Stack.run p c base) (fun q allowed j m ok => by
      obtain ⟨child,pushed,aligned,_⟩ := SerialTermination.child_entry entry q allowed.1
      obtain ⟨last,v,next,ran,holds,inv⟩ := Stack.run_returns p c address base j child m aligned ok
      exact ⟨child,last,v,next,pushed,ran,holds,inv⟩) s.opening prior good
  have owner := Stack.runs_map issued Root.Exit.answer
  dsimp only [code] at owner
  rw [Root.erase] at owner
  obtain ⟨snap,best,writes,_,checked⟩ := SerialRootCalls.frame_replay owner covers within window (fun t h => by
    obtain ⟨_,_,_,_,w,narrow⟩ := SerialRootCalls.frame_edges owner window t h
    exact ⟨_,SerialExecution.child_replay entry.2.1 entry.1 (children t h) w narrow⟩)
  let slot := prior.tt.image.slot b
  refine ⟨k,out.answer,after.apply slot out.proposal,?_,⟨snap,best,writes,checked⟩,?_⟩
  · dsimp only [trial]
    rw [returned]
    rfl
  · apply after.apply_good valid slot (b := b) (d := ctx.request.depth) (pv := true) (v := out.answer.value) _ window
    · exact PvsParallel.Root.value_zero covers (by simpa using within) ▸ checked.1
    · intro w saved
      obtain ⟨_,board,depth,value,tag,bound,hint⟩ := Root.writes issued saved
      exact ⟨true,board,depth,value,hint,tag,by simpa [Writes.word] using bound⟩

private theorem interpret_clock {code : Program Request α} {descend : Stack.Child l}
    (child : ∀ j ctx prev last w post, descend j ctx prev = some (last,w,post) → j ≤ last)
    (ran : Stack.interpret (mc := mc) (fc := fc) descend i ctx m code = some (k,v,next)) : i ≤ k := by
  induction code generalizing i m with
  | fail => cases ran
  | pure x => cases ran; exact Nat.le_refl _
  | call q resume ih =>
    obtain ⟨f,_,body⟩ := Option.bind_eq_some_iff.mp ran
    obtain ⟨⟨j,w,prev⟩,sub,rest⟩ := Option.bind_eq_some_iff.mp body
    exact Nat.le_trans (Nat.le_succ i) (Nat.le_trans (child _ _ _ _ _ _ sub) (ih w rest))

private theorem fuel_clock {p : Pool} {c : SerialScout.Config} {base : Inputs mc fc} {m : Memory l}
    (ran : Stack.runFuel p c base fuel i ctx m = some (k,v,next)) : i ≤ k := by
  induction fuel generalizing i ctx m k v next with
  | zero => cases ran
  | succ n ih =>
    unfold Stack.runFuel at ran
    split at ran
    next leaf =>
      obtain ⟨⟨j,word,cells⟩,result,out⟩ := Option.map_eq_some_iff.mp ran
      obtain ⟨rfl,rfl,_⟩ := Prod.mk.inj out
      unfold SerialCacheStack.Cells.run at result
      dsimp only [SerialCacheStack.Cells.runFuel] at result
      rw [ite_eq_left leaf] at result
      obtain ⟨⟨⟨w,h⟩,state⟩,_,more⟩ := Option.bind_eq_some_iff.mp result
      obtain ⟨updated,_,done_⟩ := Option.bind_eq_some_iff.mp more
      cases done_
      exact Nat.le_refl _
    next =>
      dsimp only at ran
      obtain ⟨⟨j,out,post⟩,result,done_⟩ := Option.bind_eq_some_iff.mp ran
      cases done_
      exact interpret_clock (fun _ _ _ _ _ _ => ih) result

/-- The first activation of a retry, at closing + 1, therefore follows every
    activation of the completed root trial. -/
theorem opening_le {p : Pool} {c : SerialScout.Config} {base : Inputs mc fc} {prior : Memory l}
    (ran : trial p c b base root s prior = some (k,ans,next)) : s.opening ≤ k := by
  dsimp only [trial] at ran
  obtain ⟨⟨j,out,post⟩,result,done_⟩ := Option.bind_eq_some_iff.mp ran
  cases done_
  exact interpret_clock (fun _ _ _ _ _ _ => fuel_clock) result

end NeuralReversi.PvsParallel.SerialTable.Trial

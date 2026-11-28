module
public import NeuralReversi.PvsParallel.SerialCacheStack.Projection

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialCacheStack

open SerialCalls (Program Request Executes)
open SerialExecution (Frame Inputs Invariant body push)
open SerialTermination (Allowed Entry)
open EndgameTreeCache (soundHistory)
open SerialCalls.ScoreWords (encode)
open PvsBoardTree (Position)
open AspirationMir.SmallRoot (Answer)

theorem depth_le (edge : ExactDepth.RecursiveCall mc fc b d child cd) (full : ExactDepth.Full b d) :
    cd.toNat ≤ d.toNat := by
  cases edge with
  | ply m d =>
    have positive := (ExactDepth.move_depth mc fc b d m full).1
    rw [BitVec.toNat_sub_of_le (by change 1 ≤ d.toNat; exact positive)]
    omega
  | forcedPass => exact Nat.le_refl _

theorem request_of_push (entered : push mc fc ctx q = some child) : child.request = q := by
  obtain ⟨_, _, rfl⟩ := Option.map_eq_some_iff.mp entered
  rfl

theorem fuel_returns (c : SerialScout.Config) (inputs : Inputs mc fc) (facts : SharedFacts inputs limit)
    (fuel i : Nat) (ctx : Frame) (past : History) (rank : Board.playRank mc ctx.request.board.val < fuel)
    (bound : ctx.request.depth.toNat ≤ limit) (entry : Entry ctx) (ok : soundHistory mc fc past) :
    ∃ k v after, runFuel c inputs fuel i ctx past = some (k,v,after) ∧ soundHistory mc fc after ∧
      past <+: after ∧ Invariant mc fc ctx v ∧ SourceFrame c inputs i ctx past k v := by
  induction fuel generalizing i ctx past with
  | zero => omega
  | succ n ih =>
    have ⟨valid, hints⟩ := shared_reads facts past ok
    let proofInputs : Inputs mc fc := source (capped inputs limit) (fun _ => past)
    have sameNode := body_at c inputs i ctx past bound
    have certify {v calls} (ran : Program.Runs (body c (source inputs (fun _ => past)) i ctx) calls v)
        (used : ∀ t ∈ calls, ∃ child, push mc fc ctx t.1 = some child ∧ Invariant mc fc child t.2) :
        Invariant mc fc ctx v :=
      SerialExecution.localRule c proofInputs valid i ctx calls v (sameNode.symm ▸ ran) used
    by_cases leaf : ctx.request.pv = false ∧ ctx.request.depth.toNat ≤ 12
    · have fit := entry.1
      have full := entry.2.1
      have ⟨lo, ordered, hi⟩ := entry.2.2.1
      have a : (encode ctx.request.alpha).raw.toInt = ctx.request.alpha :=
        BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
      have z : (encode ctx.request.beta).raw.toInt = ctx.request.beta :=
        BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
      have size : ctx.empties.count.toNat ≤ 12 := by rw [fit.1, ← full]; exact leaf.2
      have width : (encode ctx.request.beta).raw.toInt = (encode ctx.request.alpha).raw.toInt + 1 := by
        rw [a,z]; exact entry.2.2.2 leaf.1
      obtain ⟨v,after,computed,_,sound,extension⟩ := EndgameSearch.SerialLeafRestore.contract_proven c mc fc
        ctx.request.board ctx.empties _ _ past fit size ok width
      have sourceLeaf : body c (source inputs (fun _ => past)) i ctx = .pure v := by
        unfold body
        simp only [leaf.1, Bool.false_eq_true, ite_false,leaf.2,ite_true]
        change Program.ofOption ((EndgameSearch.SerialLeafRestore.leaf c mc fc ctx.request.board ctx.empties
          (encode ctx.request.alpha) (encode ctx.request.beta) past).map _) = _
        rw [computed]; rfl
      refine ⟨i,v,after,?_,sound,extension,certify (calls := []) (by rw [sourceLeaf]; exact .pure) (by simp),?_⟩
      · simp [runFuel,leaf,computed]
      · exact ⟨fun _ => past,[],fun _ => ok,rfl,by rw [sourceLeaf]; exact .pure⟩
    · have ready := SerialTermination.frame_ready c proofInputs valid hints i ctx entry
      rw [show body c proofInputs i ctx = _ from sameNode] at ready
      obtain ⟨k,v,after,ran,sound,extension,model⟩ := interpret_returns (c := c) (inputs := inputs) ready
        (runFuel c inputs n) (fun q admitted j h supported => by
          obtain ⟨child,entered,aligned,smaller⟩ := SerialTermination.child_entry entry q admitted
          have rq := request_of_push entered
          have childBound : child.request.depth.toNat ≤ limit := by
            rw [rq]; exact Nat.le_trans (depth_le admitted.1 entry.2.1) bound
          obtain ⟨k,w,out,done_,sound,ext_,inv,sub⟩ :=
            ih j child h (by omega) childBound aligned supported
          exact ⟨child,k,w,out,entered,done_,sound,ext_,inv,sub⟩) i past ok
      obtain ⟨_, forest, _, issued, children, _⟩ := model.execution
      exact ⟨k,v,after,by simpa [runFuel,leaf] using ran,sound,extension,
        certify issued children,model.asFrame ok⟩

theorem root_ready (c : SerialScout.Config) (b : Position) (inputs : Inputs mc fc)
    (root : SerialRecursion.RootInput) (s : AspirationMir.SerialStack.State)
    (covers : Root.Covers mc b (s.rows.map RootMoveStorage.Row.square)) (within : 0 < s.rows.length)
    (window : Aspiration.Window s.regs.alpha.toInt s.regs.beta.toInt) :
    let ctx := SerialRootExecution.context b s.regs.alpha s.regs.beta
    Entry ctx ∧ Program.Ready (fun q => Allowed mc fc ctx q ∧ q.depth.toNat ≤ (Board.emptyCount b.val).toNat-1)
      (SerialRootExecution.attempt c inputs s.opening b root s.rows 0 s.regs.alpha s.regs.beta) := by
  let ctx := SerialRootExecution.context b s.regs.alpha s.regs.beta
  have entry : Entry ctx := by
    simp only [ctx,Entry,SerialRootExecution.context,ContextPathMir.start_eq]
    exact ⟨EndgameSearch.emptyState_initial b,rfl,window,by simp⟩
  let code := SerialRootExecution.attempt c inputs s.opening b root s.rows 0 s.regs.alpha s.regs.beta
  have ready : Program.Ready (Allowed mc fc ctx) code :=
    SerialRootCalls.frame_ready covers within entry.2.1 window
  have restricted : Program.Ready (fun q => Allowed mc fc ctx q ∧ q.depth.toNat ≤ (Board.emptyCount b.val).toNat-1) code := by
    apply Program.Ready.of_runs (ready.weaken (fun _ _ => trivial))
    intro calls ans owner t mem
    obtain ⟨m, board, depth, _, w, narrow⟩ := SerialRootCalls.frame_edges owner window t mem
    have positive := (ExactDepth.move_depth mc fc b _ m entry.2.1).1
    refine ⟨⟨?_,w,narrow⟩,?_⟩
    · rw [board,depth]; exact .ply b m _
    · rw [depth,BitVec.toNat_sub_of_le (by change 1 ≤ ctx.request.depth.toNat; exact positive)]
      change ctx.request.depth.toNat - 1 ≤ (Board.emptyCount b.val).toNat - 1
      exact entry.2.1 ▸ Nat.le_refl _
  exact ⟨entry, restricted⟩

theorem trial_returns (c : SerialScout.Config) (b : Position) (inputs : Inputs mc fc)
    (root : SerialRecursion.RootInput) (s : AspirationMir.SerialStack.State) (past : History)
    (facts : SharedFacts inputs ((Board.emptyCount b.val).toNat - 1)) (ok : soundHistory mc fc past)
    (covers : Root.Covers mc b (s.rows.map RootMoveStorage.Row.square)) (within : 0 < s.rows.length)
    (window : Aspiration.Window s.regs.alpha.toInt s.regs.beta.toInt) :
    ∃ k ans after, trial c b inputs root s past = some (k,ans,after) ∧ AspirationMir.SerialStack.Certified mc fc b s ans ∧
      soundHistory mc fc after ∧ past <+: after ∧ SourceProgram c inputs s.opening
        (SerialRootExecution.context b s.regs.alpha s.regs.beta)
        (SerialRootExecution.attempt c inputs s.opening b root s.rows 0 s.regs.alpha s.regs.beta) k ans := by
  obtain ⟨entry,restricted⟩ := root_ready c b inputs root s covers within window
  obtain ⟨k,ans,after,returned,sound,extension,model⟩ := interpret_returns (c := c) (inputs := inputs) restricted
    (run c inputs) (fun q admitted i h good => by
      obtain ⟨child, pushed, aligned, _⟩ := SerialTermination.child_entry entry q admitted.1
      have limit : child.request.depth.toNat ≤ (Board.emptyCount b.val).toNat-1 := request_of_push pushed ▸ admitted.2
      obtain ⟨last,w,history,ran,supported,ext_,inv,trace⟩ :=
        fuel_returns c inputs facts (Board.playRank mc child.request.board.val + 1) i child h (by omega) limit aligned good
      exact ⟨child,last,w,history,pushed,ran,supported,ext_,inv,trace⟩) s.opening past ok
  obtain ⟨_, forest, _, ran, children, _⟩ := model.execution
  obtain ⟨snap,best,writes,_,checked⟩ := SerialRootCalls.frame_replay ran covers within window (fun t h => by
    obtain ⟨_,_,_,_,w,narrow⟩ := SerialRootCalls.frame_edges ran window t h
    exact ⟨_,SerialExecution.child_replay entry.2.1 entry.1 (children t h) w narrow⟩)
  exact ⟨k,ans,after,returned,⟨snap,best,writes,checked⟩,sound,extension,model⟩

end NeuralReversi.PvsParallel.SerialCacheStack

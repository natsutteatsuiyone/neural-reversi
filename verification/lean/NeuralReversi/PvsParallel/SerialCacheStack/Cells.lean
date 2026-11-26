import NeuralReversi.PvsParallel.SerialCacheStack
import NeuralReversi.EndgameSearch.SerialCacheReads

namespace NeuralReversi.PvsParallel.SerialCacheStack.Cells

open EndgameSearch.CacheHistory (Pool Local)
open EndgameSearch.SerialLeafCache (Reachable)
open SerialCalls (Request Program)
open SerialExecution (Frame Inputs push)
open SerialTermination (Allowed Entry)
open PvsBoardTree (Position)
open SerialCalls.ScoreWords (encode)
open AspirationMir.SmallRoot (Answer)
open Bitboard.Moves (X86Config)
open Cutoffs (FlipConfig)

abbrev Output (α : Type) := Nat × α × Local
abbrev Child := Nat → Frame → Local → Option (Output Int)

def interpret {mc : X86Config} {fc : FlipConfig} (descend : Child) (i : Nat) (ctx : Frame) (prior : Local) :
    Program Request α → Option (Output α)
  | .pure v => some (i,v,prior)
  | .fail => none
  | .call q resume => do
    let child ← push mc fc ctx q
    let (k,w,next) ← descend (i+1) child prior
    interpret (mc := mc) (fc := fc) descend k ctx next (resume w)

def runFuel (p : Pool) (c : SerialScout.Config) (inputs : Inputs mc fc) : Nat → Child
  | 0, _, _, _ => none
  | fuel+1, i, ctx, prior =>
    let q := ctx.request
    if q.pv = false ∧ q.depth.toNat ≤ 12 then do
      let ((w,after),_) ← EndgameSearch.SerialCacheReads.leaf p c mc fc q.board ctx.empties
        (encode q.alpha) (encode q.beta) prior
      let cells ← (after.drop prior.history.length).foldlM (EndgameSearch.CacheMir.writeCells p.width p.key) prior.cells
      return (i,w,⟨after,cells⟩)
    else interpret (mc := mc) (fc := fc) (runFuel p c inputs fuel) i ctx prior
      (SerialExecution.body c (source inputs (fun _ => prior.history)) i ctx)

def run (p : Pool) (c : SerialScout.Config) (inputs : Inputs mc fc) (i : Nat) (ctx : Frame) (prior : Local) :=
  runFuel p c inputs (Board.playRank mc ctx.request.board.val + 1) i ctx prior

def trial (p : Pool) (c : SerialScout.Config) (b : Position) (inputs : Inputs mc fc)
    (root : SerialRecursion.RootInput) (s : AspirationMir.SerialStack.State) (prior : Local) : Option (Output Answer) :=
  let a := s.regs.alpha
  let z := s.regs.beta
  interpret (mc := mc) (fc := fc) (run p c inputs) s.opening (SerialRootExecution.context b a z) prior
    (SerialRootExecution.attempt c inputs s.opening b root s.rows 0 a z)

def Together (p : Pool) (mc : X86Config) (fc : FlipConfig)
    (actual : Option (Output α)) (history : Option (SerialCacheStack.Output α)) : Prop :=
  ∃ k v next, actual = some (k,v,next) ∧ history = some (k,v,next.history) ∧ Reachable p mc fc next

private theorem interpret_together {prog : Program Request α} (ready : Program.Ready allowed prog)
    (left : Child) (right : SerialCacheStack.Child)
    (descend : ∀ q, allowed q → ∀ j prior, Reachable p mc fc prior →
      ∃ child, push mc fc ctx q = some child ∧ Together p mc fc (left j child prior) (right j child prior.history))
    (i : Nat) (prior : Local) (boot : Reachable p mc fc prior) :
    Together p mc fc (interpret (mc := mc) (fc := fc) left i ctx prior prog)
      (SerialCacheStack.interpret (mc := mc) (fc := fc) right i ctx prior.history prog) := by
  induction ready generalizing i prior with
  | pure => exact ⟨_,_,prior,rfl,rfl,boot⟩
  | @call q resume ok more ih =>
    obtain ⟨child,pushed,mid,w,next,l,r,reach⟩ := descend q ok (i+1) prior boot
    obtain ⟨k,v,post,afterL,afterR,final⟩ := ih w mid next reach
    exact ⟨k,v,post,by simpa [interpret,pushed,l] using afterL,
      by simpa [SerialCacheStack.interpret,pushed,r] using afterR,final⟩

theorem fuel_together (p : Pool) (c : SerialScout.Config) (address : c.address = p.address)
    (inputs : Inputs mc fc) (facts : SharedFacts inputs limit) (fuel i : Nat) (ctx : Frame) (prior : Local)
    (rank : Board.playRank mc ctx.request.board.val < fuel) (bound : ctx.request.depth.toNat ≤ limit)
    (entry : Entry ctx) (boot : Reachable p mc fc prior) :
    Together p mc fc (runFuel p c inputs fuel i ctx prior) (SerialCacheStack.runFuel c inputs fuel i ctx prior.history) := by
  induction fuel generalizing i ctx prior with
  | zero => omega
  | succ n ih =>
    by_cases leaf : ctx.request.pv = false ∧ ctx.request.depth.toNat ≤ 12
    · have full := entry.2.1
      have aligned := entry.1
      obtain ⟨lo,ordered,hi⟩ := entry.2.2.1
      have size : ctx.empties.count.toNat ≤ 12 := by rw [aligned.1,← full]; exact leaf.2
      have narrow : (encode ctx.request.beta).raw.toInt = (encode ctx.request.alpha).raw.toInt + 1 := by
        have a : (encode ctx.request.alpha).raw.toInt = ctx.request.alpha :=
          BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
        have z : (encode ctx.request.beta).raw.toInt = ctx.request.beta :=
          BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
        rw [a,z]; exact entry.2.2.2 leaf.1
      let args : EndgameSearch.SerialLeafCache.Call := ⟨c,ctx.request.board,ctx.empties,
        encode ctx.request.alpha,encode ctx.request.beta,aligned,size,narrow⟩
      obtain ⟨w,next,ret⟩ := EndgameSearch.SerialLeafCache.leaf_step args p mc fc prior boot
      have config : {c with address := p.address} = c := by rw [← address]
      have score := ret.returned
      simp only [EndgameSearch.SerialLeafCache.Call.leaf,args,config] at score
      have ran := ret.executed
      simp only [EndgameSearch.SerialLeafCache.Call.execute,EndgameSearch.SerialLeafCache.Call.leaf,args,config,score] at ran
      refine ⟨i,w,next,?_,by simp [SerialCacheStack.runFuel,leaf,score],ret.nextReachable⟩
      simp only [runFuel,leaf,EndgameSearch.SerialCacheReads.leaf_eq p c address mc fc _ _ _ _ prior boot,score,bind,Option.bind] at ran ⊢
      split at ran <;> simp_all
    · obtain ⟨valid,hints⟩ := shared_reads facts prior.history boot.1
      have ready := SerialTermination.frame_ready c _ valid hints i ctx entry
      rw [body_at c inputs i ctx prior.history bound] at ready
      simp only [runFuel,SerialCacheStack.runFuel,leaf,ite_false]
      apply interpret_together ready _ _ _ i prior boot
      intro q ok j prev reachable
      obtain ⟨child,pushed,aligned,smaller⟩ := SerialTermination.child_entry entry q ok
      have depth : child.request.depth.toNat ≤ limit := by
        rw [request_of_push pushed]; exact Nat.le_trans (depth_le ok.1 entry.2.1) bound
      exact ⟨child,pushed,ih j child prev (by omega) depth aligned reachable⟩

theorem trial_together (p : Pool) (c : SerialScout.Config) (address : c.address = p.address)
    (b : Position) (inputs : Inputs mc fc) (root : SerialRecursion.RootInput)
    (s : AspirationMir.SerialStack.State) (prior : Local)
    (facts : SharedFacts inputs ((Board.emptyCount b.val).toNat-1)) (boot : Reachable p mc fc prior)
    (covers : Root.Covers mc b (s.rows.map RootMoveStorage.Row.square)) (within : 0 < s.rows.length)
    (window : Aspiration.Window s.regs.alpha.toInt s.regs.beta.toInt) :
    Together p mc fc (trial p c b inputs root s prior) (SerialCacheStack.trial c b inputs root s prior.history) := by
  obtain ⟨entry,ready⟩ := root_ready c b inputs root s covers within window
  apply interpret_together ready _ _ _ s.opening prior boot
  intro q ok i prev reachable
  obtain ⟨child,pushed,aligned,_⟩ := SerialTermination.child_entry entry q ok.1
  have bound : child.request.depth.toNat ≤ (Board.emptyCount b.val).toNat-1 := request_of_push pushed ▸ ok.2
  exact ⟨child,pushed,fuel_together p c address inputs facts _ i child prev (by omega) bound aligned reachable⟩

theorem trial_eq (p : Pool) (c : SerialScout.Config) (address : c.address = p.address)
    (b : Position) (inputs : Inputs mc fc) (root : SerialRecursion.RootInput)
    (s : AspirationMir.SerialStack.State) (prior : Local)
    (facts : SharedFacts inputs ((Board.emptyCount b.val).toNat-1)) (boot : Reachable p mc fc prior)
    (covers : Root.Covers mc b (s.rows.map RootMoveStorage.Row.square)) (within : 0 < s.rows.length)
    (window : Aspiration.Window s.regs.alpha.toInt s.regs.beta.toInt)
    (done_ : SerialCacheStack.trial c b inputs root s prior.history = some (k,answer,after)) :
    trial p c b inputs root s prior = some (k,answer,⟨after,EndgameSearch.SerialLeafCache.snapshot p after⟩) ∧
      Reachable p mc fc ⟨after,EndgameSearch.SerialLeafCache.snapshot p after⟩ := by
  obtain ⟨j,v,next,observed,hist,reachable⟩ :=
    trial_together p c address b inputs root s prior facts boot covers within window
  rw [done_] at hist
  obtain ⟨rfl,rfl,rfl⟩ := Prod.mk.inj (Option.some.inj hist)
  obtain ⟨h,cells⟩ := next
  have eq := Option.some.inj (reachable.2.symm.trans (EndgameSearch.SerialLeafCache.materialize_eq p h))
  cases eq
  exact ⟨observed,reachable⟩

end NeuralReversi.PvsParallel.SerialCacheStack.Cells

module
public import NeuralReversi.PvsParallel.IteratorFrames
public import NeuralReversi.PvsParallel.NativeFrames.Returns

@[expose] public section

namespace NeuralReversi.PvsParallel.IteratorFrames

open NativeFrames (Args Capture)
open NativeWorkerFeed (Prefix goodRow stepOk)
open SplitWorkerMir (Query Step scoutQuery wideQuery)
open PvsBoardTree (Position GeneratedSquare play)

variable {s : MoveList.ConcurrentTrace.State w}
variable (x : Posted rc s journal stream earlier report seq)

/-- Entry registers at the two search calls in search_split_point. Both calls
    use the move returned by this worker's actual iterator operation. -/
def childArgs (parent : Board.Board) (mv : Move.Move) (q : Query) : Args :=
  ⟨Board.place parent mv.flipped mv.sq,BitVec.ofNat 32 q.depth,⟨q.alpha⟩,⟨q.beta⟩,
    true,Cutoffs.exactSelectivity,q.pv,q.cut,none⟩

structure EntryOperands (inputs : Nat → Nat → Option Args) (parent : Board.Board)
    (x : Posted rc s journal stream earlier report seq) : Prop where
  scout : inputs x.op.worker x.lock.scoutOpening = some (childArgs parent x.move (scoutQuery x.input))
  wide : SplitWorkerMir.retry x.input = true →
    inputs x.op.worker x.lock.wideOpening = some (childArgs parent x.move (wideQuery x.input))

private theorem EntryOperands.at_return (op : EntryOperands inputs parent x)
    (issued : Step.returned q ∈ x.body.steps ++ SplitWorkerMir.lockedSteps) :
    inputs x.op.worker (x.lock.prefix.opening q.pv) = some (childArgs parent x.move q) ∧ q.depth < 2^32 := by
  have bound : x.input.depth - 1 < 2^32 := Nat.lt_of_le_of_lt (Nat.sub_le _ _) x.lock.task.depth.isLt
  rw [SplitWorkerMir.issued_of_search x.searched] at issued
  cases again : SplitWorkerMir.retry x.input with
  | false =>
    obtain rfl : q = scoutQuery x.input := by simpa [SplitWorkerMir.issued,again,SplitWorkerMir.lockedSteps] using issued
    exact ⟨op.scout,bound⟩
  | true =>
    have kinds : q = scoutQuery x.input ∨ q = wideQuery x.input := by simpa [SplitWorkerMir.issued,again,SplitWorkerMir.lockedSteps] using issued
    rcases kinds with rfl | rfl
    · exact ⟨op.scout,bound⟩
    · exact ⟨op.wide again,bound⟩

private theorem frozen_cell {i : Nat} {mv : Move.Move} {b : Position} {order : List (GeneratedSquare mc b.val)}
    (layout : NativeWorkerFeed.frozen mc fc b rc order = true) (cell : rc[i]? = some mv) :
    ∃ m, order[i]? = some m ∧ mv.sq = m.val ∧ mv.flipped = Flip.Dispatch.resultValue fc m.val b.val.player b.val.opponent := by
  have pair := congrArg (fun xs : List (Bitboard.Square × Bitboard.Bits) => xs[i]?) (of_decide_eq_true layout)
  simp only [List.getElem?_map,cell,Option.map_some] at pair
  obtain ⟨m,where_,sqFlip⟩ := Option.map_eq_some_iff.mp pair.symm
  exact ⟨m,where_,(congrArg Prod.fst sqFlip).symm,(congrArg Prod.snd sqFlip).symm⟩

/-- A published report gets its score from a normal leave on this worker.
    The row predicate used by both the root and the internal split replay
    follows from the input registers, iterator cell, and decoded frame stream. -/
theorem Posted.child_step {b : Position} {order : List (GeneratedSquare mc b.val)} {cap : Capture pool}
    (parsed : NativeFrames.decode pool inputs stream = some cap)
    (layout : NativeWorkerFeed.frozen mc fc b rc order = true)
    (op : EntryOperands inputs b.val x) (q : Query)
    (issued : Step.returned q ∈ x.body.steps ++ SplitWorkerMir.lockedSteps) :
    stepOk mc fc b order rc x.op.old cap.rows x.returned x.lock.prefix (.returned q) = true := by
  have live : x.lock.locked.stopped = false := by
    have paid := x.credited
    rw [SplitWorkerMir.recordLocked_eq] at paid
    cases h : x.lock.locked.stopped
    · rfl
    · simp [h] at paid
  obtain ⟨idx,recorded,worker,ret,member,usable,opening,finish,word⟩ :=
    NativeWorkerFeed.issued_return x.lock.prefix x.returned x.lock.task x.op.old x.lock.alpha x.lock.fresh
      (fun wide => x.lock.stamp (.load wide)) x.lock.ends x.lock.acquired (live ▸ x.control)
      (fun _ => rfl) x.searched issued
  obtain ⟨stamp,event⟩ := NativeWorkerFeed.returned_event x.returns member usable
  obtain ⟨args,depth⟩ := op.at_return x issued
  obtain ⟨r,inRows,actor,opened,returned,line,fields⟩ :=
    NativeFrames.Returns.row_of_leave parsed event (opening ▸ args) (by rfl)
  obtain ⟨m,found,square,flips⟩ := frozen_cell layout x.cell
  have actual : FrameReturns.OwnerCalls.fields r.frame =
      ⟨(play mc fc b m).val.val,q.depth,q.pv,q.cut,q.alpha.toInt,q.beta.toInt,q.result.toInt⟩ := by
    simpa [NativeFrames.Returns.argsFields,childArgs,square,flips,play,Nat.mod_eq_of_lt depth,word] using fields
  have sameBoard : r.frame.position = (play mc fc b m).val := Subtype.ext (congrArg FrameReturns.OwnerCalls.ChildReturn.board actual)
  rw [stepOk,x.ticket,idx,found]
  simp only [square,flips,recorded]
  apply List.any_eq_true.mpr
  refine ⟨r,inRows,decide_eq_true ?_⟩
  unfold goodRow
  exact ⟨by simp [NativeFrames.whole,line,childArgs],opened.trans opening,actor.trans x.worker.symm,worker,
    returned.trans (congrArg some finish),sameBoard,congrArg FrameReturns.OwnerCalls.ChildReturn.depth actual,
    congrArg FrameReturns.OwnerCalls.ChildReturn.pv actual,congrArg FrameReturns.OwnerCalls.ChildReturn.cut actual,
    congrArg FrameReturns.OwnerCalls.ChildReturn.alpha actual,congrArg FrameReturns.OwnerCalls.ChildReturn.beta actual,
    congrArg FrameReturns.OwnerCalls.ChildReturn.score actual⟩

end NeuralReversi.PvsParallel.IteratorFrames

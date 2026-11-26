import NeuralReversi.PvsParallel.SerialTable.Writes
import NeuralReversi.PvsParallel.SerialTermination

namespace NeuralReversi.PvsParallel.SerialTable.Execution

open SerialCalls (Program Request)
open SerialExecution (Frame Inputs)
open SerialCalls.ScoreWords (encode)

def body (c : SerialScout.Config) (inputs : Inputs mc fc) (i : Nat) (ctx : Frame) : Program Request Writes.Result :=
  let r := ctx.request
  let op := inputs i ctx
  if r.pv then
    Writes.pv mc fc r.board r.depth c.paired c.checked r.cut op.memory r.alpha r.beta
      (SerialRecursion.pvOrder r.board op)
  else if r.depth.toNat ≤ 12 then
    .ofOption ((EndgameSearch.SerialLeafRestore.leaf c mc fc r.board ctx.empties (encode r.alpha) (encode r.beta)
      op.cache).map fun out => ⟨out.1.1,none⟩)
  else
    Writes.nonPv mc fc r.board r.depth (encode r.alpha) (encode r.beta) c.paired c.checked r.cut
      op.memory op.probe op.etc (EndgameOrderMir.framePriority mc fc r.board op.ordering op.tie)

theorem body_score (c : SerialScout.Config) (inputs : Inputs mc fc) (i : Nat) (ctx : Frame) :
    (body c inputs i ctx).map Writes.Result.score = SerialExecution.body c inputs i ctx := by
  unfold body SerialExecution.body
  dsimp only
  split
  · apply Writes.pv_score
  split
  · cases EndgameSearch.SerialLeafRestore.leaf c mc fc ctx.request.board ctx.empties (encode ctx.request.alpha)
      (encode ctx.request.beta) (inputs i ctx).cache <;> rfl
  · apply Writes.nonPv_score

theorem produced (c : SerialScout.Config) (inputs : Inputs mc fc) (i : Nat) (ctx : Frame)
    (entry : SerialTermination.Entry ctx) (ran : Program.Runs (body c inputs i ctx) children out)
    (saved : out.store = some w) (finite : SerialCalls.ScoreWords.Finite out.score) :
    let q := ctx.request
    Writes.Produced q.board q.depth q.pv (Writes.parentPv q.pv q.board (inputs i ctx).probe) q.alpha q.beta out.score w := by
  obtain ⟨_,_,⟨lo,ordered,hi⟩,null⟩ := entry
  have a : (encode ctx.request.alpha).raw.toInt = ctx.request.alpha :=
    BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
  have z : (encode ctx.request.beta).raw.toInt = ctx.request.beta :=
    BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
  dsimp only
  unfold body at ran
  dsimp only at ran
  cases pv : ctx.request.pv with
  | true =>
    simp only [pv,ite_true] at ran
    exact Writes.pv_produced ran saved
  | false =>
    simp only [pv,Bool.false_eq_true,ite_false] at ran
    split at ran
    · obtain ⟨ret,_⟩ := Program.runs_ofOption ran
      obtain ⟨_,_,rfl⟩ := Option.map_eq_some_iff.mp ret
      contradiction
    · have small := null pv
      have proof := Writes.nonPv_produced ran saved (by rw [a,z]; exact small) finite
      simpa only [a,z] using proof

end NeuralReversi.PvsParallel.SerialTable.Execution

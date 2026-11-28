module
public import NeuralReversi.PvsParallel.SerialTable.WeakStack.Program
public import NeuralReversi.PvsParallel.SerialTable.WeakPv

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialTable.WeakStack

open SerialExecution (Frame Inputs)
open EndgameSearch.CacheHistory (Pool)
open TranspositionTable (Selectivity)
open SerialCalls (Program)
open SerialCalls.ScoreWords (Finite encode)
open SerialTermination (Entry)

private theorem local_return (m : Memory l) (c : SerialScout.Config) (base : Inputs mc fc)
    (level : Selectivity) (screen : Screen) (i : Nat) (ctx : Frame) (entry : Entry ctx)
    (valid : m.Valid p mc fc)
    (ran : Program.Runs (body m c base level screen i ctx) cs out) (finite : ∀ t ∈ cs, Finite t.2) :
    Finite out.score ∧ ∀ w, out.store = some w → TableReads.LegalChoice w.board w.move := by
  dsimp only [body] at ran
  cases role : ctx.request.pv <;> simp only [role,Bool.false_eq_true,ite_false,ite_true] at ran
  · have encoded (v : Int) (lo : -32767 ≤ v) (hi : v ≤ 32767) : (encode v).raw.toInt = v :=
      BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
    obtain ⟨lo,mid,hi⟩ := entry.2.2.1
    have a := encoded ctx.request.alpha lo (by omega)
    have z := encoded ctx.request.beta (by omega) hi
    have narrow : (encode ctx.request.beta).raw.toInt = (encode ctx.request.alpha).raw.toInt+1 := by
      rw [a,z]; exact entry.2.2.2 role
    have window : Aspiration.Window (encode ctx.request.alpha).raw.toInt (encode ctx.request.beta).raw.toInt := by
      rw [a,z]; exact entry.2.2.1
    have v := SelectiveFrame.frame_finite entry.2.1 window narrow (reads m c base i ctx) valid.2 ran finite
    refine ⟨v,fun w saved => ?_⟩
    have h := SelectiveFrame.frame_produced ran saved narrow v
    rw [h.board]
    exact h.hint
  · refine ⟨pv_finite entry.2.1 (EndgameOrderMir.ordinary_perm _ _ _ _ _) entry.2.2.1 ran finite,fun w saved => ?_⟩
    have h := Writes.pv_produced ran saved
    rw [h.board]
    exact h.hint

private theorem shallow_table {ctx : Frame} {m : Memory l} {base : Inputs mc fc}
    (leaf : ctx.request.pv = false ∧ ctx.request.depth.toNat ≤ 12)
    (ran : Stack.run p c base i ctx m = some (k,w,after)) : after.tt = m.tt := by
  simp only [Stack.run,Stack.runFuel,leaf] at ran
  obtain ⟨⟨_,_,_⟩,_,eq⟩ := Option.map_eq_some_iff.mp ran
  cases eq
  rfl

theorem fuel_returns (p : Pool) (c : SerialScout.Config) (address : c.address = p.address)
    (base : Inputs mc fc) (level : Selectivity) (weak : level.val < 5) (screen : Screen)
    (fuel i : Nat) (ctx : Frame) (m : Memory l) (rank : Board.playRank mc ctx.request.board.val < fuel)
    (entry : Entry ctx) (valid : m.Valid p mc fc) :
    ∃ k w next, runFuel p c base level screen fuel i ctx m = some (k,w,next) ∧ next.Valid p mc fc ∧ Finite w := by
  induction fuel generalizing i ctx m with
  | zero => omega
  | succ n ih =>
    by_cases leaf : ctx.request.pv = false ∧ ctx.request.depth.toNat ≤ 12
    · obtain ⟨k,v,next,ran,good,inv⟩ := Stack.run_returns p c address base i ctx m entry valid.1
      have eq := shallow_table leaf ran
      exact ⟨k,v,next,by simpa [runFuel,leaf] using ran,⟨good,eq ▸ valid.2⟩,
        SerialCalls.ScoreWords.finite_of_replay entry.2.2.1 entry.2.2.2
          (inv entry.1 entry.2.1 entry.2.2.1 entry.2.2.2)⟩
    · have ready := body_ready m c base level screen i ctx valid entry leaf
      obtain ⟨k,out,after,ran,post,cs,issued,children⟩ := interpret_returns ready (runFuel p c base level screen n)
        (fun q allowed j prev ok => by
          obtain ⟨child,pushed,childEntry,smaller⟩ := SerialTermination.child_entry entry q allowed
          obtain ⟨last,v,next,done_,good,h⟩ := ih j child prev (by omega) childEntry ok
          exact ⟨child,last,v,next,pushed,done_,good,h⟩) i m valid
      have ⟨range,legal⟩ := local_return m c base level screen i ctx entry valid issued children
      let slot := m.tt.image.slot ctx.request.board
      refine ⟨k,out.score,after.apply slot out.store level,?_,after.apply_weak post slot level weak legal,range⟩
      simp only [runFuel,leaf,ite_false,ran,bind,Option.bind,pure,slot]

theorem run_returns (p : Pool) (c : SerialScout.Config) (address : c.address = p.address)
    (base : Inputs mc fc) (level : Selectivity) (weak : level.val < 5) (screen : Screen)
    (i : Nat) (ctx : Frame) (m : Memory l) (entry : Entry ctx) (valid : m.Valid p mc fc) :
    ∃ k w next, run p c base level screen i ctx m = some (k,w,next) ∧ next.Valid p mc fc ∧ Finite w :=
  fuel_returns p c address base level weak screen _ i ctx m (by omega) entry valid

end NeuralReversi.PvsParallel.SerialTable.WeakStack

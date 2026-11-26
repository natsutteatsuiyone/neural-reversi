import NeuralReversi.PvsParallel.SerialTable.Finite
import NeuralReversi.PvsParallel.SerialTable.Trial

namespace NeuralReversi.PvsParallel.SerialTable.ExactRetention

open TranspositionTable
open EndgameSearch.CacheHistory (Pool)
open SerialExecution (Frame Inputs)
open SerialCalls (Program Request)
open PvsMoveLoop (scoreMax)
open PvsBoardTree (Position)

/-- During an exact phase, the numeric range of *earlier* exact-bound words is inherited.
    New full-depth words obtain their range from the mini-max bounds of their own board. -/
private def Retained (m : Memory l) : Prop :=
  FiniteHistory m.tt.history ∧ ∀ b e, (b,e) ∈ m.tt.history → ¬Cutoffs.ExactEndgame e →
    e.bound = Packing.boundByte .exact → -scoreMax ≤ e.score.toInt ∧ e.score.toInt ≤ scoreMax

private theorem admitted {b prev data old e} (made : Save.proposal b prev data old = some e)
    (tag : Cutoffs.ExactEndgame data) : Cutoffs.ExactEndgame e := by
  rw [(Save.contract_proven ..).1] at made
  split at made
  · contradiction
  · cases Option.some.inj made
    unfold Save.payload
    split <;> exact tag

private theorem apply_retains (m : Memory l) (h : Retained m) (slot : l.Index) (proposal : Option Writes.Proposal) :
    Retained (m.apply slot proposal) := by
  cases proposal with
  | none => exact h
  | some w =>
    refine ⟨m.tt.store_finite m.supported h.1 _ _ _ _ _ _ _ _ _,?_⟩
    dsimp only [Memory.apply]
    rw [Table.store,WriterMir.store_input]
    by_cases ok : (Writes.word w.score).raw.toInt.natAbs < 32767 <;> simp only [ok,ite_true,ite_false]
    · rw [m.tt.save_eq _ _ _ m.supported]
      split
      next e made =>
        intro b data member other bound
        rcases List.mem_append.mp member with before | last
        · exact h.2 b data before other bound
        · obtain ⟨rfl,rfl⟩ := Prod.mk.inj (List.mem_singleton.mp last)
          apply False.elim (other (admitted made ?_))
          change (1#8 ||| (if w.ttPv then 2 else 0)).getLsbD 0 = true ∧ 5 ≤ (5#8).toNat
          cases w.ttPv <;> decide
      next => exact h.2
    · exact h.2

private theorem interpret {code : Program Request α} {descend : Stack.Child l}
    (children : ∀ i ctx m k w post, Retained m → descend i ctx m = some (k,w,post) → Retained post)
    (h : Retained m) (ok : Stack.interpret (mc := mc) (fc := fc) descend i ctx m code = some (k,x,post)) :
    Retained post := by
  induction code generalizing i m with
  | fail => cases ok
  | pure => cases ok; exact h
  | call q resume ih =>
    obtain ⟨child,_,more⟩ := Option.bind_eq_some_iff.mp ok
    obtain ⟨⟨j,w,after⟩,ran,rest⟩ := Option.bind_eq_some_iff.mp more
    exact ih w (children _ _ _ _ _ _ h ran) rest

private theorem fuel {p : Pool} {c : SerialScout.Config} {base : Inputs mc fc} {m : Memory l}
    (h : Retained m) (ran : Stack.runFuel p c base n i ctx m = some (k,w,post)) : Retained post := by
  induction n generalizing i ctx m k w post with
  | zero => cases ran
  | succ n ih =>
    unfold Stack.runFuel at ran
    split at ran
    next =>
      obtain ⟨⟨_,_,_⟩,_,eq⟩ := Option.map_eq_some_iff.mp ran
      cases eq
      exact h
    next =>
      obtain ⟨⟨j,out,after⟩,steps,done_⟩ := Option.bind_eq_some_iff.mp ran
      cases done_
      exact apply_retains _ (interpret (fun _ _ _ _ _ _ => ih) h steps) _ _

private theorem fields {post : Memory l} (h : Retained post) (sound : post.Good p mc fc) :
    FiniteHistory post.tt.history ∧ ExactScores post.tt.history := by
  refine ⟨h.1,fun board e member bound => ?_⟩
  by_cases tag : Cutoffs.ExactEndgame e
  · have correct := sound.1.publications.sound board e member tag
    have lo := correct.1 (by rw [bound]; decide)
    have hi := correct.2 (by rw [bound]; decide)
    have limits := PvsMoveLoop.minimax_range (PvsBoardTree.build mc fc board)
    change -scoreMax ≤ PvsBoardTree.minimax mc fc board ∧ PvsBoardTree.minimax mc fc board ≤ scoreMax at limits
    omega
  · exact h.2 board e member tag bound

theorem frame {p : Pool} {c : SerialScout.Config} {base : Inputs mc fc} {m : Memory l}
    (finite : FiniteHistory m.tt.history) (seeds : ExactScores m.tt.history)
    (ran : Stack.runFuel p c base n i ctx m = some (k,w,post)) (sound : post.Good p mc fc) :
    FiniteHistory post.tt.history ∧ ExactScores post.tt.history :=
  fields (fuel ⟨finite,fun b e member _ => seeds b e member⟩ ran) sound

theorem trial {p : Pool} {c : SerialScout.Config} {base : Inputs mc fc} {m : Memory l}
    (finite : FiniteHistory m.tt.history) (seeds : ExactScores m.tt.history)
    (ran : Trial.trial p c b base roots s m = some (k,ans,post)) (sound : post.Good p mc fc) :
    FiniteHistory post.tt.history ∧ ExactScores post.tt.history := by
  dsimp only [Trial.trial] at ran
  obtain ⟨⟨j,out,after⟩,steps,done_⟩ := Option.bind_eq_some_iff.mp ran
  cases done_
  have prior : Retained m := ⟨finite,fun b e member _ => seeds b e member⟩
  have h := apply_retains _ (interpret (fun _ _ _ _ _ _ => fuel) prior steps) (m.tt.image.slot b) out.proposal
  exact fields h sound

end NeuralReversi.PvsParallel.SerialTable.ExactRetention

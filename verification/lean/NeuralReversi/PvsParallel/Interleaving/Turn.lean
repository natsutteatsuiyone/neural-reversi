import NeuralReversi.PvsParallel.Interleaving.Join
import NeuralReversi.PvsParallel.Prefix

namespace NeuralReversi.PvsParallel.Interleaving

open Aspiration (Window SoundResult)
open PvsMoveLoop (Bound ChildRelation ChildSearch bestValue scoreInf scoreMax)
open PvsPostpone (Queue)
open Threading

/-- The owner can finish in the ordinary PVS move loop, or hand the remaining queue
    to workers. Busy claims may have rotated the suffix before that handoff. -/
inductive Turn (done_ : ChildRelation M) (moves : List M) (pv : Bool) (original beta : Int) : Int → Prop where
  | returned (reply : ChildSearch M)
      (prefixRun : Prefix.Serial.Run done_ reply pv beta ⟨0,moves,[]⟩ ⟨original,-scoreInf⟩
        visited remaining (.returned score)) : Turn done_ moves pv original beta score
  | joined (reply : ChildSearch M) (fallback : M) (root cut : Bool) (depth w : Nat)
      (prefixRun : Prefix.Serial.Run done_ reply pv beta ⟨0,moves,[]⟩ ⟨original,-scoreInf⟩
        visited remaining (.handoff entry))
      (fits : (visited ++ remaining.pending).length < 2^w)
      (j : Join ⟨pv,root,cut,depth,beta,entry⟩
        (fun i => done_ ((visited ++ remaining.pending)[i]?.getD fallback))
        (visited ++ remaining.pending).length (BitVec.ofNat w remaining.idx))
      (used : j.answer = some score) : Turn done_ moves pv original beta score

theorem Turn.bound {value : M → Int} (turn : Turn done_ moves pv original beta score)
    (nonempty : moves ≠ []) (w : Window original beta) (narrow : pv = false → beta = original+1)
    (limits : ∀ m ∈ moves, value m ≤ scoreMax)
    (children : ∀ m ∈ moves, ∀ cp lo hi v, Window lo hi → (cp = false → hi = lo+1) →
      done_ m cp lo hi v → SoundResult (-(value m)) lo hi v) :
    Bound (bestValue value moves (-scoreInf)) original beta score := by
  cases turn with
  | returned reply execution =>
    obtain ⟨order,_,same,calls⟩ := execution.reproduced
    simp only [Queue.pending,List.append_nil] at order
    rw [←same]
    have sound := PvsMoveLoop.selectedContract_proven _ pv reply value done_ _ original beta
      (fun empty => nonempty (List.perm_nil.mp (empty ▸ order.symm))) w narrow
      (fun m hm => limits m (order.mem_iff.mp hm)) calls
      (fun m hm => children m (order.mem_iff.mp hm))
    exact PvsMoveLoop.Bound.ofSoundResult ((PvsMoveLoop.bestValue_eq_of_same_moves value (-scoreInf)
      (fun _ => order.mem_iff)) ▸ sound)
  | @joined visited remaining entry _ reply fallback root cut depth width execution fits j used =>
    let labels := visited ++ remaining.pending
    let label := fun i : Nat => labels[i]?.getD fallback
    have ⟨order,cursor,continued,calls⟩ := execution.reproduced
    simp only [Queue.pending,List.append_nil] at order
    change labels.Perm moves at order
    have count : remaining.idx = visited.length := by simpa using cursor
    have cursorOK : remaining.idx < 2^width := by simp [count,List.length_append] at fits ⊢; omega
    have atIndex (i : Nat) (valid : i < labels.length) : label i ∈ moves :=
      order.mem_iff.mp (by simp [label,List.getElem?_eq_getElem valid])
    have early (m) (member : m ∈ visited) := order.mem_iff.mp (List.mem_append_left remaining.pending member)
    obtain ⟨win,cap,small,seed⟩ := Prefix.selectedContract_proven _ pv reply value visited done_ original beta
      entry w narrow continued (fun m member => limits m (early m member)) calls
      (fun m member => children m (early m member))
    have integer : (BitVec.ofNat width remaining.idx).toNat = remaining.idx := by
      simp [BitVec.toNat_ofNat,Nat.mod_eq_of_lt cursorOK]
    have maxPrefix : bestValue value visited (-scoreInf) ≤ scoreMax :=
      PvsMoveLoop.bestValue_le _ _ _ _ (by decide) (fun m hm => limits m (early m hm))
    obtain ⟨result,same⟩ := j.accepted (value := value ∘ label) fits used win cap small seed maxPrefix
      (fun i member => limits _ (atIndex i (SplitResult.mem_slots.mp member).2))
      (fun i _ inside => children (label i) (atIndex i inside))
    have bound := result.bound
    rw [same,integer,count] at bound
    have exactValue := SplitResult.partitionLabels visited remaining.pending fallback value (-scoreInf)
    have eq := PvsMoveLoop.bestValue_eq_of_same_moves value (-scoreInf) (fun _ => order.mem_iff)
    exact eq ▸ exactValue ▸ bound

end NeuralReversi.PvsParallel.Interleaving

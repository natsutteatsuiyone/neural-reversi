import Std

namespace NeuralReversi.PvsParallel.Interleaving.EventOrder

/-- The original event identities in a linear extension of the supplied edges. -/
structure Extension (n : Nat) (before : Fin n → Fin n → Bool) where
  indices : List (Fin n)
  permutation : indices.Perm (List.finRange n)
  extendsHB : ∀ i j, before i j = true → indices.idxOf i < indices.idxOf j

/-- A proposed permutation from an external sorter is only used after checking
    coverage and every ordering edge against the original event occurrences. -/
def check (before : Fin n → Fin n → Bool) (proposed : List (Fin n)) : Option (Extension n before) :=
  if ok : proposed.Perm (List.finRange n) ∧
      ∀ i j, before i j = true → proposed.idxOf i < proposed.idxOf j then
    some ⟨proposed,ok.1,ok.2⟩
  else none

theorem checked_indices (before : Fin n → Fin n → Bool) (xs : List (Fin n)) {accepted}
    (hit : check before xs = some accepted) : accepted.indices = xs := by
  unfold check at hit
  split at hit
  · cases hit
    rfl
  · contradiction

/-- Physical event IDs may originally be assigned in any order. Sort by the
    number of strict predecessors in the finite happens-before relation. Two
    incomparable events can have the same count and keep distinct positions. -/
def predecessors (hb : Fin n → Fin n → Prop) [DecidableRel hb] (r : Fin n) :=
  (List.finRange n).filter fun w => decide (hb w r)

def events (hb : Fin n → Fin n → Prop) [DecidableRel hb] :=
  (List.finRange n).mergeSort fun i j => decide ((predecessors hb i).length ≤ (predecessors hb j).length)

theorem permutation (hb : Fin n → Fin n → Prop) [DecidableRel hb] :
    (events hb).Perm (List.finRange n) := List.mergeSort_perm ..

private theorem inOrder (hb : Fin n → Fin n → Prop) [DecidableRel hb] (i : Fin n) :
    i ∈ events hb := (permutation hb).mem_iff.mpr (List.mem_finRange i)

def position (hb : Fin n → Fin n → Prop) [DecidableRel hb] (i : Fin n) : Fin n :=
  ⟨(events hb).idxOf i,by
    have bound := List.idxOf_lt_length_of_mem (inOrder hb i)
    simpa [(permutation hb).length_eq] using bound⟩

theorem atPosition (hb : Fin n → Fin n → Prop) [DecidableRel hb] (i : Fin n) :
    (events hb)[(position hb i).val]? = some i := by
  change (events hb)[(events hb).idxOf i]? = some i
  rw [List.getElem?_eq_getElem (List.idxOf_lt_length_of_mem (inOrder hb i))]
  exact congrArg some (List.getElem_idxOf ..)

theorem injective (hb : Fin n → Fin n → Prop) [DecidableRel hb] {u v : Fin n}
    (equal : position hb u = position hb v) : u = v := by
  have h := atPosition hb u
  rw [equal,atPosition] at h
  exact (Option.some.inj h).symm

theorem extendsHB (hb : Fin n → Fin n → Prop) [DecidableRel hb]
    (irrefl : ∀ a, ¬ hb a a) (trans : ∀ {a b c}, hb a b → hb b c → hb a c)
    (edge : hb u v) : position hb u < position hb v := by
  have missing : u ∉ predecessors hb u := by simp [predecessors,irrefl]
  have distinct := (List.nodup_finRange n).filter (fun w => decide (hb w u))
  have nodup : (u :: predecessors hb u).Nodup := List.nodup_cons.mpr ⟨missing,distinct⟩
  have subset : u :: predecessors hb u ⊆ predecessors hb v := by
    intro w member
    rcases List.mem_cons.mp member with rfl | earlier
    · simp [predecessors,edge]
    · exact List.mem_filter.mpr ⟨List.mem_finRange _,decide_eq_true
        (trans (of_decide_eq_true (List.mem_filter.mp earlier).2) edge)⟩
  have strict : (predecessors hb u).length < (predecessors hb v).length :=
    nodup.length_le_of_subset subset
  have sorted : (events hb).Pairwise (fun a b => (predecessors hb a).length ≤ (predecessors hb b).length) :=
    (List.pairwise_mergeSort (le := fun a b => decide ((predecessors hb a).length ≤ (predecessors hb b).length))
      (by simp only [decide_eq_true_eq]; intros; omega)
      (by simp only [Bool.or_eq_true,decide_eq_true_eq]; intro a b; exact Nat.le_total _ _) _).imp of_decide_eq_true
  apply Nat.lt_of_not_ge
  intro reversed
  rcases Nat.eq_or_lt_of_le reversed with equal | back
  · have same := injective hb (Fin.ext equal)
    simp [same] at strict
  · have atU := List.getElem?_eq_some_iff.mp (atPosition hb u)
    have atV := List.getElem?_eq_some_iff.mp (atPosition hb v)
    have contradiction := List.pairwise_iff_getElem.mp sorted _ _ atV.1 atU.1 back
    rw [atV.2,atU.2] at contradiction
    omega

/-- Filtering the single event order reproduces any happens-before ordered
    list containing precisely the selected events. -/
theorem projection (hb : Fin n → Fin n → Prop) [DecidableRel hb] (irrefl : ∀ w, ¬ hb w w)
    (trans : ∀ {u v w}, hb u v → hb v w → hb u w) (select : Fin n → Bool) (xs : List (Fin n))
    (inside : ∀ w, select w = true ↔ w ∈ xs) (chain : xs.Pairwise hb) :
    (events hb).filter select = xs := by
  let pos (w : Fin n) := (position hb w).val
  have distinct := (permutation hb).symm.nodup (List.nodup_finRange n)
  have timeline : (events hb).Pairwise (fun u v => pos u < pos v) := by
    apply List.pairwise_iff_getElem.mpr
    intro i j hi hj less
    change (events hb).idxOf (events hb)[i] < (events hb).idxOf (events hb)[j]
    simpa only [distinct.idxOf_getElem i hi,distinct.idxOf_getElem j hj] using less
  have unique : xs.Nodup := by
    apply chain.imp
    intro a b h eq
    subst b
    exact irrefl a h
  have perm : ((events hb).filter select).Perm xs := by
    apply (List.perm_ext_iff_of_nodup (distinct.filter _) unique).mpr
    intro w
    simp [(permutation hb).mem_iff,List.mem_finRange,inside]
  exact List.Perm.eq_of_pairwise (fun _ _ _ _ f b => False.elim (Nat.lt_asymm f b))
    (timeline.filter _) (chain.imp (extendsHB hb irrefl trans)) perm

/-- All worker-local instruction subsequences survive renumbering, including
    equal-word loads and interrupted calls. The return value is the original
    event ID, so the statement also preserves their individual identities. -/
theorem workerProjection (hb : Fin n → Fin n → Prop) [DecidableRel hb] (worker : Fin n → Nat)
    (irrefl : ∀ w, ¬ hb w w) (trans : ∀ {u v w}, hb u v → hb v w → hb u w)
    (program : ∀ u v, u < v → worker u = worker v → hb u v) (who : Nat) :
    (events hb).filter (fun i => worker i == who) = (List.finRange n).filter (fun i => worker i == who) := by
  let pick (i : Fin n) := worker i == who
  apply projection hb irrefl trans pick
  · intro w
    simp [pick,List.mem_finRange]
  · apply ((List.pairwise_lt_finRange n).filter pick).imp_of_mem
    intro a b ma mb lt
    exact program a b lt ((beq_iff_eq.mp (List.mem_filter.mp ma).2).trans (beq_iff_eq.mp (List.mem_filter.mp mb).2).symm)

-- A reader may precede an unrelated modification in this order while a
-- synchronized publisher numbered later in the original capture moves first.
private def forkHB (u v : Fin 4) : Prop := u.val = 3 ∧ (v.val = 0 ∨ v.val = 1)
private instance : DecidableRel forkHB := fun u v => inferInstanceAs (Decidable (u.val = 3 ∧ (v.val = 0 ∨ v.val = 1)))
example : (events forkHB).map Fin.val = [2,3,0,1] := by
  have range : List.finRange 4 = [0,1,2,3] := rfl
  simp [events,range,predecessors,forkHB,List.mergeSort,List.MergeSort.Internal.splitInTwo,List.merge]
  decide +kernel

end NeuralReversi.PvsParallel.Interleaving.EventOrder

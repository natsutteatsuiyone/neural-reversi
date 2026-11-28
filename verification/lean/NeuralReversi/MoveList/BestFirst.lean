module

@[expose] public section

namespace NeuralReversi.MoveList.BestFirst

variable {α : Type} (value : α → Int32)

def scan (best : α) (bestIdx next : Nat) : List α → Nat
  | [] => bestIdx
  | x :: xs =>
    if value x > value best then scan x next (next + 1) xs
    else scan best bestIdx (next + 1) xs

private theorem scan_eq (seen rest : List α) (h : seen ≠ []) :
    scan value (seen.maxOn value h) (seen.maxIdxOn value h) seen.length rest =
      (seen ++ rest).maxIdxOn value (by simp [h]) := by
  induction rest generalizing seen with
  | nil => simp [scan]
  | cons x xs ih =>
    have step := ih (seen ++ [x]) (by simp)
    rw [List.maxIdxOn_append h (by simp), List.maxOn_append h (by simp)] at step
    simp only [List.maxOn_singleton, List.maxIdxOn_singleton, Nat.add_zero, List.length_append,
      List.length_cons, List.length_nil, List.append_assoc, List.cons_append, List.nil_append] at step
    rw [scan]
    by_cases better : value x > value (seen.maxOn value h)
    · have cmp : ¬ value x ≤ value (seen.maxOn value h) := Int32.not_le.mpr better
      simpa [better, cmp, maxOn_eq_right cmp] using step
    · have cmp : value x ≤ value (seen.maxOn value h) := Int32.not_lt.mp better
      simpa [better, cmp, maxOn_eq_left cmp] using step

def bestIndex (first : α) (rest : List α) : Nat :=
  scan value first 0 1 rest

private theorem index_eq (first : α) (rest : List α) :
    bestIndex value first rest = (first :: rest).maxIdxOn value (by simp) :=
  by simpa [bestIndex] using scan_eq value [first] rest (by simp)

def takeBest (first : α) (rest : List α) : α × List α :=
  let xs := first :: rest
  let i := bestIndex value first rest
  have hi : i < xs.length := by
    dsimp [i, xs]
    rw [index_eq]
    exact List.maxIdxOn_lt_length _
  (xs[i], (xs.set i first).tail)

private theorem remainder_length (first : α) (rest : List α) :
    (takeBest value first rest).2.length = rest.length := by
  simp [takeBest]

private theorem step_perm (first : α) (rest : List α) :
    ((takeBest value first rest).1 :: (takeBest value first rest).2).Perm (first :: rest) := by
  have replace (i : Nat) (valid : i < (first :: rest).length) :
      ((first :: rest)[i] :: ((first :: rest).set i first).tail).Perm (first :: rest) := by
    have perm := List.set_set_perm valid (show 0 < (first :: rest).length by simp)
    cases i with
    | zero => simp
    | succ i => simpa using perm
  apply replace

def ordered : List α → List α
  | [] => []
  | first :: rest =>
    (takeBest value first rest).1 :: ordered (takeBest value first rest).2
termination_by xs => xs.length
decreasing_by simp [remainder_length]

private theorem scan_map {β : Type} (f : β → α) (best : β) (i next : Nat) (xs : List β) :
    scan value (f best) i next (xs.map f) = scan (value ∘ f) best i next xs := by
  induction xs generalizing best i next with
  | nil => rfl
  | cons x rest ih =>
    simp only [List.map_cons, scan, Function.comp_apply]
    split <;> apply ih

private theorem takeBest_map {β : Type} (f : β → α) (first : β) (rest : List β) :
    let step := takeBest (value ∘ f) first rest
    (f step.1, step.2.map f) = takeBest value (f first) (rest.map f) := by
  have index : bestIndex value (f first) (rest.map f) = bestIndex (value ∘ f) first rest :=
    scan_map value f first 0 1 rest
  dsimp only
  apply Prod.ext
  · simp only [takeBest]
    change f ((first :: rest)[bestIndex (value ∘ f) first rest]'_) =
      ((first :: rest).map f)[bestIndex value (f first) (rest.map f)]'_
    simp only [List.getElem_map, index]
  · simp only [takeBest, List.map_tail, List.map_set, List.map_cons, index]

theorem ordered_map {β : Type} (f : β → α) (xs : List β) :
    (ordered (value ∘ f) xs).map f = ordered value (xs.map f) := by
  induction xs using (ordered.induct (value ∘ f)) with
  | case1 => simp [ordered]
  | case2 first rest ih =>
    have step := takeBest_map value f first rest
    dsimp only at step
    rw [ordered.eq_2, List.map_cons, ih, List.map_cons, ordered.eq_2, ← step]

private theorem ordered_spec (xs : List α) :
    (ordered value xs).Perm xs ∧
    (ordered value xs).Pairwise (fun a b => value b ≤ value a) := by
  induction xs using (ordered.induct value) with
  | case1 => simp [ordered]
  | case2 first rest ih =>
    rw [ordered]
    obtain ⟨perm, sorted⟩ := ih
    have step := step_perm value first rest
    refine ⟨(perm.cons _).trans step, List.pairwise_cons.mpr ⟨?_, sorted⟩⟩
    intro x hx
    have member := step.mem_iff.mp (List.mem_cons_of_mem _ (perm.mem_iff.mp hx))
    simpa [takeBest, index_eq] using List.le_apply_maxOn_of_mem (f := value) member

def OrderContract : Prop :=
  ∀ (α : Type) (value : α → Int32),
    (∀ xs, (ordered value xs).Perm xs ∧
      (ordered value xs).Pairwise (fun a b => value b ≤ value a)) ∧
    ∀ first rest,
      let xs := first :: rest
      let i := bestIndex value first rest
      ∃ hi : i < xs.length, (takeBest value first rest).1 = xs[i] ∧
        (∀ j, (hj : j < xs.length) → value xs[j] ≤ value xs[i]) ∧
        ∀ j, (hj : j < i) → value xs[j] < value xs[i]

theorem orderContract_proven : OrderContract := by
  intro α value
  refine ⟨ordered_spec value, ?_⟩
  intro first rest
  dsimp only
  obtain ⟨hi, upper, tie⟩ := (List.maxIdxOn_eq_iff _).mp (index_eq value first rest).symm
  exact ⟨hi, rfl, upper, tie⟩

def RemainingContract : Prop :=
  ∀ (α : Type) (value : α → Int32) first rest cursor len,
    cursor + (first :: rest).length = len →
    cursor + 1 + (takeBest value first rest).2.length = len ∧ len - (cursor + 1) = rest.length

theorem remainingContract_proven : RemainingContract := by
  intro α value first rest cursor len partition
  rw [remainder_length]
  simp only [List.length_cons] at partition
  omega

example :
    (ordered Prod.snd ([(0, -1), (1, 1), (2, -1), (3, 1)] : List (Nat × Int32))).map Prod.fst =
      [1, 3, 2, 0] := by decide +kernel

end NeuralReversi.MoveList.BestFirst

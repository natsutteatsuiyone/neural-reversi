import NeuralReversi.EmptyListInit

namespace NeuralReversi.EmptyListRemoval
open EmptyListLinks EmptyListInit

private theorem keep_edge {nodes : Links} {sq : Square} {a b : Index}
    (linked : Linked nodes sq) (old : edge nodes a b)
    (aNe : a ≠ slot sq) (bNe : b ≠ slot sq) : edge (remove nodes sq) a b := by
  have sameNext : a ≠ (nodes (slot sq)).prev := by
    intro h
    exact bNe (old.1.symm.trans (h ▸ linked.fromPrev))
  have samePrev : b ≠ (nodes (slot sq)).next := by
    intro h
    exact aNe (old.2.symm.trans (h ▸ linked.fromNext))
  exact ⟨by rw [remove_next, ite_eq_right sameNext, old.1],
    by rw [remove_prev, ite_eq_right samePrev, old.2]⟩

private theorem keep_chain {nodes : Links} {sq : Square} (linked : Linked nodes sq)
    (xs : List Index) (last : Index) (path : chain nodes last xs)
    (lastNe : last ≠ slot sq) (absent : slot sq ∉ xs) : chain (remove nodes sq) last xs := by
  induction xs generalizing last with
  | nil => exact keep_edge linked path lastNe (slot_ne_head sq).symm
  | cons a tail ih =>
    have aNe : a ≠ slot sq := fun eq => absent (eq ▸ List.mem_cons_self)
    exact ⟨keep_edge linked path.1 lastNe aNe,
      ih a path.2 aNe (fun mem => absent (List.mem_cons_of_mem _ mem))⟩

private theorem join_edge {nodes : Links} {sq : Square} {a b : Index}
    (left : edge nodes a (slot sq)) (right : edge nodes (slot sq) b) :
    edge (remove nodes sq) a b := by
  exact ⟨by rw [remove_next, left.2, ite_eq_left rfl, right.1],
    by rw [remove_prev, right.1, ite_eq_left rfl, left.2]⟩

private theorem erase_chain {nodes : Links} {sq : Square} (linked : Linked nodes sq)
    (xs : List Index) (last : Index) (path : chain nodes last xs)
    (unique : xs.Nodup) (lastNe : last ≠ slot sq) :
    chain (remove nodes sq) last (xs.erase (slot sq)) := by
  induction xs generalizing last with
  | nil => exact keep_edge linked path lastNe (slot_ne_head sq).symm
  | cons a tail ih =>
    obtain ⟨absent, uniqueTail⟩ := List.nodup_cons.mp unique
    by_cases same : a = slot sq
    · subst a
      rw [List.erase_cons_head]
      cases tail with
      | nil => exact join_edge path.1 path.2
      | cons b rest =>
        have bNe : b ≠ slot sq := fun eq => absent (eq ▸ List.mem_cons_self)
        exact ⟨join_edge path.1 path.2.1,
          keep_chain linked rest b path.2.2 bNe (fun mem => absent (List.mem_cons_of_mem _ mem))⟩
    · rw [List.erase_cons_tail (by simpa using same)]
      exact ⟨keep_edge linked path.1 lastNe same, ih a path.2 uniqueTail same⟩

theorem chain_remove (nodes : Links) (xs : List Index) (sq : Square)
    (path : chain nodes head xs) (unique : (head :: xs).Nodup) (present : slot sq ∈ xs) :
    chain (remove nodes sq) head (xs.erase (slot sq)) := by
  have linked := linked_of_mem path unique (slot_ne_head sq) present
  exact erase_chain linked xs head path (List.nodup_cons.mp unique).2 (slot_ne_head sq).symm

def remaining (xs : List Index) (moves : List Square) : List Index :=
  xs.filter fun i => !(moves.map slot).contains i

private theorem remaining_cons (xs : List Index) (sq : Square) (rest : List Square)
    (unique : xs.Nodup) :
    remaining xs (sq :: rest) = remaining (xs.erase (slot sq)) rest := by
  unfold remaining
  rw [unique.erase_eq_filter, List.filter_filter]
  apply List.filter_congr
  intro i _
  by_cases same : i = slot sq <;> simp [same, Bool.and_comm]

private theorem remove_sequence (nodes : Links) (xs : List Index) (moves : List Square)
    (path : chain nodes head xs) (unique : (head :: xs).Nodup)
    (distinct : moves.Nodup) (available : ∀ sq ∈ moves, slot sq ∈ xs) :
    ValidRemovals nodes moves ∧ chain (moves.foldl remove nodes) head (remaining xs moves) := by
  induction moves generalizing nodes xs with
  | nil => simpa [ValidRemovals, remaining, List.filter_eq_self.mpr] using path
  | cons sq rest ih =>
    obtain ⟨notRepeated, tailDistinct⟩ := List.nodup_cons.mp distinct
    have present := available sq (by simp)
    have linked := linked_of_mem path unique (slot_ne_head sq) present
    have nextUnique : (head :: xs.erase (slot sq)).Nodup :=
      unique.sublist (List.Sublist.cons_cons _ List.erase_sublist)
    have tailAvailable (s : Square) (mem : s ∈ rest) : slot s ∈ xs.erase (slot sq) := by
      apply (List.mem_erase_of_ne _).mpr (available s (List.mem_cons_of_mem _ mem))
      intro eq
      have same : s = sq := Fin.ext (congrArg (fun i : Index => i.val) eq)
      exact notRepeated (same ▸ mem)
    obtain ⟨valid, finalPath⟩ := ih (remove nodes sq) _ (chain_remove nodes xs sq path unique present)
      nextUnique tailDistinct tailAvailable
    refine ⟨⟨linked, valid⟩, ?_⟩
    rw [remaining_cons xs sq rest (List.nodup_cons.mp unique).2]
    exact finalPath

def RemovalContract : Prop :=
  ∀ (empties : Bitboard.Bits) (moves : List Square), moves.Nodup →
    (∀ sq ∈ moves, empties.getLsbD sq.val = true) →
    let initial := newLinks empties
    let nodes := moves.foldl remove initial
    let order := remaining (scanOrder empties) moves
    ValidRemovals initial moves ∧ chain nodes head order ∧
      (∀ sq, slot sq ∈ order → Linked nodes sq) ∧
      moves.reverse.foldl restore nodes = initial

theorem removalContract_proven : RemovalContract := by
  intro empties moves distinct emptyAt
  have unique := scanOrder_unique empties
  obtain ⟨valid, path⟩ := remove_sequence (newLinks empties) (scanOrder empties) moves
    (initContract_proven empties).1 unique distinct (fun sq mem => (order_mem empties sq).mpr (emptyAt sq mem))
  have finalUnique : (head :: remaining (scanOrder empties) moves).Nodup :=
    unique.sublist (List.Sublist.cons_cons head List.filter_sublist)
  exact ⟨valid, path, fun sq => linked_of_mem path finalUnique (slot_ne_head sq),
    linksContract_proven.2 _ _ valid⟩


end NeuralReversi.EmptyListRemoval

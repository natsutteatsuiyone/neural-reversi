import NeuralReversi.Stability.EdgeModel

namespace NeuralReversi.Stability.Edge
set_option linter.unusedSimpArgs false

def Subset (a b : Byte) : Prop := a &&& b = a

theorem subset_refl (a : Byte) : Subset a a := by simp [Subset]
theorem subset_zero (a : Byte) : Subset 0 a := by simp [Subset]
theorem subset_trans {a b c : Byte} (ab : Subset a b) (bc : Subset b c) : Subset a c := by
  unfold Subset at *
  calc
    a &&& c = (a &&& b) &&& c := by rw [ab]
    _ = a &&& (b &&& c) := BitVec.and_assoc _ _ _
    _ = a := by rw [bc, ab]

theorem subset_and_left (a b : Byte) : Subset (a &&& b) a := by
  unfold Subset
  rw [BitVec.and_comm (a &&& b) a, ← BitVec.and_assoc, BitVec.and_self]

theorem subset_and_right (a b : Byte) : Subset (a &&& b) b := by
  simp [Subset, BitVec.and_assoc]

theorem findAux_subset (lookup : Byte → Byte → Byte) (p o stable : Byte) (xs : List (Fin 8)) :
    Subset (findAux lookup p o stable xs) stable := by
  induction xs generalizing stable with
  | nil => exact subset_refl stable
  | cons x xs ih =>
    simp only [findAux]
    split
    · exact subset_refl stable
    · split
      · split
        · exact subset_and_left _ _
        · split
          · exact subset_trans (subset_and_left _ _) (subset_and_left _ _)
          · exact subset_trans (ih _) (subset_trans (subset_and_left _ _) (subset_and_left _ _))
      · exact ih stable

theorem findAux_children (lookup : Byte → Byte → Byte) (p o stable : Byte)
    (xs : List (Fin 8)) (x : Fin 8) (member : x ∈ xs) (empty : emptyAt p o x = true) :
    Subset (findAux lookup p o stable xs) (lookup (firstChild p o x).1 (firstChild p o x).2) ∧
    Subset (findAux lookup p o stable xs) (lookup (secondChild p o x).1 (secondChild p o x).2) := by
  induction xs generalizing stable with
  | nil => simp at member
  | cons y ys ih =>
    by_cases hz : stable = 0
    · simp [findAux, hz, Subset]
    · simp only [findAux, beq_iff_eq, hz, ↓reduceIte]
      by_cases hy : emptyAt p o y = true
      · simp only [hy, ↓reduceIte]
        by_cases ha : stable &&& lookup (firstChild p o y).1 (firstChild p o y).2 = 0
        · simp only [ha, ↓reduceIte]
          exact ⟨subset_zero _, subset_zero _⟩
        · simp only [ha, ↓reduceIte]
          by_cases hb : stable &&& lookup (firstChild p o y).1 (firstChild p o y).2 &&&
              lookup (secondChild p o y).1 (secondChild p o y).2 = 0
          · simp only [hb, ↓reduceIte]
            exact ⟨subset_zero _, subset_zero _⟩
          · simp only [hb, ↓reduceIte]
            rcases List.mem_cons.mp member with hxy | hxs
            · subst y
              have hs := findAux_subset lookup p o
                (stable &&& lookup (firstChild p o x).1 (firstChild p o x).2 &&&
                  lookup (secondChild p o x).1 (secondChild p o x).2) ys
              exact ⟨subset_trans hs (subset_trans (subset_and_left _ _) (subset_and_right _ _)),
                subset_trans hs (subset_and_right _ _)⟩
            · exact ih _ hxs
      · simp only [hy, ↓reduceIte]
        have hxs : x ∈ ys := by
          rcases List.mem_cons.mp member with hxy | hxs
          · subst y; exact False.elim (hy empty)
          · exact hxs
        exact ih stable hxs

private theorem full_not_empty (p o : Byte) (x : Fin 8)
    (full : ~~~(p ||| o) = 0) : emptyAt p o x = false := by
  have h := congrArg (fun b : Byte => b.getLsbD x.val) full
  simp only [BitVec.getLsbD_not, BitVec.getLsbD_or, BitVec.getLsbD_zero] at h
  apply Bool.eq_false_iff.mpr
  intro he
  have e : (p ||| o) &&& bit8 x.val = 0 := by simpa only [emptyAt, beq_iff_eq] using he
  have eh := congrArg (fun b : Byte => b.getLsbD x.val) e
  simp [bit8, x.isLt, -BitVec.getLsbD_eq_getElem] at eh
  simp_all

theorem findWith_subset (lookup : Byte → Byte → Byte) (p o : Byte) :
    Subset (findWith lookup p o) p := by
  unfold findWith
  split
  · exact subset_refl p
  · exact findAux_subset _ _ _ _ _

theorem findWith_children (lookup : Byte → Byte → Byte) (p o : Byte) (x : Fin 8)
    (empty : emptyAt p o x = true) :
    Subset (findWith lookup p o) (lookup (firstChild p o x).1 (firstChild p o x).2) ∧
    Subset (findWith lookup p o) (lookup (secondChild p o x).1 (secondChild p o x).2) := by
  unfold findWith
  split
  · rename_i h
    simp only [Bool.or_eq_true, beq_iff_eq] at h
    rcases h with hp | hf
    · subst p; exact ⟨subset_zero _, subset_zero _⟩
    · have hn := full_not_empty p o x hf
      simp [empty] at hn
  · exact findAux_children lookup p o p _ x (by simp) empty

theorem findEdgeStable_contract (p o : Byte) (table : EdgeTable) :
    Subset (findEdgeStable p o table) p ∧
    ∀ x : Fin 8, emptyAt p o x = true →
      Subset (findEdgeStable p o table)
        (edgeStability table (firstChild p o x).1 (firstChild p o x).2) ∧
      Subset (findEdgeStable p o table)
        (edgeStability table (secondChild p o x).1 (secondChild p o x).2) := by
  exact ⟨findWith_subset _ _ _, fun x h => findWith_children _ _ _ x h⟩

theorem findAux_congr (f g : Byte → Byte → Byte) (p o stable : Byte) (xs : List (Fin 8))
    (agree : ∀ x : Fin 8, emptyAt p o x = true →
      f (firstChild p o x).1 (firstChild p o x).2 = g (firstChild p o x).1 (firstChild p o x).2 ∧
      f (secondChild p o x).1 (secondChild p o x).2 = g (secondChild p o x).1 (secondChild p o x).2) :
    findAux f p o stable xs = findAux g p o stable xs := by
  induction xs generalizing stable with
  | nil => rfl
  | cons x xs ih =>
    simp only [findAux]
    split
    · rfl
    · split
      · rename_i hx
        have h := agree x hx
        rw [h.1, h.2]
        split
        · rfl
        · split
          · rfl
          · exact ih _
      · exact ih _

theorem findWith_congr (f g : Byte → Byte → Byte) (p o : Byte)
    (agree : ∀ x : Fin 8, emptyAt p o x = true →
      f (firstChild p o x).1 (firstChild p o x).2 = g (firstChild p o x).1 (firstChild p o x).2 ∧
      f (secondChild p o x).1 (secondChild p o x).2 = g (secondChild p o x).1 (secondChild p o x).2) :
    findWith f p o = findWith g p o := by
  unfold findWith
  split
  · rfl
  · exact findAux_congr _ _ _ _ _ _ agree

theorem findEdgeStable_eq_next (fuel : Nat) (p o : Byte) (table : EdgeTable)
    (children : ∀ x : Fin 8, emptyAt p o x = true →
      edgeStability table (firstChild p o x).1 (firstChild p o x).2 =
        stableFuel fuel (firstChild p o x).1 (firstChild p o x).2 ∧
      edgeStability table (secondChild p o x).1 (secondChild p o x).2 =
        stableFuel fuel (secondChild p o x).1 (secondChild p o x).2) :
    findEdgeStable p o table = stableFuel (fuel + 1) p o :=
  findWith_congr _ _ _ _ children

end NeuralReversi.Stability.Edge

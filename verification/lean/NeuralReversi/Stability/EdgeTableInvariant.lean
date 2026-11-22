import NeuralReversi.Stability.EdgeContracts

namespace NeuralReversi.Stability.Edge
set_option linter.unusedSimpArgs false

theorem edgeIndex_lt (p o : Byte) : edgeIndex p o < 65536 := by
  have hp := p.isLt
  have ho := o.isLt
  simp only [edgeIndex]
  omega

theorem edgeIndex_injective (p o q r : Byte)
    (h : edgeIndex p o = edgeIndex q r) : p = q ∧ o = r := by
  have hp := p.isLt
  have ho := o.isLt
  have hq := q.isLt
  have hr := r.isLt
  simp only [edgeIndex] at h
  have hpq : p.toNat = q.toNat := by omega
  have hor : o.toNat = r.toNat := by omega
  exact ⟨BitVec.eq_of_toNat_eq hpq, BitVec.eq_of_toNat_eq hor⟩

def TableOwn (table : EdgeTable) : Prop :=
  ∀ p o : Byte, Subset (edgeStability table p o) p

def InvalidZero (table : EdgeTable) : Prop :=
  ∀ p o : Byte, p &&& o ≠ 0 → edgeStability table p o = 0

structure TableInvariant (table : EdgeTable) : Prop where
  size_eq : table.size = 65536
  own : TableOwn table
  invalid_zero : InvalidZero table

theorem edgeStability_set (table : EdgeTable) (p o q r value : Byte) :
    edgeStability (table.set! (edgeIndex p o) value) q r =
      if edgeIndex p o = edgeIndex q r then
        if edgeIndex p o < table.size then value else 0
      else edgeStability table q r := by
  simp only [edgeStability, Array.set!_eq_setIfInBounds, Array.getElem?_setIfInBounds]
  split
  · split <;> simp_all
  · simp_all

theorem tableOwn_set (table : EdgeTable) (p o value : Byte)
    (ht : TableOwn table) (hv : Subset value p) :
    TableOwn (table.set! (edgeIndex p o) value) := by
  intro q r
  rw [edgeStability_set]
  by_cases he : edgeIndex p o = edgeIndex q r
  · obtain ⟨rfl,rfl⟩ := edgeIndex_injective p o q r he
    simp only [↓reduceIte]
    split
    · exact hv
    · exact subset_zero _
  · simpa only [he, ↓reduceIte] using ht q r

theorem invalidZero_set (table : EdgeTable) (p o value : Byte)
    (ht : InvalidZero table) (hd : p &&& o = 0) :
    InvalidZero (table.set! (edgeIndex p o) value) := by
  intro q r hqr
  have he : edgeIndex p o ≠ edgeIndex q r := by
    intro he
    obtain ⟨rfl,rfl⟩ := edgeIndex_injective p o q r he
    exact hqr hd
  rw [edgeStability_set, ite_eq_right he]
  exact ht q r hqr

theorem tableInvariant_set (table : EdgeTable) (p o value : Byte)
    (ht : TableInvariant table) (hv : Subset value p) (hd : p &&& o = 0) :
    TableInvariant (table.set! (edgeIndex p o) value) := by
  exact ⟨by simpa using ht.size_eq, tableOwn_set _ _ _ _ ht.own hv,
    invalidZero_set _ _ _ _ ht.invalid_zero hd⟩

theorem subset_opponent_disjoint (m p : Byte) : p &&& (m &&& ~~~p) = 0 := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [BitVec.getLsbD_and, BitVec.getLsbD_not, BitVec.getLsbD_zero]
  cases hp : p.getLsbD i <;> simp [hp]

theorem fillSubsets_invariant (m p : Byte) (table : EdgeTable) (fuel : Nat)
    (ht : TableInvariant table) : TableInvariant (fillSubsets m p table fuel) := by
  induction fuel generalizing p table with
  | zero => exact ht
  | succ fuel ih =>
    have hs : TableInvariant
        (table.set! (edgeIndex p (m &&& ~~~p)) (findEdgeStable p (m &&& ~~~p) table)) :=
      tableInvariant_set _ _ _ _ ht (findEdgeStable_contract _ _ _).1
        (subset_opponent_disjoint m p)
    simp only [fillSubsets]
    split
    · exact hs
    · exact ih _ _ hs

theorem fillOccupancy_invariant (occupied : Nat) (table : EdgeTable)
    (ht : TableInvariant table) : TableInvariant (fillOccupancy occupied table) := by
  unfold fillOccupancy
  apply List.foldlRecOn
  · exact ht
  · intro t hinv n hn
    dsimp only
    split
    · exact fillSubsets_invariant _ _ _ _ hinv
    · exact hinv

theorem initialTable_invariant : TableInvariant (Array.replicate 65536 (0 : Byte)) := by
  refine ⟨by simp, ?_, ?_⟩
  · intro p o
    simp [edgeStability, Array.getElem?_replicate, edgeIndex_lt p o, Subset]
  · intro p o hd
    simp [edgeStability, Array.getElem?_replicate, edgeIndex_lt p o]

theorem initEdgeStability_invariant : TableInvariant initEdgeStability := by
  unfold initEdgeStability
  apply List.foldlRecOn
  · exact initialTable_invariant
  · intro table ht occupied hmem
    exact fillOccupancy_invariant occupied table ht

theorem initEdgeStability_size : initEdgeStability.size = 65536 :=
  initEdgeStability_invariant.size_eq

theorem initEdgeStability_own (p o : Byte) :
    Subset (edgeStability initEdgeStability p o) p := initEdgeStability_invariant.own p o

theorem initEdgeStability_invalid_zero (p o : Byte) (hd : p &&& o ≠ 0) :
    edgeStability initEdgeStability p o = 0 := initEdgeStability_invariant.invalid_zero p o hd

theorem initEdgeStability_index_safe (p o : Byte) :
    edgeIndex p o < initEdgeStability.size := by rw [initEdgeStability_size]; exact edgeIndex_lt p o


def checkedLookup (table : EdgeTable) (p o : Byte) (h : edgeIndex p o < table.size) : Byte :=
  table[edgeIndex p o]'h

theorem edgeStability_eq_checkedLookup (table : EdgeTable) (p o : Byte)
    (h : edgeIndex p o < table.size) :
    edgeStability table p o = checkedLookup table p o h := by
  simp only [edgeStability, checkedLookup, Array.getElem?_eq_getElem h, Option.getD_some]

theorem initEdgeStability_lookup (p o : Byte) :
    edgeStability initEdgeStability p o =
      checkedLookup initEdgeStability p o (initEdgeStability_index_safe p o) :=
  edgeStability_eq_checkedLookup initEdgeStability p o (initEdgeStability_index_safe p o)

end NeuralReversi.Stability.Edge

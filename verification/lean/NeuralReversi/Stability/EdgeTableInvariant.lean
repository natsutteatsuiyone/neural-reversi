import NeuralReversi.Stability.EdgeContracts

namespace NeuralReversi.Stability.Edge
set_option linter.unusedSimpArgs false

theorem edgeLookupIndex_lt (p occ : Byte) : edgeLookupIndex p occ < 65536 := by
  have hp := p.isLt
  have ho := occ.isLt
  simp only [edgeLookupIndex]
  omega

theorem edgeIndex_lt (p o : Byte) : edgeIndex p o < 65536 :=
  edgeLookupIndex_lt p (p ||| o)

theorem edgeLookupIndex_injective (p occ q occ' : Byte)
    (h : edgeLookupIndex p occ = edgeLookupIndex q occ') : p = q ∧ occ = occ' := by
  have hp := p.isLt
  have ho := occ.isLt
  have hq := q.isLt
  have ho' := occ'.isLt
  simp only [edgeLookupIndex] at h
  have hpq : p.toNat = q.toNat := by omega
  have hoo : occ.toNat = occ'.toNat := by omega
  exact ⟨BitVec.eq_of_toNat_eq hpq, BitVec.eq_of_toNat_eq hoo⟩

theorem edgeIndex_disjoint_injective (p o q r : Byte)
    (hp : p &&& o = 0) (hq : q &&& r = 0)
    (h : edgeIndex p o = edgeIndex q r) : p = q ∧ o = r := by
  obtain ⟨ep, eocc⟩ := edgeLookupIndex_injective p (p ||| o) q (q ||| r) h
  subst ep
  refine ⟨rfl, ?_⟩
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  have h1 := congrArg (fun b : Byte => b.getLsbD i) hp
  have h2 := congrArg (fun b : Byte => b.getLsbD i) hq
  have h3 := congrArg (fun b : Byte => b.getLsbD i) eocc
  simp only [BitVec.getLsbD_and, BitVec.getLsbD_or, BitVec.getLsbD_zero] at h1 h2 h3
  cases hp' : p.getLsbD i <;> cases ho' : o.getLsbD i <;> cases hr' : r.getLsbD i <;>
    simp [hp', ho', hr'] at h1 h2 h3 ⊢

def TableOwn (table : EdgeTable) : Prop :=
  ∀ p occ : Byte, Subset (edgeStabilityOcc table p occ) p

def InvalidZero (table : EdgeTable) : Prop :=
  ∀ p occ : Byte, p &&& ~~~occ ≠ 0 → edgeStabilityOcc table p occ = 0

structure TableInvariant (table : EdgeTable) : Prop where
  size_eq : table.size = 65536
  own : TableOwn table
  invalid_zero : InvalidZero table

theorem edgeStabilityOcc_set (table : EdgeTable) (p o q occ value : Byte) :
    edgeStabilityOcc (table.set! (edgeIndex p o) value) q occ =
      if edgeIndex p o = edgeLookupIndex q occ then
        if edgeIndex p o < table.size then value else 0
      else edgeStabilityOcc table q occ := by
  simp only [edgeStabilityOcc, Array.set!_eq_setIfInBounds, Array.getElem?_setIfInBounds]
  split
  · split <;> simp_all
  · simp_all

theorem edgeStability_set (table : EdgeTable) (p o q r value : Byte) :
    edgeStability (table.set! (edgeIndex p o) value) q r =
      if edgeIndex p o = edgeIndex q r then
        if edgeIndex p o < table.size then value else 0
      else edgeStability table q r :=
  edgeStabilityOcc_set table p o q (q ||| r) value

theorem tableOwn_set (table : EdgeTable) (p o value : Byte)
    (ht : TableOwn table) (hv : Subset value p) :
    TableOwn (table.set! (edgeIndex p o) value) := by
  intro q occ
  rw [edgeStabilityOcc_set]
  by_cases he : edgeIndex p o = edgeLookupIndex q occ
  · obtain ⟨rfl, rfl⟩ := edgeLookupIndex_injective p (p ||| o) q occ he
    simp only [he, ↓reduceIte]
    split
    · exact hv
    · exact subset_zero _
  · simpa only [he, ↓reduceIte] using ht q occ

theorem invalidZero_set (table : EdgeTable) (p o value : Byte)
    (ht : InvalidZero table) :
    InvalidZero (table.set! (edgeIndex p o) value) := by
  intro q occ hqr
  have he : edgeIndex p o ≠ edgeLookupIndex q occ := by
    intro he
    obtain ⟨rfl, rfl⟩ := edgeLookupIndex_injective p (p ||| o) q occ he
    apply hqr
    apply BitVec.eq_of_getLsbD_eq
    intro i hi
    simp only [BitVec.getLsbD_and, BitVec.getLsbD_not, BitVec.getLsbD_or, BitVec.getLsbD_zero]
    cases p.getLsbD i <;> cases o.getLsbD i <;> simp
  rw [edgeStabilityOcc_set, ite_eq_right he]
  exact ht q occ hqr

theorem tableInvariant_set (table : EdgeTable) (p o value : Byte)
    (ht : TableInvariant table) (hv : Subset value p) (_hd : p &&& o = 0) :
    TableInvariant (table.set! (edgeIndex p o) value) := by
  exact ⟨by simpa using ht.size_eq, tableOwn_set _ _ _ _ ht.own hv,
    invalidZero_set _ _ _ _ ht.invalid_zero⟩

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
  · intro p occ
    simp [edgeStabilityOcc, Array.getElem?_replicate, edgeLookupIndex_lt p occ, Subset]
  · intro p occ hd
    simp [edgeStabilityOcc, Array.getElem?_replicate, edgeLookupIndex_lt p occ]

theorem initEdgeStability_invariant : TableInvariant initEdgeStability := by
  unfold initEdgeStability
  apply List.foldlRecOn
  · exact initialTable_invariant
  · intro table ht occupied hmem
    exact fillOccupancy_invariant occupied table ht

theorem initEdgeStability_size : initEdgeStability.size = 65536 :=
  initEdgeStability_invariant.size_eq

theorem initEdgeStability_own (p o : Byte) :
    Subset (edgeStability initEdgeStability p o) p := initEdgeStability_invariant.own p (p ||| o)

theorem initEdgeStability_invalid_zero (p occ : Byte) (hd : p &&& ~~~occ ≠ 0) :
    edgeStabilityOcc initEdgeStability p occ = 0 := initEdgeStability_invariant.invalid_zero p occ hd

theorem initEdgeStability_index_safe (p o : Byte) :
    edgeIndex p o < initEdgeStability.size := by rw [initEdgeStability_size]; exact edgeIndex_lt p o

theorem initEdgeStability_lookup_safe (p occ : Byte) :
    edgeLookupIndex p occ < initEdgeStability.size := by
  rw [initEdgeStability_size]; exact edgeLookupIndex_lt p occ

def checkedLookup (table : EdgeTable) (p occ : Byte) (h : edgeLookupIndex p occ < table.size) : Byte :=
  table[edgeLookupIndex p occ]'h

theorem edgeStabilityOcc_eq_checkedLookup (table : EdgeTable) (p occ : Byte)
    (h : edgeLookupIndex p occ < table.size) :
    edgeStabilityOcc table p occ = checkedLookup table p occ h := by
  simp only [edgeStabilityOcc, checkedLookup, Array.getElem?_eq_getElem h, Option.getD_some]

theorem initEdgeStability_lookup (p occ : Byte) :
    edgeStabilityOcc initEdgeStability p occ =
      checkedLookup initEdgeStability p occ (initEdgeStability_lookup_safe p occ) :=
  edgeStabilityOcc_eq_checkedLookup initEdgeStability p occ (initEdgeStability_lookup_safe p occ)

end NeuralReversi.Stability.Edge

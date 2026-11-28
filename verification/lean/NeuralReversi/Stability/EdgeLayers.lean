module
public import NeuralReversi.Stability.EdgeEnumeration

@[expose] public section

namespace NeuralReversi.Stability.Edge

def updateState (table : EdgeTable) (st : State) : EdgeTable :=
  table.set! (edgeIndex st.1 st.2) (findEdgeStable st.1 st.2 table)

def splitMask (mask player : Byte) : State := (player, mask &&& ~~~player)

def maskStates (mask : Byte) : List State :=
  (sourceSubsets mask 0 256).map (splitMask mask)

def layerStates (occupied : Nat) : List State :=
  (List.range 256).flatMap fun n =>
    let mask := BitVec.ofNat 8 n
    if mask.cpop.toNat == occupied then maskStates mask else []

theorem fillSubsets_eq_fold (mask player : Byte) (table : EdgeTable) (fuel : Nat) :
    fillSubsets mask player table fuel =
      ((sourceSubsets mask player fuel).map (splitMask mask)).foldl updateState table := by
  induction fuel generalizing player table with
  | zero => rfl
  | succ fuel ih =>
    simp only [fillSubsets, sourceSubsets, List.map_cons, List.foldl_cons]
    split
    · rfl
    · exact ih _ _

theorem fillOccupancy_eq_fold (occupied : Nat) (table : EdgeTable) :
    fillOccupancy occupied table = (layerStates occupied).foldl updateState table := by
  unfold fillOccupancy layerStates
  rw [List.foldl_flatMap]
  congr 1
  funext table n
  dsimp only
  split
  · exact fillSubsets_eq_fold _ _ _ _
  · rfl

theorem sourceSubsets_mem (mask player : Byte) :
    player ∈ sourceSubsets mask 0 256 ↔ player &&& mask = player := by
  rw [source_subsets_exact]
  constructor
  · intro h
    obtain ⟨n, hn, he⟩ := List.mem_map.mp h
    have hf := (List.mem_filter.mp hn).2
    subst player
    exact beq_iff_eq.mp hf
  · intro h
    apply List.mem_map.mpr
    refine ⟨player.toNat, ?_, ?_⟩
    · apply List.mem_filter.mpr
      refine ⟨List.mem_range.mpr player.isLt, ?_⟩
      simpa only [BitVec.ofNat_toNat, BitVec.setWidth_eq, beq_iff_eq] using h
    · simp

theorem splitMask_disjoint (mask player : Byte) :
    (splitMask mask player).1 &&& (splitMask mask player).2 = 0 := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [splitMask, BitVec.getLsbD_and, BitVec.getLsbD_not]
  cases player.getLsbD i <;> cases mask.getLsbD i <;> simp

theorem splitMask_union (mask player : Byte) (subset : player &&& mask = player) :
    (splitMask mask player).1 ||| (splitMask mask player).2 = mask := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  have h := congrArg (fun b : Byte => b.getLsbD i) subset
  simp only [BitVec.getLsbD_and] at h
  simp only [splitMask, BitVec.getLsbD_or, BitVec.getLsbD_and, BitVec.getLsbD_not]
  cases hp : player.getLsbD i <;> cases hm : mask.getLsbD i <;> simp_all

theorem splitMask_eq (p o : Byte) (valid : p &&& o = 0) : splitMask (p ||| o) p = (p,o) := by
  refine Prod.ext rfl ?_
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  have h := congrArg (fun b : Byte => b.getLsbD i) valid
  simp only [BitVec.getLsbD_and] at h
  simp only [splitMask, BitVec.getLsbD_or, BitVec.getLsbD_and, BitVec.getLsbD_not]
  cases hp : p.getLsbD i <;> cases ho : o.getLsbD i <;> simp_all

theorem maskStates_properties (mask : Byte) (st : State) (member : st ∈ maskStates mask) :
    st.1 &&& st.2 = 0 ∧ st.1 ||| st.2 = mask := by
  obtain ⟨p, hp, rfl⟩ := List.mem_map.mp member
  exact ⟨splitMask_disjoint mask p, splitMask_union mask p ((sourceSubsets_mem mask p).mp hp)⟩

theorem layerStates_properties (occupied : Nat) (st : State) (member : st ∈ layerStates occupied) :
    st.1 &&& st.2 = 0 ∧ (st.1 ||| st.2).cpop.toNat = occupied := by
  obtain ⟨n, _, hn⟩ := List.mem_flatMap.mp member
  dsimp only at hn
  split at hn
  · rename_i h
    have prop := maskStates_properties (BitVec.ofNat 8 n) st hn
    exact ⟨prop.1, by rw [prop.2]; exact beq_iff_eq.mp h⟩
  · simp at hn

theorem layerStates_complete (p o : Byte) (valid : p &&& o = 0) :
    (p,o) ∈ layerStates (p ||| o).cpop.toNat := by
  apply List.mem_flatMap.mpr
  refine ⟨(p ||| o).toNat, List.mem_range.mpr (p ||| o).isLt, ?_⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq, beq_self_eq_true, ↓reduceIte]
  apply List.mem_map.mpr
  refine ⟨p, ?_, splitMask_eq p o valid⟩
  apply (sourceSubsets_mem (p ||| o) p).mpr
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [BitVec.getLsbD_and, BitVec.getLsbD_or]
  cases p.getLsbD i <;> cases o.getLsbD i <;> rfl

end NeuralReversi.Stability.Edge

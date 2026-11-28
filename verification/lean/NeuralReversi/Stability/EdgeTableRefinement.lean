module
public import NeuralReversi.Stability.EdgeTableInvariant
public import NeuralReversi.Stability.EdgeLayers
public import NeuralReversi.Stability.EdgeFixedPoint

@[expose] public section

namespace NeuralReversi.Stability.Edge
set_option linter.unusedSimpArgs false

def CorrectEntry (table : EdgeTable) (st : State) : Prop :=
  edgeStability table st.1 st.2 = recursiveStable st.1 st.2

def CorrectFrom (table : EdgeTable) (occupied : Nat) : Prop :=
  ∀ st : State, st.1 &&& st.2 = 0 → occupied ≤ (st.1 ||| st.2).cpop.toNat → CorrectEntry table st

theorem updateState_size (table : EdgeTable) (st : State) :
    (updateState table st).size = table.size := by simp [updateState]

theorem updateState_correct (table : EdgeTable) (st : State)
    (size : table.size = 65536)
    (value : findEdgeStable st.1 st.2 table = recursiveStable st.1 st.2) :
    CorrectEntry (updateState table st) st := by
  have hb : edgeIndex st.1 st.2 < table.size := by rw [size]; exact edgeIndex_lt _ _
  simp only [CorrectEntry, updateState, edgeStability_set, ↓reduceIte, hb]
  exact value

theorem updateState_preserves_correct (table : EdgeTable) (st : State)
    (size : table.size = 65536)
    (value : findEdgeStable st.1 st.2 table = recursiveStable st.1 st.2)
    (hstd : st.1 &&& st.2 = 0)
    (q : State) (hqd : q.1 &&& q.2 = 0) (correct : CorrectEntry table q) :
    CorrectEntry (updateState table st) q := by
  by_cases he : edgeIndex st.1 st.2 = edgeIndex q.1 q.2
  · have eq := edgeIndex_disjoint_injective st.1 st.2 q.1 q.2 hstd hqd he
    have hs : st = q := Prod.ext eq.1 eq.2
    subst q
    exact updateState_correct table st size value
  · simpa only [CorrectEntry, updateState, edgeStability_set, he, ↓reduceIte] using correct

theorem source_update_correct_from (table : EdgeTable) (k : Nat) (st : State)
    (valid : st.1 &&& st.2 = 0) (occupied : (st.1 ||| st.2).cpop.toNat = k)
    (higher : CorrectFrom table (k+1)) :
    findEdgeStable st.1 st.2 table = recursiveStable st.1 st.2 := by
  apply source_update_eq_fixedpoint _ _ _ valid
  intro x empty
  have hf := step_progress valid (show Step st (firstChild st.1 st.2 x) from ⟨x,empty,Or.inl rfl⟩)
  have ho := step_progress valid (show Step st (secondChild st.1 st.2 x) from ⟨x,empty,Or.inr rfl⟩)
  exact ⟨higher _ hf.1 (by omega), higher _ ho.1 (by omega)⟩

theorem layer_fold_contract (states : List State) (k : Nat) (table : EdgeTable)
    (size : table.size = 65536) (higher : CorrectFrom table (k+1))
    (members : ∀ st ∈ states, st.1 &&& st.2 = 0 ∧ (st.1 ||| st.2).cpop.toNat = k) :
    (states.foldl updateState table).size = 65536 ∧
    CorrectFrom (states.foldl updateState table) (k+1) ∧
    (∀ st ∈ states, CorrectEntry (states.foldl updateState table) st) ∧
    (∀ st, st.1 &&& st.2 = 0 → CorrectEntry table st → CorrectEntry (states.foldl updateState table) st) := by
  induction states generalizing table with
  | nil => exact ⟨size,higher,by simp,fun _ _ h => h⟩
  | cons st states ih =>
    have hst := members st (by simp)
    have value := source_update_correct_from table k st hst.1 hst.2 higher
    have size' : (updateState table st).size = 65536 := by rw [updateState_size, size]
    have preserve := updateState_preserves_correct table st size value hst.1
    have higher' : CorrectFrom (updateState table st) (k+1) := by
      intro q valid count
      exact preserve q valid (higher q valid count)
    have tail := ih (updateState table st) size' higher' (by
      intro q hq; exact members q (by simp [hq]))
    refine ⟨tail.1, tail.2.1, ?_, ?_⟩
    · intro q hq
      rcases List.mem_cons.mp hq with he | hm
      · subst q
        exact tail.2.2.2 st hst.1 (updateState_correct table st size value)
      · exact tail.2.2.1 q hm
    · intro q hqd hq
      exact tail.2.2.2 q hqd (preserve q hqd hq)

theorem fillOccupancy_correct_from (k : Nat) (table : EdgeTable)
    (size : table.size = 65536) (higher : CorrectFrom table (k+1)) :
    (fillOccupancy k table).size = 65536 ∧ CorrectFrom (fillOccupancy k table) k := by
  rw [fillOccupancy_eq_fold]
  have fold := layer_fold_contract (layerStates k) k table size higher
    (fun st hm => layerStates_properties k st hm)
  refine ⟨fold.1, ?_⟩
  intro st valid count
  by_cases he : (st.1 ||| st.2).cpop.toNat = k
  · apply fold.2.2.1 st
    rw [← he]
    exact layerStates_complete st.1 st.2 valid
  · exact fold.2.1 st valid (by omega)

theorem descending_layers_correct (n : Nat) (table : EdgeTable)
    (size : table.size = 65536) (higher : CorrectFrom table n) :
    (((List.range n).reverse).foldl (fun t k => fillOccupancy k t) table).size = 65536 ∧
    CorrectFrom (((List.range n).reverse).foldl (fun t k => fillOccupancy k t) table) 0 := by
  induction n generalizing table with
  | zero => exact ⟨size,higher⟩
  | succ n ih =>
    have layer := fillOccupancy_correct_from n table size higher
    simpa only [List.range_succ, List.reverse_append, List.reverse_cons, List.reverse_nil,
      List.nil_append, List.singleton_append, List.foldl_cons] using
      ih (fillOccupancy n table) layer.1 layer.2

theorem initEdgeStability_eq_recursive (p o : Byte) (valid : p &&& o = 0) :
    edgeStability initEdgeStability p o = recursiveStable p o := by
  have start : CorrectFrom (Array.replicate 65536 (0 : Byte)) 9 := by
    intro st hd hcount
    have bound := BitVec.toNat_cpop_le (st.1 ||| st.2)
    omega
  have all := descending_layers_correct 9 (Array.replicate 65536 (0 : Byte)) (by simp) start
  exact all.2 (p,o) valid (by omega)

end NeuralReversi.Stability.Edge

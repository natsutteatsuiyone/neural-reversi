import NeuralReversi.Stability.EdgeTableRefinement

namespace NeuralReversi.Stability.Edge

theorem edge_table_subset_player (p o : Byte) (valid : p &&& o = 0) :
    Subset (edgeStability initEdgeStability p o) p := by
  rw [initEdgeStability_eq_recursive p o valid]
  exact stableFuel_subset 8 p o

theorem edge_table_step_preserves {a b : State} (valid : a.1 &&& a.2 = 0) (step : Step a b) :
    Subset (edgeStability initEdgeStability a.1 a.2)
      (edgeStability initEdgeStability b.1 b.2) := by
  have next := step_progress valid step
  rw [initEdgeStability_eq_recursive a.1 a.2 valid,
    initEdgeStability_eq_recursive b.1 b.2 next.1]
  have one := stableFuel_step 8 step
  have fixed := stableFuel_sufficient 8 1 a.1 a.2 valid (by omega)
  simpa only [fixed, recursiveStable] using one

theorem edge_table_trace_safe {n : Nat} {a b : State}
    (valid : a.1 &&& a.2 = 0) (trace : Trace n a b) :
    Subset (edgeStability initEdgeStability a.1 a.2) b.1 := by
  rw [initEdgeStability_eq_recursive a.1 a.2 valid]
  exact recursiveStable_trace_safe valid trace

theorem edge_table_never_flips {n : Nat} {a b : State}
    (valid : a.1 &&& a.2 = 0) (trace : Trace n a b) (i : Fin 8)
    (marked : (edgeStability initEdgeStability a.1 a.2).getLsbD i.val = true) :
    b.1.getLsbD i.val = true := by
  rw [initEdgeStability_eq_recursive a.1 a.2 valid] at marked
  exact recursiveStable_never_flips valid trace i marked

end NeuralReversi.Stability.Edge

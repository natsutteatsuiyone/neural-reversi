module
public import NeuralReversi.Stability.EdgeMoveProperty

@[expose] public section

namespace NeuralReversi.Stability.Edge

theorem applyEdgeMove_correct (p o : Byte) (x : Fin 8) : MoveCorrect p o x :=
  moveCorrect_all p o x

theorem applyEdgeMove_progress (p o : Byte) (x : Fin 8) : MoveProgress p o x := by
  intro valid empty
  have h := applyEdgeMove_correct p o x valid empty
  exact ⟨h.1, h.2.1⟩

theorem applyEdgeMove_eq_coordinate_spec (p o : Byte) (x : Fin 8)
    (valid : p &&& o = 0) (empty : emptyAt p o x = true) :
    applyEdgeMove x p o = moveSpec p o x :=
  (applyEdgeMove_correct p o x valid empty).2.2

end NeuralReversi.Stability.Edge

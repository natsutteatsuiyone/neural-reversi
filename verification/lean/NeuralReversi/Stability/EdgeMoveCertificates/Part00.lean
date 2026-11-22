import NeuralReversi.Stability.EdgeMoveProperty
namespace NeuralReversi.Stability.Edge
set_option maxRecDepth 100000
set_option maxHeartbeats 0
theorem progress_0 : ∀ on : Fin 256, ∀ x : Fin 8,
    MoveCorrect 0 (BitVec.ofNat 8 on.val) x := by unfold MoveCorrect; decide +kernel
theorem progress_1 : ∀ on : Fin 256, ∀ x : Fin 8,
    MoveCorrect 1 (BitVec.ofNat 8 on.val) x := by unfold MoveCorrect; decide +kernel
theorem progress_2 : ∀ on : Fin 256, ∀ x : Fin 8,
    MoveCorrect 2 (BitVec.ofNat 8 on.val) x := by unfold MoveCorrect; decide +kernel
theorem progress_3 : ∀ on : Fin 256, ∀ x : Fin 8,
    MoveCorrect 3 (BitVec.ofNat 8 on.val) x := by unfold MoveCorrect; decide +kernel
theorem progress_4 : ∀ on : Fin 256, ∀ x : Fin 8,
    MoveCorrect 4 (BitVec.ofNat 8 on.val) x := by unfold MoveCorrect; decide +kernel
theorem progress_5 : ∀ on : Fin 256, ∀ x : Fin 8,
    MoveCorrect 5 (BitVec.ofNat 8 on.val) x := by unfold MoveCorrect; decide +kernel
theorem progress_6 : ∀ on : Fin 256, ∀ x : Fin 8,
    MoveCorrect 6 (BitVec.ofNat 8 on.val) x := by unfold MoveCorrect; decide +kernel
theorem progress_7 : ∀ on : Fin 256, ∀ x : Fin 8,
    MoveCorrect 7 (BitVec.ofNat 8 on.val) x := by unfold MoveCorrect; decide +kernel
theorem progress_8 : ∀ on : Fin 256, ∀ x : Fin 8,
    MoveCorrect 8 (BitVec.ofNat 8 on.val) x := by unfold MoveCorrect; decide +kernel
theorem progress_9 : ∀ on : Fin 256, ∀ x : Fin 8,
    MoveCorrect 9 (BitVec.ofNat 8 on.val) x := by unfold MoveCorrect; decide +kernel
theorem progress_10 : ∀ on : Fin 256, ∀ x : Fin 8,
    MoveCorrect 10 (BitVec.ofNat 8 on.val) x := by unfold MoveCorrect; decide +kernel
theorem progress_11 : ∀ on : Fin 256, ∀ x : Fin 8,
    MoveCorrect 11 (BitVec.ofNat 8 on.val) x := by unfold MoveCorrect; decide +kernel
theorem progress_12 : ∀ on : Fin 256, ∀ x : Fin 8,
    MoveCorrect 12 (BitVec.ofNat 8 on.val) x := by unfold MoveCorrect; decide +kernel
theorem progress_13 : ∀ on : Fin 256, ∀ x : Fin 8,
    MoveCorrect 13 (BitVec.ofNat 8 on.val) x := by unfold MoveCorrect; decide +kernel
theorem progress_14 : ∀ on : Fin 256, ∀ x : Fin 8,
    MoveCorrect 14 (BitVec.ofNat 8 on.val) x := by unfold MoveCorrect; decide +kernel
theorem progress_15 : ∀ on : Fin 256, ∀ x : Fin 8,
    MoveCorrect 15 (BitVec.ofNat 8 on.val) x := by unfold MoveCorrect; decide +kernel
end NeuralReversi.Stability.Edge

import NeuralReversi.Move.Spec

namespace NeuralReversi.Move

theorem create_spec (checked : Bool) (sq : Square.Square) (flipped : Bitboard.Bits) :
    ConstructorSpec checked sq flipped (create checked sq flipped) := by
  cases checked <;> by_cases empty : flipped = 0
  all_goals simp_all [create,ConstructorSpec]

theorem create_release_injective (sq1 sq2 : Square.Square) (f1 f2 : Bitboard.Bits)
    (same : create false sq1 f1 = create false sq2 f2) : sq1 = sq2 ∧ f1 = f2 := by
  change Except.ok (Move.mk f1 0 sq1) = Except.ok (Move.mk f2 0 sq2) at same
  cases same
  exact ⟨rfl,rfl⟩

theorem create_failure_text (checked : Bool) (sq : Square.Square)
    (flipped : Bitboard.Bits) (reason : Failure)
    (failed : create checked sq flipped = .error reason) :
    FailureTextSpec checked sq flipped reason.message := by
  have classified := create_spec checked sq flipped
  rw [failed] at classified
  cases reason with
  | emptyFlips => exact ⟨classified.1,classified.2,rfl⟩

def ConstructorContract : Prop :=
  (∀ checked sq flipped, ConstructorSpec checked sq flipped (create checked sq flipped)) ∧
  (∀ checked sq flipped reason, create checked sq flipped = .error reason →
    FailureTextSpec checked sq flipped reason.message) ∧
  (∀ sq1 sq2 f1 f2, create false sq1 f1 = create false sq2 f2 → sq1 = sq2 ∧ f1 = f2)

theorem constructorContract_proven : ConstructorContract :=
  ⟨create_spec,create_failure_text,create_release_injective⟩


end NeuralReversi.Move

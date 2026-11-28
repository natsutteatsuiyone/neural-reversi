module
public import NeuralReversi.Bitboard.ApplyMoveModel
public import NeuralReversi.Bitboard.SquareProperties

@[expose] public section

namespace NeuralReversi.Bitboard

theorem applyMoveAny_valid (debugAssertions overflowChecks : Bool) (bits flipped : Bits) (sq : Square) :
    applyMoveAny debugAssertions overflowChecks bits flipped sq =
      .ok (applyMove bits flipped sq) := by
  rw [applyMoveAny,(squareMaskContract_proven sq debugAssertions overflowChecks).2.1]
  rfl

def FullApplyMoveContract : Prop :=
  ∀ debugAssertions overflowChecks bits flipped (sq : Square), ∃ result,
    applyMoveAny debugAssertions overflowChecks bits flipped sq = .ok result ∧
    ∀ i, result.getLsbD i =
      Bool.xor (Bool.xor (bits.getLsbD i) (flipped.getLsbD i)) (decide (i = sq.val))

theorem fullApplyMoveContract_proven : FullApplyMoveContract := by
  intro debugAssertions overflowChecks bits flipped sq
  refine ⟨applyMove bits flipped sq,applyMoveAny_valid debugAssertions overflowChecks bits flipped sq,?_⟩
  intro i
  simp only [applyMove,BitVec.getLsbD_xor,squareMask_bit]

end NeuralReversi.Bitboard

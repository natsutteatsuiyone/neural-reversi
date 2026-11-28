module
public import NeuralReversi.Bitboard.Model

@[expose] public section

namespace NeuralReversi.Bitboard

def FlipCancellation : Prop :=
  ∀ bits flipped : Bits,
    applyFlip (applyFlip bits flipped) flipped = bits ∧
    (applyFlip bits flipped ^^^ bits) = flipped

theorem flipCancellation_proven : FlipCancellation := by
  intro bits flipped
  constructor
  · simp [applyFlip, BitVec.xor_assoc]
  · calc
      (applyFlip bits flipped ^^^ bits) = flipped ^^^ (bits ^^^ bits) := by
        unfold applyFlip
        ac_rfl
      _ = flipped := by simp

def MoveDelta : Prop :=
  ∀ (bits flipped : Bits) (sq : Square),
    (applyMove bits flipped sq ^^^ bits) = (flipped ^^^ squareMask sq) ∧
    applyMove (applyMove bits flipped sq) flipped sq = bits

theorem moveDelta_proven : MoveDelta := by
  intro bits flipped sq
  constructor
  · calc
      (applyMove bits flipped sq ^^^ bits) =
          (bits ^^^ bits) ^^^ (flipped ^^^ squareMask sq) := by
        unfold applyMove
        ac_rfl
      _ = flipped ^^^ squareMask sq := by simp
  · calc
      applyMove (applyMove bits flipped sq) flipped sq =
          bits ^^^ ((flipped ^^^ flipped) ^^^ (squareMask sq ^^^ squareMask sq)) := by
        unfold applyMove
        ac_rfl
      _ = bits := by simp

end NeuralReversi.Bitboard

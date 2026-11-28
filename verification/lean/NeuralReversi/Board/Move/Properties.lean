module
public import NeuralReversi.Board.Move.Model
public import NeuralReversi.Board.Move.Spec
public import NeuralReversi.Board.Properties
public import NeuralReversi.Bitboard.ApplyMoveProperties
public import NeuralReversi.Flip.Dispatch.Properties

@[expose] public section

namespace NeuralReversi.Board

theorem placeAny_valid_eq (debugAssertions overflowChecks : Bool) (b : Board)
    (flipped : Bitboard.Bits) (sq : Bitboard.Square) :
    placeAny debugAssertions overflowChecks b flipped sq = .ok (place b flipped sq) :=
  placeWithFlippedAny_valid debugAssertions overflowChecks b flipped sq

theorem place_coordinate (b : Board) (flipped : Bitboard.Bits) (sq : Bitboard.Square) :
    MoveSpec.Placement b flipped sq.val (.ok (place b flipped sq)) := by
  refine ⟨place b flipped sq, rfl, ?_⟩
  intro i
  by_cases hi : i < 64
  · constructor <;>
      simp only [place, BitVec.getLsbD_and, BitVec.getLsbD_or, BitVec.getLsbD_not,
        Bitboard.squareMask_bit, hi, decide_true, Bool.true_and]
  · have hge : 64 ≤ i := by omega
    have hne : i ≠ sq.val := by have := sq.isLt; omega
    constructor
    · simp only [place, BitVec.getLsbD_of_ge _ i hge, Bool.false_and]
    · simp only [place, BitVec.getLsbD_of_ge _ i hge, hne, decide_false, Bool.false_and, Bool.or_false]

theorem placeAny_valid (debugAssertions overflowChecks : Bool) (b : Board)
    (flipped : Bitboard.Bits) (sq : Bitboard.Square) :
    MoveSpec.Placement b flipped sq.val (placeAny debugAssertions overflowChecks b flipped sq) := by
  rw [placeAny_valid_eq]
  exact place_coordinate b flipped sq

def FullPlaceContract : Prop := ∀ debugAssertions overflowChecks b flipped sq,
  MoveSpec.PlaceOutcome debugAssertions overflowChecks b flipped sq
    (placeAny debugAssertions overflowChecks b flipped sq)

theorem fullPlaceContract_proven : FullPlaceContract := by
  intro debugAssertions overflowChecks b flipped sq
  exact placeAny_valid debugAssertions overflowChecks b flipped sq

theorem emptySquares_contains_eq (debugAssertions overflowChecks : Bool) (b : Board)
    (sq : Bitboard.Square) :
    Bitboard.containsSquare debugAssertions overflowChecks (emptySquares b) sq =
      .ok (!(b.player.getLsbD sq.val || b.opponent.getLsbD sq.val)) := by
  rw [Bitboard.squareMembershipContract_proven]
  simp only [emptySquares, BitVec.getLsbD_not, BitVec.getLsbD_or, sq.isLt, decide_true, Bool.true_and]

theorem makeMove_resolved (config : Flip.Dispatch.X86Config) (debugAssertions overflowChecks : Bool)
    (b : Board) (sq : Bitboard.AnySquare) :
    makeMove config debugAssertions overflowChecks b sq =
      if debugAssertions && decide (b.player.getLsbD sq.val = true ∨ b.opponent.getLsbD sq.val = true) then
        .error "sq must be an empty square"
      else if debugAssertions && decide (Flip.Dispatch.resultValue config sq b.player b.opponent = 0) then
        .error "assertion failed: !flipped.is_empty()"
      else placeAny debugAssertions overflowChecks b (Flip.Dispatch.resultValue config sq b.player b.opponent) sq := by
  unfold makeMove
  cases debugAssertions with
  | false =>
    simp only [Bool.false_and, Bool.false_eq_true, ↓reduceIte]
    rw [Flip.Dispatch.flip_eq_value]
    rfl
  | true =>
    simp only [Bool.true_and, ↓reduceIte]
    rw [emptySquares_contains_eq]
    cases hp : b.player.getLsbD sq.val <;> cases ho : b.opponent.getLsbD sq.val <;>
      simp only [Bool.or_true, Bool.or_false,
        Bool.not_true, Bool.not_false, Bool.false_eq_true, ↓reduceIte,
        or_self, or_true, true_or, decide_true, decide_false]
    · change (do
        let flipped ← Flip.Dispatch.flip config overflowChecks sq b.player b.opponent
        if Bitboard.isEmpty flipped then throw "assertion failed: !flipped.is_empty()"
        else placeAny true overflowChecks b flipped sq) = _
      rw [Flip.Dispatch.flip_eq_value]
      change (if Bitboard.isEmpty (Flip.Dispatch.resultValue config sq b.player b.opponent) then
        .error "assertion failed: !flipped.is_empty()"
      else placeAny true overflowChecks b (Flip.Dispatch.resultValue config sq b.player b.opponent) sq) = _
      simp [Bitboard.isEmpty]
    · rfl
    · rfl
    · rfl

def FullMakeMoveContract : Prop := ∀ config debugAssertions overflowChecks b sq,
  MoveSpec.MakeOutcome config debugAssertions overflowChecks b sq
    (makeMove config debugAssertions overflowChecks b sq)

theorem fullMakeMoveContract_proven : FullMakeMoveContract := by
  intro config debugAssertions overflowChecks b sq
  rw [makeMove_resolved]
  unfold MoveSpec.MakeOutcome
  split
  · rfl
  · refine ⟨Flip.Dispatch.resultValue config sq b.player b.opponent,
      Flip.Dispatch.value_raw_coordinate config sq b.player b.opponent, ?_⟩
    split
    · rfl
    · exact fullPlaceContract_proven debugAssertions overflowChecks b _ sq

end NeuralReversi.Board

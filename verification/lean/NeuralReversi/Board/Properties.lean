import NeuralReversi.Board.Model
import NeuralReversi.Bitboard.SquareProperties

namespace NeuralReversi.Board
open Bitboard

def PerspectiveExchange : Prop :=
  ∀ b : Board,
    (switchPlayers b).player = b.opponent ∧
    (switchPlayers b).opponent = b.player ∧
    switchPlayers (switchPlayers b) = b ∧
    emptySquares (switchPlayers b) = emptySquares b

theorem perspectiveExchange_proven : PerspectiveExchange := by
  intro b
  cases b
  simp [switchPlayers, emptySquares, BitVec.or_comm]

def EmptyPartition : Prop :=
  ∀ b : Board,
    emptySquares b &&& (b.player ||| b.opponent) = 0 ∧
    emptySquares b ||| (b.player ||| b.opponent) = ~~~(0 : Bits)

theorem emptyPartition_proven : EmptyPartition := by
  intro b
  simp [emptySquares]

def PlacementInvariant : Prop :=
  ∀ (b : Board) (flipped : Bits) (sq : Square),
    wellFormed b → flipped &&& ~~~b.opponent = 0 →
    squareMask sq &&& (b.player ||| b.opponent) = 0 →
    let result := place b flipped sq
    result.player = b.opponent &&& ~~~flipped ∧
    result.opponent = b.player ||| flipped ||| squareMask sq ∧
    wellFormed result ∧
    result.player ||| result.opponent = (b.player ||| b.opponent) ||| squareMask sq

theorem place_wellFormed (b : Board) (flipped : Bits) (sq : Square) (wf : wellFormed b) :
    wellFormed (place b flipped sq) := by
  simp only [wellFormed, place] at wf ⊢
  ext i hi
  have hd := congrArg (fun value : Bits => value[i]) wf
  simp only [BitVec.getElem_and, BitVec.getElem_or, BitVec.getElem_not] at *
  cases hp : b.player[i] <;> cases ho : b.opponent[i] <;>
    cases hflip : flipped[i] <;> cases hs : (squareMask sq)[i] <;> simp_all

theorem placementInvariant_proven : PlacementInvariant := by
  intro b flipped sq disjoint subset empty
  refine ⟨rfl, ?_, place_wellFormed b flipped sq disjoint, ?_⟩
  all_goals
    simp only [wellFormed, place] at disjoint ⊢
    ext i hi
    have hd := congrArg (fun value : Bits => value[i]) disjoint
    have hf := congrArg (fun value : Bits => value[i]) subset
    have he := congrArg (fun value : Bits => value[i]) empty
    simp only [BitVec.getElem_and, BitVec.getElem_or, BitVec.getElem_not] at *
    cases hp : b.player[i] <;> cases ho : b.opponent[i] <;>
      cases hflip : flipped[i] <;> cases hs : (squareMask sq)[i] <;> simp_all

def MovePreconditions (b : Board) (flipped : Bits) (sq : Square) : Prop :=
  flipped ≠ 0 ∧ emptySquares b &&& squareMask sq ≠ 0 ∧ flipped &&& ~~~b.opponent = 0

def PlayableGuardedMoveContract : Prop :=
  ∀ (b : Board) (flipped : Bits) (sq : Square),
    makeMoveWithFlipped false b flipped sq = .ok (place b flipped sq) ∧
    (makeMoveWithFlipped true b flipped sq = .ok (place b flipped sq) ↔
      MovePreconditions b flipped sq) ∧
    (flipped = 0 → makeMoveWithFlipped true b flipped sq = .error "flipped must not be empty") ∧
    (flipped ≠ 0 → emptySquares b &&& squareMask sq = 0 →
      makeMoveWithFlipped true b flipped sq = .error "sq must be an empty square") ∧
    (flipped ≠ 0 → emptySquares b &&& squareMask sq ≠ 0 → flipped &&& ~~~b.opponent ≠ 0 →
      makeMoveWithFlipped true b flipped sq = .error "flipped must be a subset of opponent's discs")

theorem playableGuardedMoveContract_proven : PlayableGuardedMoveContract := by
  intro b flipped sq
  by_cases hf : flipped = 0 <;>
    by_cases hs : emptySquares b &&& squareMask sq = 0 <;>
    by_cases ho : flipped &&& ~~~b.opponent = 0 <;>
    simp_all [makeMoveWithFlipped, MovePreconditions]

def PlayableCheckedPlacementInvariant : Prop :=
  ∀ (b : Board) (flipped : Bits) (sq : Square) (result : Board),
    wellFormed b → makeMoveWithFlipped true b flipped sq = .ok result →
    result.player = b.opponent &&& ~~~flipped ∧
    result.opponent = b.player ||| flipped ||| squareMask sq ∧
    wellFormed result ∧
    result.player ||| result.opponent = (b.player ||| b.opponent) ||| squareMask sq

theorem playableCheckedPlacementInvariant_proven : PlayableCheckedPlacementInvariant := by
  intro b flipped sq result disjoint accepted
  have resultEq : result = place b flipped sq := by
    by_cases hf : flipped = 0 <;>
      by_cases hs : emptySquares b &&& squareMask sq = 0 <;>
      by_cases ho : flipped &&& ~~~b.opponent = 0 <;>
      simp_all [makeMoveWithFlipped]
  have success : makeMoveWithFlipped true b flipped sq = .ok (place b flipped sq) := by
    simpa [resultEq] using accepted
  have pre := ((playableGuardedMoveContract_proven b flipped sq).2.1).mp success
  have empty : squareMask sq &&& (b.player ||| b.opponent) = 0#64 :=
    (empty_squareMask (b.player ||| b.opponent) sq).mp (by simpa [emptySquares] using pre.2.1)
  have invariant := placementInvariant_proven b flipped sq disjoint pre.2.2 empty
  simpa [resultEq] using invariant

theorem placeWithFlippedAny_valid (debugAssertions overflowChecks : Bool)
    (b : Board) (flipped : Bits) (sq : Square) :
    placeWithFlippedAny debugAssertions overflowChecks b flipped sq =
      .ok (place b flipped sq) := by
  rw [placeWithFlippedAny, (squareMaskContract_proven sq debugAssertions overflowChecks).2.1]
  rfl

theorem makeMoveWithFlippedAny_valid (debugAssertions overflowChecks : Bool)
    (b : Board) (flipped : Bits) (sq : Square) :
    makeMoveWithFlippedAny debugAssertions overflowChecks b flipped sq =
      makeMoveWithFlipped debugAssertions b flipped sq := by
  have mask := (squareMaskContract_proven sq debugAssertions overflowChecks).2.1
  unfold makeMoveWithFlippedAny
  simp only [containsSquare, mask, Except.map, placeWithFlippedAny_valid]
  cases debugAssertions <;>
    by_cases hf : flipped = 0 <;>
    by_cases hs : emptySquares b &&& squareMask sq = 0 <;>
    by_cases ho : flipped &&& ~~~b.opponent = 0 <;>
    simp_all [makeMoveWithFlipped]

def FullGuardedMoveContract : Prop :=
  ∀ debugAssertions overflowChecks b flipped (sq : Square),
    makeMoveWithFlippedAny debugAssertions overflowChecks b flipped sq =
      makeMoveWithFlipped debugAssertions b flipped sq

theorem fullGuardedMoveContract_proven : FullGuardedMoveContract :=
  makeMoveWithFlippedAny_valid

def GuardedMoveContract : Prop := PlayableGuardedMoveContract ∧ FullGuardedMoveContract

theorem guardedMoveContract_proven : GuardedMoveContract :=
  ⟨playableGuardedMoveContract_proven, fullGuardedMoveContract_proven⟩

def FullCheckedPlacementInvariant : Prop :=
  ∀ overflowChecks (b : Board) (flipped : Bits) (sq : AnySquare) (result : Board),
    wellFormed b → makeMoveWithFlippedAny true overflowChecks b flipped sq = .ok result →
    result.player = b.opponent &&& ~~~flipped ∧
    result.opponent = b.player ||| flipped ||| squareMask sq ∧
    wellFormed result ∧
    result.player ||| result.opponent = (b.player ||| b.opponent) ||| squareMask sq

theorem fullCheckedPlacementInvariant_proven : FullCheckedPlacementInvariant := by
  intro overflowChecks b flipped sq result disjoint accepted
  rw [makeMoveWithFlippedAny_valid] at accepted
  exact playableCheckedPlacementInvariant_proven b flipped sq result disjoint accepted

def CheckedPlacementInvariant : Prop :=
  PlayableCheckedPlacementInvariant ∧ FullCheckedPlacementInvariant

theorem checkedPlacementInvariant_proven : CheckedPlacementInvariant :=
  ⟨playableCheckedPlacementInvariant_proven, fullCheckedPlacementInvariant_proven⟩

end NeuralReversi.Board

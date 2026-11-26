import NeuralReversi.EndgameSearch.FourOrder
import NeuralReversi.Empties

namespace NeuralReversi.EndgameSearch

open PvsBoardTree (Position GeneratedSquare play pass)
open Bitboard.Moves (X86Config)

/-- An `Empties` value describes the empty squares of `b`. -/
def EmptyStateMatches (b : Position) (mask : BitVec 64) (count : BitVec 32) : Prop :=
  count = Board.emptyCount b.val ∧ Empties.Encodes mask (Board.emptySquares b.val)

theorem emptyState_initial (b : Position) :
    EmptyStateMatches b (Empties.encode (Board.emptySquares b.val)) (Board.emptyCount b.val) :=
  ⟨rfl, Empties.encodes_encode _⟩

private theorem child_empties (mc : X86Config) (fc : FlipConfig) (b : Position)
    (move : GeneratedSquare mc b.val) :
    Board.emptySquares (play mc fc b move).val.val =
      Board.emptySquares b.val &&& ~~~(Bitboard.squareMask move.val) := by
  let child := (play mc fc b move).val.val
  let flipped := Flip.Dispatch.resultValue fc move.val b.val.player b.val.opponent
  have accepted :=
    (Board.precomputedMoveContract_proven mc fc true true b.val move.val b.property move.property).1
  have placed := Board.fullCheckedPlacementInvariant_proven true b.val flipped move.val
    child b.property accepted
  change ~~~(child.player ||| child.opponent) = _
  rw [placed.2.2.2, BitVec.not_or]
  rfl

def EmptyStatePlayContract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (b : Position) (move : GeneratedSquare mc b.val)
    (mask : BitVec 64) (count : BitVec 32),
    EmptyStateMatches b mask count →
    let sq := move.val
    let next := (play mc fc b move).val
    0 < count.toNat ∧ EmptyStateMatches next (mask ^^^ Empties.bit sq) (count - 1) ∧
      (count.toNat ≤ 60 → (count - 1).toNat < 60) ∧
      (mask ^^^ Empties.bit sq) ^^^ Empties.bit sq = mask ∧ (count - 1) + 1 = count

theorem emptyStatePlayContract_proven : EmptyStatePlayContract := by
  intro mc fc b move mask count state
  let sq := move.val
  let next := (play mc fc b move).val
  obtain ⟨countEq, encoded⟩ := state
  have legal := (Board.getMoves_exact mc b.val sq.val).mp move.property
  have empty : (Board.emptySquares b.val).getLsbD sq.val = true := by
    simp [Board.emptySquares, sq.isLt, legal.2.1, legal.2.2.1, -BitVec.getLsbD_eq_getElem]
  have empties := child_empties mc fc b move
  change Board.emptySquares next.val = _ at empties
  have removed := Empties.encodes_without encoded sq empty
  rw [← empties] at removed
  have accepted :=
    (Board.precomputedMoveContract_proven mc fc true true b.val sq b.property move.property).1
  have step := (Board.checkedPlacementProgress_proven _ _ _ _ _ b.property accepted).2
  change (Board.emptyCount next.val).toNat + 1 = (Board.emptyCount b.val).toNat at step
  have nextCount : Board.emptyCount next.val + 1 = count := by
    apply BitVec.eq_of_toNat_eq
    rw [countEq, BitVec.toNat_add]
    change ((Board.emptyCount next.val).toNat + 1) % 4294967296 = _
    rw [step, Nat.mod_eq_of_lt (Board.emptyCount b.val).isLt]
  have decremented : count - 1 = Board.emptyCount next.val := by
    rw [← nextCount, BitVec.add_sub_cancel]
  have original := congrArg BitVec.toNat countEq
  refine ⟨by omega, ⟨decremented, removed⟩, ?_, by rw [BitVec.xor_assoc, BitVec.xor_self, BitVec.xor_zero],
    BitVec.sub_add_cancel _ _⟩
  intro fits
  rw [decremented]
  omega

/-- `node.odd_quadrants ^ quadrant(sq)` in `shallow_search_move` is the child's odd-quadrant mask. -/
theorem oddQuadrants_play (mc : X86Config) (fc : FlipConfig) (b : Position) (move : GeneratedSquare mc b.val) :
    Empties.oddQuadrants (Board.emptySquares b.val) ^^^ Empties.quadrant move.val =
      Empties.oddQuadrants (Board.emptySquares (play mc fc b move).val.val) := by
  have legal := (Board.getMoves_exact mc b.val move.val.val).mp move.property
  have empty : (Board.emptySquares b.val).getLsbD move.val.val = true := by
    simp [Board.emptySquares, move.val.isLt, legal.2.1, legal.2.2.1, -BitVec.getLsbD_eq_getElem]
  rw [child_empties mc fc b move]
  exact Empties.oddQuadrants_without _ _ empty

theorem emptyState_pass (b : Position) (mask : BitVec 64) (count : BitVec 32)
    (state : EmptyStateMatches b mask count) :
    EmptyStateMatches (pass b) mask count := by
  obtain ⟨counter, encoded⟩ := state
  have empties : Board.emptySquares (pass b).val = Board.emptySquares b.val := by
    change ~~~(b.val.opponent ||| b.val.player) = ~~~(b.val.player ||| b.val.opponent)
    rw [BitVec.or_comm]
  refine ⟨?_, ?_⟩
  · change count = Bitboard.count (Board.emptySquares (pass b).val)
    rw [empties]
    exact counter
  · rw [empties]
    exact encoded

theorem emptyState_four_order (b : Position) (mask : BitVec 64) (count : BitVec 32)
    (state : EmptyStateMatches b mask count) (four : count.toNat = 4) :
    Empties.sortLastFour ⟨mask, count⟩ (Empties.oddQuadrants (Board.emptySquares b.val)) =
      (presortedFourOrder b (state.1 ▸ four)).val.squares :=
  presortedFourOrder_reads b _ ⟨mask, count⟩ state.2

end NeuralReversi.EndgameSearch

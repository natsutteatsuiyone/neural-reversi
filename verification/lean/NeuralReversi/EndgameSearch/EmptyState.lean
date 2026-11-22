import NeuralReversi.EndgameSearch.FourOrder
import NeuralReversi.EmptyListRemoval

namespace NeuralReversi.EndgameSearch

open PvsBoardTree (Position GeneratedSquare play pass)
open Bitboard.Moves (X86Config)
open EmptyListLinks (Links Linked head slot remove restore)
open EmptyListInit (chain scanOrder newLinks)

def EmptyStateMatches (b : Position) (nodes : Links) (parity : BitVec 8) (count : BitVec 32) : Prop :=
  count = Board.emptyCount b.val ∧ chain nodes head (scanOrder (Board.emptySquares b.val)) ∧
    EmptyListParity.Reachable (Solve34.EmptyBoard.ofBoard b.val b.property).squares parity

theorem emptyState_initial (b : Position) :
    EmptyStateMatches b (newLinks (Board.emptySquares b.val))
      (EmptyListParity.fromEmpties (Board.emptySquares b.val)) (Board.emptyCount b.val) :=
  ⟨rfl, (EmptyListInit.initContract_proven (Board.emptySquares b.val)).1,
    .initial (Board.emptySquares b.val)⟩

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

private theorem erase_mask (xs : List Bitboard.Square) (distinct : xs.Nodup)
    (bits : Bitboard.Bits) (target : Bitboard.Square) :
    (xs.filter (fun sq => bits.getLsbD sq.val)).erase target =
      xs.filter (fun sq => (bits &&& ~~~(Bitboard.squareMask target)).getLsbD sq.val) := by
  rw [List.Nodup.erase_eq_filter (distinct.filter _) target, List.filter_filter]
  apply List.filter_congr
  intro sq _
  simp only [BitVec.getLsbD_and, BitVec.getLsbD_not, Bitboard.squareMask_bit,
    sq.isLt, decide_true, Bool.true_and]
  by_cases same : sq = target
  · subst sq
    simp
  · have val : sq.val ≠ target.val := fun eq => same (Fin.ext eq)
    simp [same, val]

private theorem erase_order (empties : Bitboard.Bits) (sq : Bitboard.Square) :
    (scanOrder empties).erase (slot sq) =
      scanOrder (empties &&& ~~~(Bitboard.squareMask sq)) := by
  have nodup := EmptyListParity.scan_perm.symm.nodup (List.nodup_finRange 64)
  have unique := (List.nodup_cons.mp (EmptyListInit.scanOrder_unique empties)).2
  rw [List.Nodup.erase_eq_filter unique]
  unfold scanOrder
  rw [List.filter_map, ← erase_mask _ nodup, List.Nodup.erase_eq_filter (nodup.filter _)]
  congr 1

def EmptyStatePlayContract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (b : Position) (move : GeneratedSquare mc b.val)
    (nodes : Links) (parity : BitVec 8) (count : BitVec 32),
    EmptyStateMatches b nodes parity count →
    let sq := move.val
    let next := (play mc fc b move).val
    let removed := remove nodes sq
    let toggled := EmptyListParity.toggle parity sq
    0 < count.toNat ∧ Linked nodes sq ∧ EmptyStateMatches next removed toggled (count - 1) ∧
      ¬ EmptyListLinks.contains removed sq ∧ (count.toNat ≤ 60 → (count - 1).toNat < 60) ∧
      restore removed sq = nodes ∧ EmptyListParity.toggle toggled sq = parity ∧
      (count - 1) + 1 = count

theorem emptyStatePlayContract_proven : EmptyStatePlayContract := by
  intro mc fc b move nodes parity count state
  let sq := move.val
  let next := (play mc fc b move).val
  obtain ⟨countEq, path, reachable⟩ := state
  have legal := (Board.getMoves_exact mc b.val sq.val).mp move.property
  have empty : (Board.emptySquares b.val).getLsbD sq.val = true := by
    simp [Board.emptySquares, sq.isLt, legal.2.1, legal.2.2.1, -BitVec.getLsbD_eq_getElem]
  have atSlot := (EmptyListInit.order_mem (Board.emptySquares b.val) sq).mpr empty
  have unique := EmptyListInit.scanOrder_unique (Board.emptySquares b.val)
  have linked := EmptyListInit.linked_of_mem path unique (EmptyListInit.slot_ne_head sq) atSlot
  have removed := EmptyListRemoval.chain_remove nodes _ sq path unique atSlot
  have empties := child_empties mc fc b move
  change Board.emptySquares next.val = _ at empties
  rw [erase_order, ← empties] at removed
  have present : sq ∈ (Solve34.EmptyBoard.ofBoard b.val b.property).squares :=
    List.mem_filter.mpr ⟨by simp, empty⟩
  have metadata := EmptyListParity.Reachable.remove sq reachable present
  have nextCells : (Solve34.EmptyBoard.ofBoard next.val next.property).squares =
      (Solve34.EmptyBoard.ofBoard b.val b.property).squares.erase sq := by
    change (List.finRange 64).filter (fun q => (Board.emptySquares next.val).getLsbD q.val) = _
    rw [empties]
    exact (erase_mask _ (List.nodup_finRange 64) (Board.emptySquares b.val) sq).symm
  rw [← nextCells] at metadata
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
  obtain ⟨_, _, absent, restored⟩ := EmptyListLinks.linksContract_proven.1 nodes sq linked
  refine ⟨by omega, linked, ⟨decremented, removed, metadata⟩, absent, ?_, restored,
    by simp [EmptyListParity.toggle, BitVec.xor_assoc], BitVec.sub_add_cancel _ _⟩
  intro fits
  rw [decremented]
  omega

theorem emptyState_pass (b : Position) (nodes : Links) (parity : BitVec 8) (count : BitVec 32)
    (state : EmptyStateMatches b nodes parity count) :
    EmptyStateMatches (pass b) nodes parity count := by
  obtain ⟨counter, path, metadata⟩ := state
  have empties : Board.emptySquares (pass b).val = Board.emptySquares b.val := by
    change ~~~(b.val.opponent ||| b.val.player) = ~~~(b.val.player ||| b.val.opponent)
    rw [BitVec.or_comm]
  refine ⟨?_, ?_, ?_⟩
  · change count = Bitboard.count (Board.emptySquares (pass b).val)
    rw [empties]
    exact counter
  · simpa only [empties] using path
  · change EmptyListParity.Reachable
      ((List.finRange 64).filter fun sq => (Board.emptySquares (pass b).val).getLsbD sq.val) parity
    rw [empties]
    exact metadata

theorem emptyState_parity (b : Position) (nodes : Links) (parity : BitVec 8) (count : BitVec 32)
    (state : EmptyStateMatches b nodes parity count) :
    parity.toNat = (boardParity b.val).val :=
  boardParity_eq_reachable _ _ _ state.2.2 (List.Perm.refl _)

theorem emptyState_four_order (b : Position) (nodes : Links) (parity : BitVec 8) (count : BitVec 32)
    (state : EmptyStateMatches b nodes parity count) (four : count.toNat = 4) :
    SortLastFour.sortLastFour nodes parity =
      some (linkedFourOrder b (state.1 ▸ four)).val.squares :=
  linkedFourOrder_reads b _ nodes parity _ state.2.1 state.2.2 (List.Perm.refl _)

end NeuralReversi.EndgameSearch

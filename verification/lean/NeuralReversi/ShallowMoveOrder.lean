module
public import NeuralReversi.Bitboard.SquareProperties
public import NeuralReversi.LegalMoves.Preservation

@[expose] public section

namespace NeuralReversi.ShallowMoveOrder
open Bitboard

def neighbors (empties : Bits) : Bits :=
  let horizontal := ((empties <<< 1) &&& 0xfefefefefefefefe) |||
    ((empties >>> 1) &&& 0x7f7f7f7f7f7f7f7f)
  let row := empties ||| horizontal
  horizontal ||| (row <<< 8) ||| (row >>> 8)

def adjacent (a b : Fin 64) : Bool :=
  decide (a ≠ b ∧ a.val % 8 ≤ b.val % 8 + 1 ∧ b.val % 8 ≤ a.val % 8 + 1 ∧
    a.val / 8 ≤ b.val / 8 + 1 ∧ b.val / 8 ≤ a.val / 8 + 1)

def HasEmptyNeighbor (empties : Bits) (sq : Fin 64) : Prop :=
  ∃ cell : Fin 64, empties.getLsbD cell.val = true ∧ adjacent cell sq = true

private theorem neighbors_or (a b : Bits) :
    neighbors (a ||| b) = neighbors a ||| neighbors b := by
  simp only [neighbors, BitVec.shiftLeft_or_distrib, BitVec.ushiftRight_or_distrib,
    BitVec.and_or_distrib_right]
  ac_rfl

set_option maxRecDepth 4096 in
set_option maxHeartbeats 2000000 in
private theorem neighbors_singleton : ∀ a b : Fin 64,
    (neighbors (squareMask a)).getLsbD b.val = adjacent a b := by decide

private def union (cells : List (Fin 64)) : Bits :=
  cells.foldr (fun cell bits => squareMask cell ||| bits) 0

private theorem union_bit (cells : List (Fin 64)) (sq : Fin 64) :
    (union cells).getLsbD sq.val = true ↔ sq ∈ cells := by
  induction cells with
  | nil => simp [union]
  | cons head tail ih =>
    change (squareMask head ||| union tail).getLsbD sq.val = true ↔ _
    simp [squareMask_bit, ih, Fin.ext_iff, -BitVec.getLsbD_eq_getElem]

private theorem neighbors_union (cells : List (Fin 64)) (sq : Fin 64) :
    (neighbors (union cells)).getLsbD sq.val = true ↔
      ∃ cell ∈ cells, adjacent cell sq = true := by
  induction cells with
  | nil => simp [union, neighbors]
  | cons head tail ih =>
    change (neighbors (squareMask head ||| union tail)).getLsbD sq.val = true ↔ _
    rw [neighbors_or, BitVec.getLsbD_or, Bool.or_eq_true, neighbors_singleton, ih]
    simp

theorem neighbors_exact (empties : Bits) (sq : Fin 64) :
    (neighbors empties).getLsbD sq.val = true ↔ HasEmptyNeighbor empties sq := by
  let cells := (List.finRange 64).filter (fun i => empties.getLsbD i.val)
  have same : union cells = empties := by
    apply BitVec.eq_of_getLsbD_eq
    intro i hi
    apply Bool.eq_iff_iff.mpr
    simpa [cells, -BitVec.getLsbD_eq_getElem] using union_bit cells ⟨i,hi⟩
  calc
    _ ↔ ∃ cell ∈ cells, adjacent cell sq = true := by rw [← same]; exact neighbors_union cells sq
    _ ↔ _ := by simp [HasEmptyNeighbor, cells]

def quadrantMask (parity : Fin 16) : Bits :=
  let table : Vector Bits 16 := #v[
    0x0000000000000000, 0x000000000F0F0F0F, 0x00000000F0F0F0F0, 0x00000000FFFFFFFF,
    0x0F0F0F0F00000000, 0x0F0F0F0F0F0F0F0F, 0x0F0F0F0FF0F0F0F0, 0x0F0F0F0FFFFFFFFF,
    0xF0F0F0F000000000, 0xF0F0F0F00F0F0F0F, 0xF0F0F0F0F0F0F0F0, 0xF0F0F0F0FFFFFFFF,
    0xFFFFFFFF00000000, 0xFFFFFFFF0F0F0F0F, 0xFFFFFFFFF0F0F0F0, 0xFFFFFFFFFFFFFFFF]
  table[parity]

def quadrant (sq : Fin 64) : Nat :=
  2 * (sq.val / 8 / 4) + sq.val % 8 / 4

theorem quadrantMask_bit : ∀ (parity : Fin 16) (sq : Fin 64),
    (quadrantMask parity).getLsbD sq.val = parity.val.testBit (quadrant sq) := by decide

def groups (moves : LegalMoves.Raw) (empties : Bits) (parity : Fin 16) : List LegalMoves.Raw :=
  let isolated := LegalMoves.bitand moves (~~~(neighbors empties))
  let rest := LegalMoves.andNot moves isolated.bits
  let odd := LegalMoves.bitand rest (quadrantMask parity)
  let even := LegalMoves.bitand rest (~~~(quadrantMask parity))
  [isolated, LegalMoves.corners odd, LegalMoves.nonCorners odd,
    LegalMoves.corners even, LegalMoves.nonCorners even]

def stagesFor (masks : List LegalMoves.Raw) (sq : Fin 64) : List Nat :=
  masks.zipIdx.filterMap fun (mask, stage) => if mask.bits.getLsbD sq.val then some stage else none

def PartitionContract : Prop := by
  classical
  exact ∀ (moves : LegalMoves.Raw) (empties : Bits) (parity : Fin 16) (sq : Fin 64),
    stagesFor (groups moves empties parity) sq =
      if !moves.bits.getLsbD sq.val then [] else
      if ¬ HasEmptyNeighbor empties sq then [0] else
      if parity.val.testBit (quadrant sq) then
        (if cornerCoordinate sq.val then [1] else [2])
      else (if cornerCoordinate sq.val then [3] else [4])

theorem partitionContract_proven : PartitionContract := by
  intro moves empties parity sq
  rw [← neighbors_exact]
  simp only [groups, stagesFor, List.zipIdx_cons, List.zipIdx_nil, List.filterMap_cons,
    List.filterMap_nil, LegalMoves.bitand, LegalMoves.andNot, LegalMoves.corners,
    LegalMoves.nonCorners, Bitboard.corners, Bitboard.nonCorners, BitVec.getLsbD_and,
    BitVec.getLsbD_not, sq.isLt, decide_true, Bool.true_and, quadrantMask_bit,
    cornerMask_bit _ sq.isLt]
  cases moves.bits.getLsbD sq.val <;> cases (neighbors empties).getLsbD sq.val <;>
    cases parity.val.testBit (quadrant sq) <;> cases cornerCoordinate sq.val <;> simp

theorem mem_groups_iff (moves : LegalMoves.Raw) (empties : Bits) (parity : Fin 16) (sq : Fin 64) :
    (∃ g ∈ groups moves empties parity, g.bits.getLsbD sq.val = true) ↔
      moves.bits.getLsbD sq.val = true := by
  classical
  let gs := groups moves empties parity
  have covered : stagesFor gs sq ≠ [] ↔ ∃ g ∈ gs, g.bits.getLsbD sq.val = true := by
    constructor
    · intro nonempty
      obtain ⟨_, hi⟩ := List.exists_mem_of_ne_nil _ nonempty
      obtain ⟨⟨g, i⟩, mem, hit⟩ := List.mem_filterMap.mp hi
      refine ⟨g, ?_, ?_⟩
      · simpa using (List.mem_map.mpr ⟨(g, i), mem, rfl⟩ : g ∈ gs.zipIdx.map Prod.fst)
      · split at hit <;> simp_all
    · rintro ⟨g, h, bit⟩
      rw [← List.zipIdx_map_fst 0 gs] at h
      obtain ⟨⟨_, i⟩, hi, rfl⟩ := List.mem_map.mp h
      apply List.ne_nil_of_mem (a := i)
      exact List.mem_filterMap.mpr ⟨_, hi, by simpa using bit⟩
  rw [← covered, show stagesFor gs sq = _ from partitionContract_proven moves empties parity sq]
  repeat
    first | split | simp_all

end NeuralReversi.ShallowMoveOrder

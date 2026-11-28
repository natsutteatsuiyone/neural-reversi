module
public import NeuralReversi.Bitboard.AdjacencyModel

@[expose] public section

namespace NeuralReversi.Bitboard

set_option maxRecDepth 4096
set_option maxHeartbeats 2000000

theorem neighbourMask_table :
    (List.range 64).all (fun square => (List.range 64).all (fun cell =>
      (neighbourMaskAt square).getLsbD cell == bracketableAdjacent square cell)) = true := by
  decide

theorem neighbourMask_bit (square cell : Nat) (squareBound : square < 64) (cellBound : cell < 64) :
    (neighbourMaskAt square).getLsbD cell = bracketableAdjacent square cell := by
  have row := List.all_eq_true.mp neighbourMask_table square (List.mem_range.mpr squareBound)
  have bit := List.all_eq_true.mp row cell (List.mem_range.mpr cellBound)
  exact beq_iff_eq.mp bit

theorem and_ne_zero_iff_cell (lhs rhs : Bits) :
    lhs &&& rhs ≠ 0#64 ↔ ∃ i : Nat, i < 64 ∧ lhs.getLsbD i = true ∧ rhs.getLsbD i = true := by
  constructor
  · intro nonzero
    have present := BitVec.getLsbD_true_ctz_of_ne_zero nonzero
    have within : (lhs &&& rhs).ctz.toNat < 64 := by
      by_cases bound : (lhs &&& rhs).ctz.toNat < 64
      · exact bound
      · have absent := BitVec.getLsbD_of_ge (lhs &&& rhs) (lhs &&& rhs).ctz.toNat (by omega)
        simp [absent] at present
    refine ⟨(lhs &&& rhs).ctz.toNat, within, ?_⟩
    simpa only [BitVec.getLsbD_and, Bool.and_eq_true] using present
  · rintro ⟨i, _, left, right⟩ zero
    have bit := congrArg (fun word : Bits => word.getLsbD i) zero
    simp [left, right] at bit

def AdjacentBitContract : Prop :=
  ∀ (bits : Bits) (sq : AnySquare),
    sq.val < neighbourMasks.size ∧
    (hasAdjacentBit bits sq = true ↔ ∃ i : Nat, i < 64 ∧
      bits.getLsbD i = true ∧ bracketableAdjacent sq.val i = true)

theorem adjacentBitContract_proven : AdjacentBitContract := by
  intro bits sq
  refine ⟨sq.isLt, ?_⟩
  simp only [hasAdjacentBit, bne_iff_ne]
  change (bits &&& neighbourMaskAt sq.val ≠ 0#64) ↔ _
  rw [and_ne_zero_iff_cell]
  constructor
  · rintro ⟨i, bound, present, selected⟩
    exact ⟨i, bound, present, by rwa [← neighbourMask_bit sq.val i sq.isLt bound]⟩
  · rintro ⟨i, bound, present, selected⟩
    exact ⟨i, bound, present, by rwa [neighbourMask_bit sq.val i sq.isLt bound]⟩

end NeuralReversi.Bitboard

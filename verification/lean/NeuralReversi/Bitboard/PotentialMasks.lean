import NeuralReversi.Bitboard.PotentialModel
import NeuralReversi.Bitboard.PotentialSpec
import NeuralReversi.Bitboard.AdjacencyProperties
import NeuralReversi.Bitboard.MovesDirectional
namespace NeuralReversi.Bitboard.Moves
set_option maxRecDepth 4096
set_option maxHeartbeats 2000000

def reaches (target cell : Nat) : Bool :=
  (horizontalMask.getLsbD cell && decide (target = cell+1)) ||
  (horizontalMask.getLsbD cell && decide (cell = target+1)) ||
  (verticalMask.getLsbD cell && decide (target = cell+8)) ||
  (verticalMask.getLsbD cell && decide (cell = target+8)) ||
  (diagonalMask.getLsbD cell && decide (target = cell+7)) ||
  (diagonalMask.getLsbD cell && decide (cell = target+7)) ||
  (diagonalMask.getLsbD cell && decide (target = cell+9)) ||
  (diagonalMask.getLsbD cell && decide (cell = target+9))

theorem reaches_table : ∀ target cell : Fin 64,
    reaches target.val cell.val = (neighbourMaskAt target.val).getLsbD cell.val := by decide

theorem masked_left_bit_exists (word mask : Bits) (d target : Nat) (bound : target < 64) :
    ((word &&& mask) <<< d).getLsbD target = true ↔
      ∃ cell, cell < 64 ∧ word.getLsbD cell = true ∧ mask.getLsbD cell = true ∧ target = cell+d := by
  rw [leftShift_true_iff _ d target bound,BitVec.getLsbD_and,Bool.and_eq_true]
  constructor
  · rintro ⟨dt,hw,hm⟩
    exact ⟨target-d,by omega,hw,hm,by omega⟩
  · rintro ⟨cell,cb,hw,hm,eq⟩
    have deq : target-d = cell := by omega
    exact ⟨by omega,by simpa only [deq] using hw,by simpa only [deq] using hm⟩

theorem masked_right_bit_exists (word mask : Bits) (d target : Nat) :
    ((word &&& mask) >>> d).getLsbD target = true ↔
      ∃ cell, cell < 64 ∧ word.getLsbD cell = true ∧ mask.getLsbD cell = true ∧ cell = target+d := by
  rw [BitVec.getLsbD_ushiftRight,BitVec.getLsbD_and,Bool.and_eq_true]
  constructor
  · rintro ⟨hw,hm⟩
    exact ⟨d+target,BitVec.lt_of_getLsbD hw,hw,hm,by omega⟩
  · rintro ⟨cell,cb,hw,hm,eq⟩
    have deq : d+target = cell := by omega
    exact ⟨by simpa only [deq] using hw,by simpa only [deq] using hm⟩

theorem candidate_reaches (opponent : Bits) (target : Nat) (bound : target < 64) :
    (potentialCandidates opponent).getLsbD target = true ↔
      ∃ cell, cell < 64 ∧ opponent.getLsbD cell = true ∧ reaches target cell = true := by
  simp only [potentialCandidates,BitVec.getLsbD_or,Bool.or_eq_true,
    masked_left_bit_exists _ _ _ _ bound,masked_right_bit_exists,reaches,Bool.and_eq_true,decide_eq_true_eq]
  simp only [and_or_left,exists_or]

theorem potentialCandidates_bit (opponent : Bits) (target : Nat) (bound : target < 64) :
    (potentialCandidates opponent).getLsbD target = true ↔ BracketableOpponent opponent target := by
  rw [candidate_reaches opponent target bound]
  constructor
  · rintro ⟨cell,cb,hw,hr⟩
    refine ⟨cell,cb,hw,?_⟩
    rw [reaches_table ⟨target,bound⟩ ⟨cell,cb⟩,neighbourMask_bit target cell (by omega) cb] at hr
    exact hr
  · rintro ⟨cell,cb,hw,hr⟩
    refine ⟨cell,cb,hw,?_⟩
    rw [reaches_table ⟨target,bound⟩ ⟨cell,cb⟩,neighbourMask_bit target cell (by omega) cb]
    exact hr

end NeuralReversi.Bitboard.Moves

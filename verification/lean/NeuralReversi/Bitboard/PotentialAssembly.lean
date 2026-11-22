import NeuralReversi.Bitboard.PotentialModel
import NeuralReversi.Bitboard.PotentialSpec

namespace NeuralReversi.Bitboard.Moves

theorem masked_potential_bit (player opponent candidates : Bits) (target : Nat)
    (bound : target < 64) :
    (candidates &&& ~~~(player ||| opponent)).getLsbD target = true ↔
      player.getLsbD target = false ∧ opponent.getLsbD target = false ∧
        candidates.getLsbD target = true := by
  simp only [BitVec.getLsbD_and, BitVec.getLsbD_not, BitVec.getLsbD_or,
    bound, decide_true, Bool.true_and]
  cases candidates.getLsbD target <;> cases player.getLsbD target <;>
    cases opponent.getLsbD target <;> decide

theorem potential_output_from_candidates (player opponent : Bits)
    (coordinates : ∀ target, target < 64 →
      ((potentialCandidates opponent).getLsbD target = true ↔
        BracketableOpponent opponent target)) (target : Nat) :
    (getMovesAndPotentialPortable player opponent).2.getLsbD target = true ↔
      PotentialMove player opponent target := by
  by_cases bound : target < 64
  · change (potentialCandidates opponent &&& ~~~(player ||| opponent)).getLsbD target = true ↔ _
    rw [masked_potential_bit player opponent _ target bound, coordinates target bound]
    simp only [PotentialMove, bound, true_and]
  · have zero := BitVec.getLsbD_of_ge
      (getMovesAndPotentialPortable player opponent).2 target (by omega)
    simp [zero, PotentialMove, bound]

end NeuralReversi.Bitboard.Moves

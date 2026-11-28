module
public import NeuralReversi.Bitboard.PotentialMasks
public import NeuralReversi.Bitboard.PotentialAssembly
public import NeuralReversi.Bitboard.PotentialSubset
public import NeuralReversi.Bitboard.MovesProperties

@[expose] public section

namespace NeuralReversi.Bitboard.Moves

def PortableMovesAndPotentialContract : Prop :=
  ∀ (player opponent : Bits) (target : Nat),
    let result := getMovesAndPotentialPortable player opponent
    (result.1.getLsbD target = true ↔ LegalMove player opponent target) ∧
    (result.2.getLsbD target = true ↔ PotentialMove player opponent target)

theorem portableMovesAndPotentialContract_proven : PortableMovesAndPotentialContract := by
  intro player opponent target
  exact ⟨getMovesPortable_exact player opponent target,
    potential_output_from_candidates player opponent (potentialCandidates_bit opponent) target⟩

theorem portable_legal_subset_potential (player opponent : Bits) :
    (getMovesAndPotentialPortable player opponent).1 &&&
      ~~~(getMovesAndPotentialPortable player opponent).2 = 0#64 := by
  apply BitVec.eq_of_getLsbD_eq
  intro target bound
  simp only [BitVec.getLsbD_and, BitVec.getLsbD_not, BitVec.getLsbD_zero,
    bound, decide_true, Bool.true_and]
  by_cases move : (getMovesAndPotentialPortable player opponent).1.getLsbD target = true
  · have exactBits := portableMovesAndPotentialContract_proven player opponent target
    have potential := exactBits.2.mpr (legalMove_implies_potentialMove player opponent target
      (exactBits.1.mp move))
    simp only [potential, Bool.not_true, Bool.and_false]
  · have empty : (getMovesAndPotentialPortable player opponent).1.getLsbD target = false :=
      Bool.eq_false_iff.mpr move
    simp only [empty, Bool.false_and]

end NeuralReversi.Bitboard.Moves

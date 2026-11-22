import NeuralReversi.Bitboard.Avx2PotentialModel
import NeuralReversi.Simd.PairProperties
import NeuralReversi.Bitboard.Avx2Properties
import NeuralReversi.Bitboard.PotentialProperties

namespace NeuralReversi.Bitboard.Moves
open NeuralReversi.Simd.X86

theorem combinedAvx2_decomposition (p o : Bits) :
    getMovesAndPotentialAvx2 p o = (getMovesAvx2 p o,potentialCandidates o &&& ~~~(p ||| o)) := by
  simp only [getMovesAndPotentialAvx2,horizontalOr2U64_exact]
  apply Prod.ext
  · simp only [getMovesAvx2,horizontalOrU64_exact]
  · simp only [or256,and256,sllvEpi64,srlvEpi64,setEpi64x,set1Epi64x]
    simp only [show (0:Fin 4).val=0 from rfl,show (1:Fin 4).val=1 from rfl,
      show (2:Fin 4).val=2 from rfl]
    simp only [show (1:Word).toNat=1 from rfl,show (8:Word).toNat=8 from rfl,
      show (9:Word).toNat=9 from rfl,show (7:Word).toNat=7 from rfl,Nat.reduceLT,↓reduceIte]
    simp only [potentialCandidates,BitVec.or_assoc,or_left_comm,BitVec.or_comm]

def Avx2MovesAndPotentialContract : Prop :=
  ∀ (p o : Bits) (target : Nat),
    let result := getMovesAndPotentialAvx2 p o
    (result.1.getLsbD target = true ↔ LegalMove p o target) ∧
    (result.2.getLsbD target = true ↔ PotentialMove p o target)
theorem avx2MovesAndPotentialContract_proven : Avx2MovesAndPotentialContract := by
  intro p o target
  rw [combinedAvx2_decomposition]
  exact ⟨getMovesAvx2_exact p o target,
    potential_output_from_candidates p o (potentialCandidates_bit o) target⟩

theorem combinedAvx2_eq_portable (p o : Bits) :
    getMovesAndPotentialAvx2 p o = getMovesAndPotentialPortable p o := by
  rw [combinedAvx2_decomposition,getMovesAvx2_eq_portable]
  rfl

theorem combinedAvx2_legal_subset_potential (p o : Bits) :
    (getMovesAndPotentialAvx2 p o).1 &&& ~~~(getMovesAndPotentialAvx2 p o).2 = 0#64 := by
  rw [combinedAvx2_eq_portable]
  exact portable_legal_subset_potential p o

end NeuralReversi.Bitboard.Moves

import NeuralReversi.Bitboard.Avx512PotentialModel
import NeuralReversi.Simd.ZmmProperties
import NeuralReversi.Bitboard.RotatePropagation
import NeuralReversi.Bitboard.Avx2PotentialProperties

namespace NeuralReversi.Bitboard.Moves
open NeuralReversi.Simd.X86

theorem combined512_lane_decomposition (p o : Bits) :
    getMovesAndPotentialAvx512 p o =
      ((rotateLane (o &&& horizontalMask) p 1 2 |||
        rotateLane (o &&& verticalMask) p 8 16 |||
        rotateLane (o &&& diagonalMask) p 9 18 |||
        rotateLane (o &&& diagonalMask) p 7 14 |||
        rotateLane (o &&& horizontalMask) p 63 62 |||
        rotateLane (o &&& verticalMask) p 56 48 |||
        rotateLane (o &&& diagonalMask) p 55 46 |||
        rotateLane (o &&& diagonalMask) p 57 50) &&& ~~~(p ||| o),
       ((o &&& horizontalMask).rotateLeft 1 |||
        (o &&& verticalMask).rotateLeft 8 |||
        (o &&& diagonalMask).rotateLeft 9 |||
        (o &&& diagonalMask).rotateLeft 7 |||
        (o &&& horizontalMask).rotateLeft 63 |||
        (o &&& verticalMask).rotateLeft 56 |||
        (o &&& diagonalMask).rotateLeft 55 |||
        (o &&& diagonalMask).rotateLeft 57) &&& ~~~(p ||| o)) := by
  rw [getMovesAndPotentialAvx512,horizontalOr2Zmm_exact]
  rfl


theorem combined512_decomposition (p o : Bits) :
    getMovesAndPotentialAvx512 p o =
      (avx2Directions p o &&& ~~~(p ||| o),potentialCandidates o &&& ~~~(p ||| o)) := by
  rcases sourceMask_gaps with ⟨g1,g8,g9,g7⟩
  rcases sourcePreMask_gaps with ⟨l1,l8,l9,l7,r1,r8,r9,r7⟩
  have cl1 := rotateLane_left o horizontalMask p 1 (by decide) (by decide) g1 l1
  have cl8 := rotateLane_left o verticalMask p 8 (by decide) (by decide) g8 l8
  have cl9 := rotateLane_left o diagonalMask p 9 (by decide) (by decide) g9 l9
  have cl7 := rotateLane_left o diagonalMask p 7 (by decide) (by decide) g7 l7
  have cr1 := rotateLane_right o horizontalMask p 1 (by decide) (by decide) (by decide) g1 r1
  have cr8 := rotateLane_right o verticalMask p 8 (by decide) (by decide) (by decide) g8 r8
  have cr9 := rotateLane_right o diagonalMask p 9 (by decide) (by decide) (by decide) g9 r9
  have cr7 := rotateLane_right o diagonalMask p 7 (by decide) (by decide) (by decide) g7 r7
  rw [combined512_lane_decomposition,cl1,cl8,cl9,cl7,cr1,cr8,cr9,cr7]
  rw [rol_negative_eq_ror _ 1 (by decide) (by decide),
      rol_negative_eq_ror _ 8 (by decide) (by decide),
      rol_negative_eq_ror _ 9 (by decide) (by decide),
      rol_negative_eq_ror _ 7 (by decide) (by decide)]
  rw [potential_rol_eq_shl o horizontalMask 1 (by decide) g1,
      potential_rol_eq_shl o verticalMask 8 (by decide) g8,
      potential_rol_eq_shl o diagonalMask 9 (by decide) g9,
      potential_rol_eq_shl o diagonalMask 7 (by decide) g7,
      potential_ror_eq_shr o horizontalMask 1 (by decide) g1,
      potential_ror_eq_shr o verticalMask 8 (by decide) g8,
      potential_ror_eq_shr o diagonalMask 9 (by decide) g9,
      potential_ror_eq_shr o diagonalMask 7 (by decide) g7]
  simp only [avx2Directions,potentialCandidates,BitVec.or_assoc,or_left_comm,BitVec.or_comm]


theorem combined512_eq_avx2 (p o : Bits) :
    getMovesAndPotentialAvx512 p o = getMovesAndPotentialAvx2 p o := by
  rw [combined512_decomposition,combinedAvx2_decomposition,avx2_decomposition]

def Avx512MovesAndPotentialContract : Prop :=
  ∀ (p o : Bits) (target : Nat),
    let result := getMovesAndPotentialAvx512 p o
    (result.1.getLsbD target = true ↔ LegalMove p o target) ∧
    (result.2.getLsbD target = true ↔ PotentialMove p o target)

theorem avx512MovesAndPotentialContract_proven : Avx512MovesAndPotentialContract := by
  intro p o target
  rw [combined512_eq_avx2]
  exact avx2MovesAndPotentialContract_proven p o target

theorem combined512_eq_portable (p o : Bits) :
    getMovesAndPotentialAvx512 p o = getMovesAndPotentialPortable p o :=
  (combined512_eq_avx2 p o).trans (combinedAvx2_eq_portable p o)

theorem combined512_legal_subset_potential (p o : Bits) :
    (getMovesAndPotentialAvx512 p o).1 &&& ~~~(getMovesAndPotentialAvx512 p o).2 = 0#64 := by
  rw [combined512_eq_avx2]
  exact combinedAvx2_legal_subset_potential p o


end NeuralReversi.Bitboard.Moves

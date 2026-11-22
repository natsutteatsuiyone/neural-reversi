import NeuralReversi.Simd.X86Properties
import NeuralReversi.Bitboard.Avx2Model
import NeuralReversi.Bitboard.MovesMaskGeometry

namespace NeuralReversi.Bitboard.Moves
open NeuralReversi.Simd.X86

def avx2Directions (p o : Bits) : Bits :=
  let h := o &&& horizontalMask
  let v := o &&& verticalMask
  let d := o &&& diagonalMask
  ((expandedLeft h p 1 <<< (1:Nat)) ||| (expandedRight h p 1 >>> (1:Nat))) |||
  ((expandedLeft v p 8 <<< (8:Nat)) ||| (expandedRight v p 8 >>> (8:Nat))) |||
  ((expandedLeft d p 9 <<< (9:Nat)) ||| (expandedRight d p 9 >>> (9:Nat))) |||
  ((expandedLeft d p 7 <<< (7:Nat)) ||| (expandedRight d p 7 >>> (7:Nat)))

theorem avx2_decomposition (p o : Bits) :
    getMovesAvx2 p o = avx2Directions p o &&& ~~~(p ||| o) := by
  simp only [getMovesAvx2,horizontalOrU64_exact]
  simp only [or256,and256,addEpi64,sllvEpi64,srlvEpi64,setEpi64x,set1Epi64x]
  simp only [show (0:Fin 4).val=0 from rfl,show (1:Fin 4).val=1 from rfl,
    show (2:Fin 4).val=2 from rfl]
  simp only [show (1:Word).toNat=1 from rfl,show (8:Word).toNat=8 from rfl,
    show (9:Word).toNat=9 from rfl,show (7:Word).toNat=7 from rfl,
    show ((1:Word)+1).toNat=2 from rfl,show ((8:Word)+8).toNat=16 from rfl,
    show ((9:Word)+9).toNat=18 from rfl,show ((7:Word)+7).toNat=14 from rfl,
    Nat.reduceLT,↓reduceIte]
  simp only [avx2Directions,expandedLeft,expandedRight,shifted_preMask]

end NeuralReversi.Bitboard.Moves

namespace NeuralReversi.Bitboard.Moves

theorem avx2Directions_iff (p o : Bits) (i : Nat) (ib : i < 64) :
    (avx2Directions p o).getLsbD i = true ↔ Bracketed p o i := by
  have l1 : (expandedLeft (o &&& horizontalMask) p 1 <<< (1:Nat)).getLsbD i = true ↔ Ray p o i 0 1 := by
    simpa only [rayMask,Int.reduceEq,↓reduceIte] using
      left_direction_exact p o i 1 0 1 ib (by decide) (by decide) (by decide)
  have r1 : (expandedRight (o &&& horizontalMask) p 1 >>> (1:Nat)).getLsbD i = true ↔ Ray p o i 0 (-1) := by
    simpa only [rayMask,Int.reduceNeg,Int.reduceEq,↓reduceIte] using
      right_direction_exact p o i 1 0 (-1) (by decide) (by decide) (by decide)
  have l8 : (expandedLeft (o &&& verticalMask) p 8 <<< (8:Nat)).getLsbD i = true ↔ Ray p o i 1 0 := by
    simpa only [rayMask,↓reduceIte] using
      rowMasked_left_direction_exact p o i 8 1 0 ib (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have r8 : (expandedRight (o &&& verticalMask) p 8 >>> (8:Nat)).getLsbD i = true ↔ Ray p o i (-1) 0 := by
    simpa only [rayMask,↓reduceIte] using
      rowMasked_right_direction_exact p o i 8 (-1) 0 ib (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have l9 : (expandedLeft (o &&& diagonalMask) p 9 <<< (9:Nat)).getLsbD i = true ↔ Ray p o i 1 1 := by
    simpa only [rowMasked_diagonal o 1 (by decide)] using
      rowMasked_left_direction_exact p o i 9 1 1 ib (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have r9 : (expandedRight (o &&& diagonalMask) p 9 >>> (9:Nat)).getLsbD i = true ↔ Ray p o i (-1) (-1) := by
    simpa only [rowMasked_diagonal o (-1) (by decide)] using
      rowMasked_right_direction_exact p o i 9 (-1) (-1) ib (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have l7 : (expandedLeft (o &&& diagonalMask) p 7 <<< (7:Nat)).getLsbD i = true ↔ Ray p o i 1 (-1) := by
    simpa only [rowMasked_diagonal o (-1) (by decide)] using
      rowMasked_left_direction_exact p o i 7 1 (-1) ib (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have r7 : (expandedRight (o &&& diagonalMask) p 7 >>> (7:Nat)).getLsbD i = true ↔ Ray p o i (-1) 1 := by
    simpa only [rowMasked_diagonal o 1 (by decide)] using
      rowMasked_right_direction_exact p o i 7 (-1) 1 ib (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  simp only [avx2Directions,BitVec.getLsbD_or,Bool.or_eq_true]
  rw [l1,r1,l8,r8,l9,r9,l7,r7]
  simp only [Bracketed,_root_.or_assoc,_root_.or_left_comm,_root_.or_comm]

theorem getMovesAvx2_exact (p o : Bits) (i : Nat) :
    (getMovesAvx2 p o).getLsbD i = true ↔ LegalMove p o i := by
  by_cases ib : i < 64
  · by_cases empty : (p ||| o).getLsbD i = false
    · have zeros := empty
      simp only [BitVec.getLsbD_or,Bool.or_eq_false_iff] at zeros
      rw [avx2_decomposition,BitVec.getLsbD_and,BitVec.getLsbD_not]
      simp only [ib,decide_true,empty,Bool.not_false,Bool.and_true]
      rw [avx2Directions_iff p o i ib]
      simp only [LegalMove,ib,zeros.1,zeros.2,true_and]
    · have full : (p ||| o).getLsbD i = true := by cases h : (p ||| o).getLsbD i <;> simp_all
      rw [avx2_decomposition,BitVec.getLsbD_and,BitVec.getLsbD_not]
      simp only [full,Bool.not_true,Bool.and_false,Bool.false_eq_true,false_iff]
      intro legal
      have contradict : (p ||| o).getLsbD i = false := by
        simp only [BitVec.getLsbD_or,legal.2.1,legal.2.2.1,Bool.false_or]
      exact empty contradict
  · have zero := BitVec.getLsbD_of_ge (getMovesAvx2 p o) i (by omega)
    simp [zero,LegalMove,ib]

theorem getMovesAvx2_eq_portable (p o : Bits) : getMovesAvx2 p o = getMovesPortable p o := by
  apply BitVec.eq_of_getLsbD_eq
  intro i _
  have same := (getMovesAvx2_exact p o i).trans (getMovesPortable_exact p o i).symm
  cases h₁ : (getMovesAvx2 p o).getLsbD i <;>
    cases h₂ : (getMovesPortable p o).getLsbD i <;> simp_all

def Avx2MovesContract : Prop :=
  ∀ p o i, (getMovesAvx2 p o).getLsbD i = true ↔ LegalMove p o i

theorem avx2MovesContract_proven : Avx2MovesContract := by
  intro p o i
  exact getMovesAvx2_exact p o i

end NeuralReversi.Bitboard.Moves

import NeuralReversi.Bitboard.MovesGeometry
import NeuralReversi.Bitboard.PotentialModel

namespace NeuralReversi.Bitboard.Moves

theorem verticalMask_bits : ∀ i : Fin 64,
    verticalMask.getLsbD i.val = decide (0 < i.val / 8 ∧ i.val / 8 < 7) := by decide

theorem verticalOpponent_bit (o : Bits) (i : Nat) (hi : i < 64) :
    (o &&& verticalMask).getLsbD i = true ↔
      o.getLsbD i = true ∧ 0 < i/8 ∧ i/8 < 7 := by
  rw [BitVec.getLsbD_and,verticalMask_bits ⟨i,hi⟩]
  simp only [Bool.and_eq_true,decide_eq_true_eq]

theorem OnRay_interior_rank (anchor target index distance k : Nat) (dr dc : Int)
    (ab : anchor < 64) (tb : target < 64)
    (drLow : -1 ≤ dr) (drHigh : dr ≤ 1) (drNonzero : dr ≠ 0)
    (kp : 0 < k) (kd : k < distance)
    (last : OnRay anchor target distance dr dc) (middle : OnRay anchor index k dr dc) :
    0 < index/8 ∧ index/8 < 7 := by
  have hcases : dr = -1 ∨ dr = 1 := by omega
  rcases hcases with rfl|rfl
  all_goals simp only [OnRay,Int.mul_neg,Int.mul_one] at last middle
  all_goals omega

theorem Ray_verticalMask_iff (p o : Bits) (target : Nat) (dr dc : Int)
    (tb : target < 64) (drLow : -1 ≤ dr) (drHigh : dr ≤ 1) (drNonzero : dr ≠ 0) :
    Ray p (o &&& verticalMask) target dr dc ↔ Ray p o target dr dc := by
  constructor
  · rintro ⟨anchor,distance,ab,low,high,hp,last,chain⟩
    refine ⟨anchor,distance,ab,low,high,hp,last,?_⟩
    intro k kp kd
    obtain ⟨index,ib,middle,ho⟩ := chain k kp kd
    exact ⟨index,ib,middle,((verticalOpponent_bit o index ib).mp ho).1⟩
  · rintro ⟨anchor,distance,ab,low,high,hp,last,chain⟩
    refine ⟨anchor,distance,ab,low,high,hp,last,?_⟩
    intro k kp kd
    obtain ⟨index,ib,middle,ho⟩ := chain k kp kd
    exact ⟨index,ib,middle,(verticalOpponent_bit o index ib).mpr
      ⟨ho,OnRay_interior_rank anchor target index distance k dr dc ab tb drLow drHigh drNonzero kp kd last middle⟩⟩

theorem rowMasked_left_direction_exact (p o : Bits) (i d : Nat) (dr dc : Int)
    (ib : i < 64) (drLow : -1 ≤ dr) (drHigh : dr ≤ 1) (drNonzero : dr ≠ 0)
    (dcLow : -1 ≤ dc) (dcHigh : dc ≤ 1) (strideEq : (d : Int) = 8*dr+dc) :
    (expandedLeft (rayMask (o &&& verticalMask) dc) p d <<< d).getLsbD i = true ↔ Ray p o i dr dc := by
  exact (left_direction_exact p (o &&& verticalMask) i d dr dc ib dcLow dcHigh strideEq).trans
    (Ray_verticalMask_iff p o i dr dc ib drLow drHigh drNonzero)

theorem rowMasked_right_direction_exact (p o : Bits) (i d : Nat) (dr dc : Int)
    (ib : i < 64) (drLow : -1 ≤ dr) (drHigh : dr ≤ 1) (drNonzero : dr ≠ 0)
    (dcLow : -1 ≤ dc) (dcHigh : dc ≤ 1) (strideEq : (d : Int) = -(8*dr+dc)) :
    (expandedRight (rayMask (o &&& verticalMask) dc) p d >>> d).getLsbD i = true ↔ Ray p o i dr dc := by
  exact (right_direction_exact p (o &&& verticalMask) i d dr dc dcLow dcHigh strideEq).trans
    (Ray_verticalMask_iff p o i dr dc ib drLow drHigh drNonzero)

theorem vertical_horizontal_eq_diagonal : verticalMask &&& horizontalMask = diagonalMask := by decide

theorem rowMasked_diagonal (o : Bits) (dc : Int) (nonzero : dc ≠ 0) :
    rayMask (o &&& verticalMask) dc = o &&& diagonalMask := by
  simp only [rayMask,nonzero,↓reduceIte,BitVec.and_assoc,vertical_horizontal_eq_diagonal]

end NeuralReversi.Bitboard.Moves

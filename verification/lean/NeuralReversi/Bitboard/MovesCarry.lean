module
public import NeuralReversi.Bitboard.MovesModel
public import NeuralReversi.Bitboard.MovesSpec
public import Init.Data.BitVec.Bitblast

@[expose] public section

namespace NeuralReversi.Bitboard.Moves
set_option maxRecDepth 4096
set_option maxHeartbeats 2000000

def CarryRun (h seed : BitVec w) (i : Nat) : Prop :=
  ∃ j, j < i ∧ h.getLsbD j = true ∧ seed.getLsbD j = true ∧
    ∀ k, j < k → k < i → h.getLsbD k = true

theorem masked_carry_step (h seed : BitVec w) (i : Nat) :
    BitVec.carry (i+1) h (h &&& seed) false =
      (h.getLsbD i && (seed.getLsbD i || BitVec.carry i h (h &&& seed) false)) := by
  rw [BitVec.carry_succ,BitVec.getLsbD_and]
  cases h.getLsbD i <;> cases seed.getLsbD i <;>
    cases BitVec.carry i h (h &&& seed) false <;> rfl

theorem masked_carry_iff (h seed : BitVec w) (i : Nat) :
    BitVec.carry i h (h &&& seed) false = true ↔ CarryRun h seed i := by
  induction i with
  | zero => simp [CarryRun]
  | succ i ih =>
    rw [masked_carry_step,Bool.and_eq_true,Bool.or_eq_true,ih]
    constructor
    · rintro ⟨last,first | previous⟩
      · exact ⟨i,by omega,last,first,by intro k hk hi; omega⟩
      · obtain ⟨j,hj,hh,hs,chain⟩ := previous
        refine ⟨j,by omega,hh,hs,?_⟩
        intro k hjk hki
        by_cases eq : k = i
        · simpa only [eq] using last
        · exact chain k hjk (by omega)
    · rintro ⟨j,hj,hh,hs,chain⟩
      by_cases eq : j = i
      · subst j
        exact ⟨hh,.inl hs⟩
      · have ji : j < i := by omega
        exact ⟨chain i ji (by omega),.inr ⟨j,ji,hh,hs,by intro k hjk hki; exact chain k hjk (by omega)⟩⟩

theorem masked_add_outside (h seed : BitVec w) (i : Nat) (bound : i < w)
    (empty : h.getLsbD i = false) :
    (h + (h &&& seed)).getLsbD i = true ↔ CarryRun h seed i := by
  rw [BitVec.getLsbD_add bound,BitVec.getLsbD_and,empty]
  simp only [Bool.false_and,Bool.false_xor]
  exact masked_carry_iff h seed i

theorem horizontalMask_bits : ∀ i : Fin 64,
    horizontalMask.getLsbD i.val = decide (0 < i.val % 8 ∧ i.val % 8 < 7) := by decide

theorem horizontalOpponent_bit (o : BitVec 64) (i : Nat) (hi : i < 64) :
    (o &&& horizontalMask).getLsbD i = true ↔
      o.getLsbD i = true ∧ 0 < i%8 ∧ i%8 < 7 := by
  rw [BitVec.getLsbD_and,horizontalMask_bits ⟨i,hi⟩]
  simp only [Bool.and_eq_true,decide_eq_true_eq]

def EastRay (p o : BitVec 64) (target : Nat) : Prop :=
  ∃ anchor, anchor+1 < target ∧ anchor/8 = target/8 ∧ p.getLsbD anchor = true ∧
    ∀ k, anchor < k → k < target → o.getLsbD k = true

theorem horizontal_chain_same_row (j i : Nat) (ji : j < i) (_ib : i < 64)
    (interior : ∀ k, j ≤ k → k < i → 0 < k%8 ∧ k%8 < 7) :
    0 < j ∧ (j-1)/8 = i/8 := by
  have hj := interior j (by omega) ji
  have stop : i ≤ (j/8)*8+7 := by
    by_cases good : i ≤ (j/8)*8+7
    · exact good
    have bad := good
    exfalso
    have h := interior ((j/8)*8+7) (by omega) (by omega)
    have rem : ((j/8)*8+7)%8 = 7 := by omega
    omega
  omega

theorem left1_bit (p : BitVec 64) (j : Nat) (positive : 0 < j) (bound : j < 64) :
    (p <<< (1 : Nat)).getLsbD j = p.getLsbD (j-1) := by
  simp only [BitVec.getLsbD_shiftLeft,show decide (j<64) = true by simp [bound],
    show decide (j<1) = false by simp; omega,Bool.not_false,Bool.true_and]
theorem horizontal_carry_ray (p o : BitVec 64) (i : Nat) (ib : i < 64) :
    CarryRun (o &&& horizontalMask) (p <<< (1 : Nat)) i ↔ EastRay p o i := by
  constructor
  · rintro ⟨j,ji,hj,seed,chain⟩
    have allh : ∀ k, j ≤ k → k < i → (o &&& horizontalMask).getLsbD k = true := by
      intro k jk ki
      by_cases eq : j = k
      · simpa only [←eq] using hj
      · exact chain k (by omega) ki
    have interior : ∀ k, j ≤ k → k < i → 0 < k%8 ∧ k%8 < 7 := by
      intro k jk ki
      exact ((horizontalOpponent_bit o k (by omega)).mp (allh k jk ki)).2
    obtain ⟨positive,sameRow⟩ := horizontal_chain_same_row j i ji ib interior
    refine ⟨j-1,by omega,sameRow,?_,?_⟩
    · rw [left1_bit p j positive (by omega)] at seed
      exact seed
    · intro k jk ki
      exact ((horizontalOpponent_bit o k (by omega)).mp (allh k (by omega) ki)).1
  · rintro ⟨anchor,ai,sameRow,hp,chain⟩
    have ab : anchor < 64 := by omega
    have interior : ∀ k, anchor < k → k < i → 0 < k%8 ∧ k%8 < 7 := by
      intro k ak ki
      omega
    refine ⟨anchor+1,ai,?_,?_,?_⟩
    · exact (horizontalOpponent_bit o (anchor+1) (by omega)).mpr
        ⟨chain _ (by omega) ai,interior _ (by omega) ai⟩
    · rw [left1_bit p (anchor+1) (by omega) (by omega)]
      simpa only [Nat.add_sub_cancel] using hp
    · intro k ak ki
      exact (horizontalOpponent_bit o k (by omega)).mpr
        ⟨chain k (by omega) ki,interior k (by omega) ki⟩

theorem portable_east_exact (p o : BitVec 64) (i : Nat) (ib : i < 64)
    (empty : (p ||| o).getLsbD i = false) :
    ((o &&& horizontalMask) + ((o &&& horizontalMask) &&& (p <<< (1 : Nat)))).getLsbD i = true ↔ EastRay p o i := by
  have ho : o.getLsbD i = false := by
    simp only [BitVec.getLsbD_or,Bool.or_eq_false_iff] at empty
    exact empty.2
  have hh : (o &&& horizontalMask).getLsbD i = false := by simp [ho]
  rw [masked_add_outside _ _ i ib hh,horizontal_carry_ray p o i ib]

open NeuralReversi.Bitboard.Moves in
 theorem eastRay_geometry (p o : BitVec 64) (i : Nat) (ib : i < 64) :
    EastRay p o i ↔ Ray p o i 0 1 := by
  constructor
  · rintro ⟨anchor,ai,sameRow,hp,chain⟩
    refine ⟨anchor,i-anchor,by omega,by omega,by omega,hp,?_,?_⟩
    · simp only [OnRay,Int.mul_zero,Int.add_zero,Int.mul_one]
      omega
    · intro k kp kd
      refine ⟨anchor+k,by omega,?_,chain _ (by omega) (by omega)⟩
      simp only [OnRay,Int.mul_zero,Int.add_zero,Int.mul_one]
      omega
  · rintro ⟨anchor,distance,ab,distLow,distHigh,hp,onRay,chain⟩
    simp only [OnRay,Int.mul_zero,Int.add_zero,Int.mul_one] at onRay
    have indexEq : i = anchor+distance := by omega
    refine ⟨anchor,by omega,by omega,hp,?_⟩
    intro k ak ki
    obtain ⟨index,indexBound,onPath,ho⟩ := chain (k-anchor) (by omega) (by omega)
    simp only [OnRay,Int.mul_zero,Int.add_zero,Int.mul_one] at onPath
    have eq : index = k := by omega
    simpa only [eq] using ho

end NeuralReversi.Bitboard.Moves

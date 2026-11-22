import NeuralReversi.Bitboard.PotentialModel
import NeuralReversi.Bitboard.MovesDirectional

namespace NeuralReversi.Bitboard.Moves

theorem masked_rol_eq_shl (h x : BitVec w) (d : Nat) (dw : d < w)
    (lowZero : ∀ i, i < d → h.getLsbD i = false) :
    h &&& x.rotateLeft d = h &&& (x <<< d) := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  rw [BitVec.getLsbD_and,BitVec.getLsbD_and,BitVec.getLsbD_rotateLeft_of_le dw,BitVec.getLsbD_shiftLeft]
  by_cases lo : i < d
  · simp [lo,lowZero i lo]
  · simp [lo,hi]

theorem masked_ror_eq_shr (h x : BitVec w) (d : Nat) (dw : d < w)
    (highZero : ∀ i, w-d ≤ i → i < w → h.getLsbD i = false) :
    h &&& x.rotateRight d = h &&& (x >>> d) := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  rw [BitVec.getLsbD_and,BitVec.getLsbD_and,BitVec.getLsbD_rotateRight_of_lt dw,BitVec.getLsbD_ushiftRight]
  by_cases low : i < w-d
  · simp only [low,↓reduceIte]

  · simp [low,highZero i (by omega) hi]

theorem rol_eq_shl_of_highZero (x : BitVec w) (d : Nat) (dw : d < w)
    (highZero : ∀ i, w-d ≤ i → i < w → x.getLsbD i = false) :
    x.rotateLeft d = x <<< d := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  rw [BitVec.getLsbD_rotateLeft_of_le dw,BitVec.getLsbD_shiftLeft]
  by_cases lo : i < d
  · simp [lo,highZero (w-d+i) (by omega) (by omega)]
  · simp [lo,hi]

theorem ror_eq_shr_of_lowZero (x : BitVec w) (d : Nat) (dw : d < w)
    (lowZero : ∀ i, i < d → x.getLsbD i = false) :
    x.rotateRight d = x >>> d := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  rw [BitVec.getLsbD_rotateRight_of_lt dw,BitVec.getLsbD_ushiftRight]
  by_cases low : i < w-d
  · simp only [low,↓reduceIte]

  · have z := BitVec.getLsbD_of_ge x (d+i) (by omega)
    simp [low,hi,z,lowZero (i-(w-d)) (by omega)]

theorem rol_negative_eq_ror (x : BitVec w) (d : Nat) (dp : 0 < d) (dw : d < w) :
    x.rotateLeft (w-d) = x.rotateRight d := by
  rw [BitVec.rotateLeft_def,BitVec.rotateRight_def,Nat.mod_eq_of_lt (by omega),Nat.mod_eq_of_lt dw]
  rw [Nat.sub_sub_self (by omega)]
  exact BitVec.or_comm _ _

def MaskGaps (mask : Bits) (d : Nat) : Prop :=
  ∀ i : Fin 64, (i.val < d ∨ 64-d ≤ i.val) → mask.getLsbD i.val = false

theorem sourceMask_gaps :
    MaskGaps horizontalMask 1 ∧ MaskGaps verticalMask 8 ∧
    MaskGaps diagonalMask 9 ∧ MaskGaps diagonalMask 7 := by unfold MaskGaps; decide

theorem sourcePreMask_gaps :
    (∀ i : Fin 64, i.val < 2 → (horizontalMask &&& (horizontalMask <<< (1:Nat))).getLsbD i.val = false) ∧
    (∀ i : Fin 64, i.val < 16 → (verticalMask &&& (verticalMask <<< (8:Nat))).getLsbD i.val = false) ∧
    (∀ i : Fin 64, i.val < 18 → (diagonalMask &&& (diagonalMask <<< (9:Nat))).getLsbD i.val = false) ∧
    (∀ i : Fin 64, i.val < 14 → (diagonalMask &&& (diagonalMask <<< (7:Nat))).getLsbD i.val = false) ∧
    (∀ i : Fin 64, 62 ≤ i.val → (horizontalMask &&& (horizontalMask >>> (1:Nat))).getLsbD i.val = false) ∧
    (∀ i : Fin 64, 48 ≤ i.val → (verticalMask &&& (verticalMask >>> (8:Nat))).getLsbD i.val = false) ∧
    (∀ i : Fin 64, 46 ≤ i.val → (diagonalMask &&& (diagonalMask >>> (9:Nat))).getLsbD i.val = false) ∧
    (∀ i : Fin 64, 50 ≤ i.val → (diagonalMask &&& (diagonalMask >>> (7:Nat))).getLsbD i.val = false) := by decide






theorem expandedLeft_within_mask (h p : Bits) (d i : Nat)
    (present : (expandedLeft h p d).getLsbD i = true) : h.getLsbD i = true := by
  rw [expandedLeft_exact] at present
  obtain ⟨n,low,_,hn⟩ := (six_bits _ _ i).mp present
  have ib := BitVec.lt_of_getLsbD hn
  have chain := ((powerLeft_bits h p d n i ib).mp hn).2.2
  simpa only [Nat.zero_mul,Nat.sub_zero] using chain 0 (by omega)

theorem expandedRight_within_mask (h p : Bits) (d i : Nat)
    (present : (expandedRight h p d).getLsbD i = true) : h.getLsbD i = true := by
  rw [expandedRight_exact] at present
  obtain ⟨n,low,_,hn⟩ := (six_bits _ _ i).mp present
  have chain := ((powerRight_bits h p d n i).mp hn).2
  simpa only [Nat.zero_mul,Nat.add_zero] using chain 0 (by omega)

theorem maskedOpponent_rol_eq_shl (o mask x : Bits) (d : Nat) (dw : d < 64)
    (gaps : MaskGaps mask d) :
    (o &&& mask) &&& x.rotateLeft d = (o &&& mask) &&& (x <<< d) := by
  apply masked_rol_eq_shl _ _ d dw
  intro i low
  have z := gaps ⟨i,by omega⟩ (.inl low)
  simp only [BitVec.getLsbD_and,z,Bool.and_false]

theorem maskedOpponent_ror_eq_shr (o mask x : Bits) (d : Nat) (dw : d < 64)
    (gaps : MaskGaps mask d) :
    (o &&& mask) &&& x.rotateRight d = (o &&& mask) &&& (x >>> d) := by
  apply masked_ror_eq_shr _ _ d dw
  intro i high ib
  have z := gaps ⟨i,ib⟩ (.inr high)
  simp only [BitVec.getLsbD_and,z,Bool.and_false]

theorem potential_rol_eq_shl (o mask : Bits) (d : Nat) (dw : d < 64)
    (gaps : MaskGaps mask d) :
    (o &&& mask).rotateLeft d = (o &&& mask) <<< d := by
  apply rol_eq_shl_of_highZero _ d dw
  intro i high ib
  have z := gaps ⟨i,ib⟩ (.inr high)
  simp only [BitVec.getLsbD_and,z,Bool.and_false]

theorem potential_ror_eq_shr (o mask : Bits) (d : Nat) (dw : d < 64)
    (gaps : MaskGaps mask d) :
    (o &&& mask).rotateRight d = (o &&& mask) >>> d := by
  apply ror_eq_shr_of_lowZero _ d dw
  intro i low
  have z := gaps ⟨i,by omega⟩ (.inl low)
  simp only [BitVec.getLsbD_and,z,Bool.and_false]



theorem maskedPre_rol_eq_shl (o mask x : Bits) (d : Nat) (dw : 2*d < 64)
    (gaps : ∀ i : Fin 64, i.val < 2*d → (mask &&& (mask <<< d)).getLsbD i.val = false) :
    let h := o &&& mask
    (h &&& (h <<< d)) &&& x.rotateLeft (2*d) = (h &&& (h <<< d)) &&& (x <<< (2*d)) := by
  dsimp only
  apply masked_rol_eq_shl _ _ (2*d) dw
  intro i low
  have z := gaps ⟨i,by omega⟩ low
  simp only [BitVec.getLsbD_and] at z
  simp only [BitVec.shiftLeft_and_distrib,BitVec.getLsbD_and]
  cases hm : mask.getLsbD i <;> cases hs : (mask <<< d).getLsbD i <;> simp_all

theorem maskedPre_ror_eq_shr (o mask x : Bits) (d : Nat) (dw : 2*d < 64)
    (gaps : ∀ i : Fin 64, 64-2*d ≤ i.val → (mask &&& (mask >>> d)).getLsbD i.val = false) :
    let h := o &&& mask
    (h &&& (h >>> d)) &&& x.rotateRight (2*d) = (h &&& (h >>> d)) &&& (x >>> (2*d)) := by
  dsimp only
  apply masked_ror_eq_shr _ _ (2*d) dw
  intro i high ib
  have z := gaps ⟨i,ib⟩ high
  simp only [BitVec.getLsbD_and] at z
  simp only [BitVec.ushiftRight_and_distrib,BitVec.getLsbD_and]
  cases hm : mask.getLsbD i <;> cases hs : (mask >>> d).getLsbD i <;> simp_all

theorem moves_rol_eq_shl (o mask p : Bits) (d : Nat) (dw : d < 64)
    (gaps : MaskGaps mask d) :
    (expandedLeft (o &&& mask) p d).rotateLeft d = expandedLeft (o &&& mask) p d <<< d := by
  apply rol_eq_shl_of_highZero _ d dw
  intro i high ib
  have z := gaps ⟨i,ib⟩ (.inr high)
  cases hf : (expandedLeft (o &&& mask) p d).getLsbD i
  · rfl
  · have within := expandedLeft_within_mask (o &&& mask) p d i hf
    simp [z] at within

theorem moves_ror_eq_shr (o mask p : Bits) (d : Nat) (dw : d < 64)
    (gaps : MaskGaps mask d) :
    (expandedRight (o &&& mask) p d).rotateRight d = expandedRight (o &&& mask) p d >>> d := by
  apply ror_eq_shr_of_lowZero _ d dw
  intro i low
  have z := gaps ⟨i,by omega⟩ (.inl low)
  cases hf : (expandedRight (o &&& mask) p d).getLsbD i
  · rfl
  · have within := expandedRight_within_mask (o &&& mask) p d i hf
    simp [z] at within


end NeuralReversi.Bitboard.Moves

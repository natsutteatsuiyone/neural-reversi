import NeuralReversi.Bitboard.CornerModel
import NeuralReversi.Bitboard.Counting

namespace NeuralReversi.Bitboard

set_option maxRecDepth 4096

theorem cornerMask_bit : ∀ i : Nat, i < 64 →
    cornerMask.getLsbD i = cornerCoordinate i := by
  simp only [Nat.forall_lt_succ_right]
  simp [cornerMask, cornerCoordinate]

def CornerMaskContract : Prop :=
  ∀ bits : Bits,
    (∀ i : Nat, i < 64 →
      (corners bits).getLsbD i = (bits.getLsbD i && cornerCoordinate i) ∧
      (nonCorners bits).getLsbD i = (bits.getLsbD i && !cornerCoordinate i)) ∧
    (corners bits ||| nonCorners bits) = bits ∧
    (corners bits &&& nonCorners bits) = 0

theorem cornerMaskContract_proven : CornerMaskContract := by
  intro bits
  refine ⟨?_, ?_, ?_⟩
  · intro i hi
    simp only [corners, nonCorners, BitVec.getLsbD_and, BitVec.getLsbD_not, hi,
      decide_true, Bool.true_and, cornerMask_bit i hi, and_self]
  · apply BitVec.eq_of_getLsbD_eq
    intro i hi
    simp only [corners, nonCorners, BitVec.getLsbD_or, BitVec.getLsbD_and,
      BitVec.getLsbD_not, hi, decide_true, Bool.true_and]
    cases bits.getLsbD i <;> cases cornerMask.getLsbD i <;> rfl
  · apply BitVec.eq_of_getLsbD_eq
    intro i hi
    simp only [corners, nonCorners, BitVec.getLsbD_and, BitVec.getLsbD_not,
      hi, decide_true, Bool.true_and]
    cases bits.getLsbD i <;> cases cornerMask.getLsbD i <;> simp

theorem cornerCount_exact (bits : Bits) :
    (count (corners bits)).toNat =
      (List.range 64).countP (fun i => bits.getLsbD i && cornerCoordinate i) := by
  rw [(populationCountContract_proven (corners bits)).1]
  apply List.countP_congr
  intro i member
  rw [((cornerMaskContract_proven bits).1 i (List.mem_range.mp member)).1]

theorem cornerCount_le_four (bits : Bits) : (count (corners bits)).toNat ≤ 4 := by
  rw [cornerCount_exact]
  have upper := List.countP_mono_left (l := List.range 64)
    (p := fun i => bits.getLsbD i && cornerCoordinate i) (q := cornerCoordinate)
    (by intro i _ selected; exact (Bool.and_eq_true_iff.mp selected).2)
  have four : (List.range 64).countP cornerCoordinate = 4 := by decide
  omega

def CornerWeightedCountContract : Prop :=
  ∀ bits : Bits,
    (cornerWeightedCount bits).toNat = (List.range 64).countP bits.getLsbD +
      (List.range 64).countP (fun i => bits.getLsbD i && cornerCoordinate i) ∧
    (cornerWeightedCount bits).toNat ≤ 68 ∧
    BitVec.uaddOverflow (count bits) (count (corners bits)) = false

theorem cornerWeightedCountContract_proven : CornerWeightedCountContract := by
  intro bits
  have allBound := (populationCountContract_proven bits).2.1
  have cornerBound := cornerCount_le_four bits
  have value : (cornerWeightedCount bits).toNat =
      (count bits).toNat + (count (corners bits)).toNat := by
    rw [cornerWeightedCount, BitVec.toNat_add, Nat.mod_eq_of_lt (by omega)]
  refine ⟨?_, by omega, ?_⟩
  · rw [value, (populationCountContract_proven bits).1, cornerCount_exact]
  · simp only [BitVec.uaddOverflow, decide_eq_false_iff_not]
    omega

theorem cornerStableMask_bit (bits : Bits) : ∀ i : Nat, i < 64 →
    (cornerStableMask bits).getLsbD i = cornerSupportedCell bits i := by
  simp only [Nat.forall_lt_succ_right]
  simp [cornerStableMask, cornerMask, cornerSupportedCell, cornerCoordinate, Bool.and_comm]

theorem cornerSupport_within (bits : Bits) : ∀ i : Nat, i < 64 →
    cornerSupportedCell bits i = true → cornerStencil i = true := by
  simp only [Nat.forall_lt_succ_right]
  simp [cornerSupportedCell, cornerStencil, cornerCoordinate]

def CornerStabilityContract : Prop :=
  ∀ bits : Bits,
    (cornerStability bits).toNat = (List.range 64).countP (cornerSupportedCell bits) ∧
    (cornerStability bits).toNat ≤ 12

theorem cornerStabilityContract_proven : CornerStabilityContract := by
  intro bits
  have value : (cornerStability bits).toNat =
      (List.range 64).countP (cornerSupportedCell bits) := by
    rw [cornerStability, (populationCountContract_proven (cornerStableMask bits)).1]
    apply List.countP_congr
    intro i member
    rw [cornerStableMask_bit bits i (List.mem_range.mp member)]
  have upper := List.countP_mono_left (l := List.range 64)
    (p := cornerSupportedCell bits) (q := cornerStencil)
    (by intro i member selected; exact cornerSupport_within bits i (List.mem_range.mp member) selected)
  have twelve : (List.range 64).countP cornerStencil = 12 := by decide
  exact ⟨value, by omega⟩

end NeuralReversi.Bitboard

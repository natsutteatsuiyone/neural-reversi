module
public import NeuralReversi.Bitboard.CornerModel
public import NeuralReversi.Bitboard.Counting

@[expose] public section

namespace NeuralReversi.Bitboard

private theorem cornerMask_bit_fin : ∀ i : Fin 64,
    cornerMask.getLsbD i.val = cornerCoordinate i.val := by
  decide +kernel

theorem cornerMask_bit : ∀ i : Nat, i < 64 →
    cornerMask.getLsbD i = cornerCoordinate i := fun i hi => cornerMask_bit_fin ⟨i, hi⟩

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

private theorem cornerStableMask_consts : ∀ i : Fin 64,
    (!decide (i.val < 1) && ((0x0000000000000001 : Bits).getLsbD (i.val - 1) || (0x0100000000000000 : Bits).getLsbD (i.val - 1))) =
      (decide (i.val / 8 = 0 ∨ i.val / 8 = 7) && decide (i.val % 8 = 1)) ∧
    ((0x0000000000000080 : Bits).getLsbD (1 + i.val) || (0x8000000000000000 : Bits).getLsbD (1 + i.val)) =
      (decide (i.val / 8 = 0 ∨ i.val / 8 = 7) && decide (i.val % 8 = 6)) ∧
    (!decide (i.val < 8) && ((0x0000000000000001 : Bits).getLsbD (i.val - 8) || (0x0000000000000080 : Bits).getLsbD (i.val - 8))) =
      (decide (i.val % 8 = 0 ∨ i.val % 8 = 7) && decide (i.val / 8 = 1)) ∧
    ((0x0100000000000000 : Bits).getLsbD (8 + i.val) || (0x8000000000000000 : Bits).getLsbD (8 + i.val)) =
      (decide (i.val % 8 = 0 ∨ i.val % 8 = 7) && decide (i.val / 8 = 6)) := by
  decide +kernel

theorem cornerStableMask_bit (bits : Bits) : ∀ i : Nat, i < 64 →
    (cornerStableMask bits).getLsbD i = cornerSupportedCell bits i := by
  intro i hi
  have hc := cornerStableMask_consts ⟨i, hi⟩
  simp only [cornerStableMask, cornerSupportedCell, BitVec.getLsbD_and, BitVec.getLsbD_or,
    BitVec.getLsbD_shiftLeft, BitVec.getLsbD_ushiftRight, hi, decide_true, Bool.true_and,
    cornerMask_bit i hi]
  rw [← Bool.and_assoc, hc.1, hc.2.1, ← Bool.and_assoc, hc.2.2.1, hc.2.2.2,
    Nat.add_comm 1 i, Nat.add_comm 8 i]
  cases bits.getLsbD i <;> cases cornerCoordinate i <;>
    cases decide (i / 8 = 0 ∨ i / 8 = 7) <;> cases decide (i % 8 = 0 ∨ i % 8 = 7) <;>
    cases decide (i % 8 = 1) <;> cases decide (i % 8 = 6) <;>
    cases decide (i / 8 = 1) <;> cases decide (i / 8 = 6) <;>
    simp [Bool.or_comm, Bool.or_left_comm]

theorem cornerSupport_within (bits : Bits) : ∀ i : Nat, i < 64 →
    cornerSupportedCell bits i = true → cornerStencil i = true := by
  intro i _ h
  simp only [cornerSupportedCell, Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq] at h
  simp only [cornerStencil, Bool.or_eq_true, decide_eq_true_eq]
  rcases h.2 with ((h0 | ⟨hr, (⟨hc, _⟩ | ⟨hc, _⟩)⟩) | ⟨hc, (⟨hr, _⟩ | ⟨hr, _⟩)⟩)
  · exact Or.inl (Or.inl h0)
  · exact Or.inl (Or.inr ⟨hr, Or.inl hc⟩)
  · exact Or.inl (Or.inr ⟨hr, Or.inr hc⟩)
  · exact Or.inr ⟨hc, Or.inl hr⟩
  · exact Or.inr ⟨hc, Or.inr hr⟩

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

module
public import NeuralReversi.Bitboard.Model

@[expose] public section

namespace NeuralReversi.Bitboard

theorem cpop_countP (bits : BitVec width) (n : Nat) :
    bits.cpopNatRec n 0 = (List.range n).countP bits.getLsbD := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [BitVec.cpopNatRec_succ, BitVec.cpopNatRec_eq, ih]
    simp only [List.range_succ, List.countP_append, List.countP_singleton]
    cases bit : bits.getLsbD n <;> simp_all

theorem count_toNat (bits : Bits) : (count bits).toNat = bits.cpop.toNat := by
  have bound := BitVec.toNat_cpop_le bits
  simp only [count, BitVec.toNat_setWidth]
  apply Nat.mod_eq_of_lt
  omega

def PopulationCountContract : Prop :=
  ∀ bits : Bits,
    (count bits).toNat = (List.range 64).countP bits.getLsbD ∧
    (count bits).toNat ≤ 64 ∧
    (count bits = 0#32 ↔ bits = 0#64) ∧
    (count (~~~bits)).toNat + (count bits).toNat = 64

theorem populationCountContract_proven : PopulationCountContract := by
  intro bits
  have exactCount : (count bits).toNat = (List.range 64).countP bits.getLsbD := by
    rw [count_toNat, BitVec.toNat_cpop, cpop_countP]
  have bound : (count bits).toNat ≤ 64 := by
    rw [count_toNat]
    exact BitVec.toNat_cpop_le bits
  refine ⟨exactCount, bound, ?_, ?_⟩
  · constructor
    · intro zero
      apply BitVec.zero_iff_eq_false.mpr
      have countZero : (List.range 64).countP bits.getLsbD = 0 := by
        rw [← exactCount, zero]
        rfl
      intro i
      by_cases within : i < 64
      · have absent := List.countP_eq_zero.mp countZero i (List.mem_range.mpr within)
        simpa using absent
      · exact BitVec.getLsbD_of_ge bits i (by omega)
    · intro zero
      simp [count, zero]
  · rw [count_toNat, count_toNat, BitVec.toNat_cpop_not]
    have := BitVec.toNat_cpop_le bits
    omega

theorem countP_disjoint_or (values : List α) (p q : α → Bool)
    (disjoint : ∀ value ∈ values, (p value && q value) = false) :
    values.countP (fun value => p value || q value) = values.countP p + values.countP q := by
  induction values with
  | nil => simp
  | cons value values ih =>
    have head := disjoint value (by simp)
    have tail : ∀ item ∈ values, (p item && q item) = false := by
      intro item member
      exact disjoint item (by simp [member])
    have sum := ih tail
    simp only [List.countP_cons]
    cases hp : p value <;> cases hq : q value <;> simp_all <;> omega

theorem count_or_of_disjoint (player opponent : Bits) (disjoint : player &&& opponent = 0#64) :
    (count (player ||| opponent)).toNat = (count player).toNat + (count opponent).toNat := by
  rw [(populationCountContract_proven (player ||| opponent)).1,
    (populationCountContract_proven player).1, (populationCountContract_proven opponent).1]
  have unionBits : (player ||| opponent).getLsbD =
      fun i => player.getLsbD i || opponent.getLsbD i := by
    funext i
    simp
  rw [unionBits]
  apply countP_disjoint_or
  intro i _
  have bit := congrArg (fun value : Bits => value.getLsbD i) disjoint
  simpa using bit

end NeuralReversi.Bitboard

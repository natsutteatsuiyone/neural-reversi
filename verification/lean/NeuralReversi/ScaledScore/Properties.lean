module
public import NeuralReversi.ScaledScore.Model

@[expose] public section

namespace NeuralReversi.ScaledScore

theorem scaled_value (discs : Word) :
    (discs <<< 8).toInt = (discs.toInt * 256).bmod 4294967296 := by
  have shifted : discs <<< 8 = discs * (256 : Word) := by
    apply BitVec.eq_of_toNat_eq
    simp [Nat.shiftLeft_eq]
  rw [shifted, BitVec.toInt_mul]
  simp

theorem disc_floor (score : ScaledScore) :
    (toDiscDiff score).toInt = score.raw.toInt / 256 := by
  simp [toDiscDiff, BitVec.toInt_sshiftRight, Int.shiftRight_eq_div_pow]

theorem bounded_scaled_value (discs : Word) (lo : -127 ≤ discs.toInt)
    (hi : discs.toInt ≤ 127) :
    (discs <<< 8).toInt = discs.toInt * 256 := by
  rw [scaled_value]
  exact Int.bmod_eq_of_le (by omega) (by omega)

theorem bounded_roundtrip (discs : Word) (lo : -127 ≤ discs.toInt)
    (hi : discs.toInt ≤ 127) :
    toDiscDiff ⟨discs <<< 8⟩ = discs := by
  apply BitVec.eq_of_toInt_eq
  rw [disc_floor]
  change (discs <<< 8).toInt / 256 = discs.toInt
  rw [bounded_scaled_value discs lo hi]
  omega

def ConstructorContract : Prop :=
  (∀ (checked : Bool) (raw : Word),
    (fromRaw checked raw).map (fun score => (value score).toInt) =
      if checked && decide (raw.toInt < -32767 ∨ raw.toInt > 32767) then
        .error "assertion failed: raw_value >= -Self::INF.0 && raw_value <= Self::INF.0"
      else .ok raw.toInt) ∧
  (∀ (checked : Bool) (discs : Word),
    (fromDiscDiff checked discs).map (fun score => (value score).toInt) =
      if checked && decide (discs.toInt < -127 ∨ discs.toInt > 127) then
        .error "assertion failed: disc_diff >= -SCORE_INF && disc_diff <= SCORE_INF"
      else .ok ((discs.toInt * 256).bmod 4294967296))

theorem constructorContract_proven : ConstructorContract := by
  constructor
  · intro checked raw
    unfold fromRaw
    split <;> rfl
  · intro checked discs
    unfold fromDiscDiff
    split
    · rfl
    · simp only [Except.map, value, Except.ok.injEq]
      exact scaled_value discs

def ObservationContract : Prop :=
  ∀ score : ScaledScore,
    (toDiscDiff score).toInt = score.raw.toInt / 256 ∧
    -8388608 ≤ (toDiscDiff score).toInt ∧ (toDiscDiff score).toInt ≤ 8388607 ∧
    (value score).toInt = 256 * (toDiscDiff score).toInt + score.raw.toInt % 256 ∧
    0 ≤ score.raw.toInt % 256 ∧ score.raw.toInt % 256 < 256

theorem observationContract_proven : ObservationContract := by
  intro score
  have lower := BitVec.le_toInt score.raw
  have upper := BitVec.toInt_lt (x := score.raw)
  have floor := disc_floor score
  have remainder := Int.emod_nonneg score.raw.toInt (by decide : (256 : Int) ≠ 0)
  have remBound := Int.emod_lt_of_pos score.raw.toInt (by decide : (0 : Int) < 256)
  have reconstruct := Int.mul_ediv_add_emod score.raw.toInt 256
  refine ⟨floor, ?_, ?_, ?_, remainder, remBound⟩
  · omega
  · omega
  · simp only [value]
    omega

def RoundTripContract : Prop :=
  ∀ discs : Word, -127 ≤ discs.toInt → discs.toInt ≤ 127 →
    ∃ score : ScaledScore,
      fromDiscDiff true discs = .ok score ∧ toDiscDiff score = discs ∧
      (value score).toInt = discs.toInt * 256 ∧
      fromRaw true (value score) = .ok score

theorem roundTripContract_proven : RoundTripContract := by
  intro discs lo hi
  let score : ScaledScore := ⟨discs <<< 8⟩
  have scaled : (value score).toInt = discs.toInt * 256 :=
    bounded_scaled_value discs lo hi
  refine ⟨score, ?_, bounded_roundtrip discs lo hi, scaled, ?_⟩
  · simp [fromDiscDiff, show ¬ (discs.toInt < -127 ∨ discs.toInt > 127) by omega, score]
  · have valid : ¬ ((value score).toInt < -32767 ∨ (value score).toInt > 32767) := by omega
    simp only [fromRaw, Bool.true_and, valid, decide_false, Bool.false_eq_true, ↓reduceIte]
    rfl

end NeuralReversi.ScaledScore

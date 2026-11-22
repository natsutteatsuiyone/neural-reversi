import NeuralReversi.Solve1.NeonXor
import NeuralReversi.Solve1.ScalarCounts

namespace NeuralReversi.Solve1.Neon

theorem metadata_safe : ∀ s : Fin 64, ∀ lane : Fin 4,
    classId s.val lane < 24 ∧ (pairKey (classId s.val lane)).1 < 8 ∧ rankShift s.val < 64 := by decide

theorem laneCounts_eq (s : Fin 64) (lane : Fin 4) (p : Word)
    (active : lane.val < 3 ∨ hasFourth s.val = true) :
    laneCounts s.val lane p =
      ⟨LineTable.countFlip (Scalar.lanePos s.val lane) (Scalar.laneIndex s.val lane p),
       LineTable.countFlip (Scalar.lanePos s.val lane) (Scalar.laneIndex s.val lane (~~~p))⟩ := by
  change (⟨LineTable.countFlip (classPos s.val lane) (laneIndex s.val lane p),
    LineTable.countFlip (classPos s.val lane) (laneIndex s.val lane p ^^^ laneXor s.val lane)⟩ : Packed.Counts) = _
  rw [laneIndex_eq, classPos_correct, xor_correct s lane p active]

theorem laneCounts_toNat (s : Fin 64) (lane : Fin 4) (p : Word)
    (active : lane.val < 3 ∨ hasFourth s.val = true) :
    (laneCounts s.val lane p).low.toNat = 2 * Scalar.laneReference s.val lane p ∧
    (laneCounts s.val lane p).high.toNat = 2 * Scalar.laneReference s.val lane (~~~p) := by
  rw [laneCounts_eq s lane p active]
  simp only [LineTable.countFlip_correct, Scalar.lane_correct, and_self]

theorem laneCounts_bound (s : Nat) (lane : Fin 4) (p : Word) :
    (laneCounts s lane p).low.toNat ≤ 12 ∧ (laneCounts s lane p).high.toNat ≤ 12 := by
  exact ⟨LineTable.countFlip_bound _ _, LineTable.countFlip_bound _ _⟩

theorem entries_length (s : Nat) (p : Word) : (entries s p).length ≤ 4 := by
  by_cases h : hasFourth s <;> simp [entries, h]

theorem entries_bound (s : Nat) (p : Word) : Packed.LineBounds (entries s p) := by
  intro item hitem
  by_cases h : hasFourth s
  · simp [entries, h] at hitem
    rcases hitem with hitem | hitem | hitem | hitem <;> subst item <;>
      exact laneCounts_bound s _ p
  · simp [entries, h] at hitem
    rcases hitem with hitem | hitem | hitem <;> subst item <;>
      exact laneCounts_bound s _ p

theorem totals_correct (s : Fin 64) (p : Word) :
    Packed.lowTotal (entries s.val p) = 2 * Geometry.reference p s.val ∧
    Packed.highTotal (entries s.val p) = 2 * Geometry.reference (~~~p) s.val := by
  have e0 := laneCounts_toNat s 0 p (Or.inl (by decide))
  have e1 := laneCounts_toNat s 1 p (Or.inl (by decide))
  have e2 := laneCounts_toNat s 2 p (Or.inl (by decide))
  have es := Scalar.laneReference_sum s.val p
  have eo := Scalar.laneReference_sum s.val (~~~p)
  by_cases h : hasFourth s.val = true
  · have e3 := laneCounts_toNat s 3 p (Or.inr h)
    simp only [entries, h, ↓reduceIte, List.cons_append, List.nil_append,
      Packed.lowTotal, Packed.highTotal]
    omega
  · have hf : hasFourth s.val = false := Bool.eq_false_iff.mpr h
    have hz : (Scalar.params s.val).maskD9 = 0 := by
      have eq := hasFourth_correct s
      rw [hf] at eq
      exact bne_eq_false_iff_eq.mp eq.symm
    have z : Scalar.laneReference s.val 3 p = 0 := by simp [Scalar.laneReference, hz]
    have zo : Scalar.laneReference s.val 3 (~~~p) = 0 := by simp [Scalar.laneReference, hz]
    simp only [entries, hf, Bool.false_eq_true, ↓reduceIte, List.append_nil, Packed.lowTotal, Packed.highTotal]
    omega

def countLow (s : Nat) (p : Word) : BitVec 32 := countPacked s p &&& 0xff
def countHigh (s : Nat) (p : Word) : BitVec 32 := countPacked s p >>> 8

theorem counts_correct (s : Fin 64) (p : Word) :
    (countLow s.val p).toNat = 2 * Geometry.reference p s.val ∧
    (countHigh s.val p).toNat = 2 * Geometry.reference (~~~p) s.val := by
  have unpack := Packed.unpack_exact (entries s.val p) (entries_length _ _) (entries_bound _ _)
  have totals := totals_correct s p
  simpa only [countLow, countHigh, countPacked_eq, totals.1, totals.2] using unpack

theorem counts_bounded (s : Nat) (p : Word) :
    (countLow s p).toNat ≤ 48 ∧ (countHigh s p).toNat ≤ 48 := by
  have safe := Packed.packed_value_safe (entries s p) (entries_length _ _) (entries_bound _ _)
  have unpack := Packed.unpack_exact (entries s p) (entries_length _ _) (entries_bound _ _)
  simp only [countLow, countHigh, countPacked_eq, unpack.1, unpack.2]
  exact ⟨safe.2.2.1, safe.2.2.2⟩

theorem counts_signed (s : Fin 64) (p : Word) :
    (countLow s.val p).toInt = (2 * Geometry.reference p s.val : Nat) ∧
    (countHigh s.val p).toInt = (2 * Geometry.reference (~~~p) s.val : Nat) := by
  have bounds := counts_bounded s.val p
  have counts := counts_correct s p
  constructor
  · rw [BitVec.toInt_eq_toNat_of_lt (by omega), counts.1]
  · rw [BitVec.toInt_eq_toNat_of_lt (by omega), counts.2]

theorem countPacked_adds_safe (s : Nat) (p : Word) : Packed.AddsSafe (entries s p) 0 :=
  (Packed.packed_value_safe (entries s p) (entries_length _ _) (entries_bound _ _)).2.1

theorem counts_eq_scalar (s : Fin 64) (p : Word) :
    countLow s.val p = Scalar.count s.val p ∧
    countHigh s.val p = Scalar.count s.val (~~~p) := by
  have counts := counts_correct s p
  constructor <;> apply BitVec.eq_of_toNat_eq
  · rw [counts.1, Scalar.count_toNat]
  · rw [counts.2, Scalar.count_toNat]

end NeuralReversi.Solve1.Neon

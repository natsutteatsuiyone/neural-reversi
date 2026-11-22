import NeuralReversi.Solve1.ScalarProofs.Sq63
namespace NeuralReversi.Solve1.Scalar
theorem lane_correct (s : Fin 64) (lane : Fin 4) (p : Word) :
    LineTable.reference (lanePos s.val lane) (laneIndex s.val lane p) =
      laneReference s.val lane p := by
  rcases s with ⟨s, hs⟩
  rcases lane with ⟨l, hl⟩
  match s, l with
  | 0, 0 => exact lane_0_0 p
  | 0, 1 => exact lane_0_1 p
  | 0, 2 => exact lane_0_2 p
  | 0, 3 => exact lane_0_3 p
  | 1, 0 => exact lane_1_0 p
  | 1, 1 => exact lane_1_1 p
  | 1, 2 => exact lane_1_2 p
  | 1, 3 => exact lane_1_3 p
  | 2, 0 => exact lane_2_0 p
  | 2, 1 => exact lane_2_1 p
  | 2, 2 => exact lane_2_2 p
  | 2, 3 => exact lane_2_3 p
  | 3, 0 => exact lane_3_0 p
  | 3, 1 => exact lane_3_1 p
  | 3, 2 => exact lane_3_2 p
  | 3, 3 => exact lane_3_3 p
  | 4, 0 => exact lane_4_0 p
  | 4, 1 => exact lane_4_1 p
  | 4, 2 => exact lane_4_2 p
  | 4, 3 => exact lane_4_3 p
  | 5, 0 => exact lane_5_0 p
  | 5, 1 => exact lane_5_1 p
  | 5, 2 => exact lane_5_2 p
  | 5, 3 => exact lane_5_3 p
  | 6, 0 => exact lane_6_0 p
  | 6, 1 => exact lane_6_1 p
  | 6, 2 => exact lane_6_2 p
  | 6, 3 => exact lane_6_3 p
  | 7, 0 => exact lane_7_0 p
  | 7, 1 => exact lane_7_1 p
  | 7, 2 => exact lane_7_2 p
  | 7, 3 => exact lane_7_3 p
  | 8, 0 => exact lane_8_0 p
  | 8, 1 => exact lane_8_1 p
  | 8, 2 => exact lane_8_2 p
  | 8, 3 => exact lane_8_3 p
  | 9, 0 => exact lane_9_0 p
  | 9, 1 => exact lane_9_1 p
  | 9, 2 => exact lane_9_2 p
  | 9, 3 => exact lane_9_3 p
  | 10, 0 => exact lane_10_0 p
  | 10, 1 => exact lane_10_1 p
  | 10, 2 => exact lane_10_2 p
  | 10, 3 => exact lane_10_3 p
  | 11, 0 => exact lane_11_0 p
  | 11, 1 => exact lane_11_1 p
  | 11, 2 => exact lane_11_2 p
  | 11, 3 => exact lane_11_3 p
  | 12, 0 => exact lane_12_0 p
  | 12, 1 => exact lane_12_1 p
  | 12, 2 => exact lane_12_2 p
  | 12, 3 => exact lane_12_3 p
  | 13, 0 => exact lane_13_0 p
  | 13, 1 => exact lane_13_1 p
  | 13, 2 => exact lane_13_2 p
  | 13, 3 => exact lane_13_3 p
  | 14, 0 => exact lane_14_0 p
  | 14, 1 => exact lane_14_1 p
  | 14, 2 => exact lane_14_2 p
  | 14, 3 => exact lane_14_3 p
  | 15, 0 => exact lane_15_0 p
  | 15, 1 => exact lane_15_1 p
  | 15, 2 => exact lane_15_2 p
  | 15, 3 => exact lane_15_3 p
  | 16, 0 => exact lane_16_0 p
  | 16, 1 => exact lane_16_1 p
  | 16, 2 => exact lane_16_2 p
  | 16, 3 => exact lane_16_3 p
  | 17, 0 => exact lane_17_0 p
  | 17, 1 => exact lane_17_1 p
  | 17, 2 => exact lane_17_2 p
  | 17, 3 => exact lane_17_3 p
  | 18, 0 => exact lane_18_0 p
  | 18, 1 => exact lane_18_1 p
  | 18, 2 => exact lane_18_2 p
  | 18, 3 => exact lane_18_3 p
  | 19, 0 => exact lane_19_0 p
  | 19, 1 => exact lane_19_1 p
  | 19, 2 => exact lane_19_2 p
  | 19, 3 => exact lane_19_3 p
  | 20, 0 => exact lane_20_0 p
  | 20, 1 => exact lane_20_1 p
  | 20, 2 => exact lane_20_2 p
  | 20, 3 => exact lane_20_3 p
  | 21, 0 => exact lane_21_0 p
  | 21, 1 => exact lane_21_1 p
  | 21, 2 => exact lane_21_2 p
  | 21, 3 => exact lane_21_3 p
  | 22, 0 => exact lane_22_0 p
  | 22, 1 => exact lane_22_1 p
  | 22, 2 => exact lane_22_2 p
  | 22, 3 => exact lane_22_3 p
  | 23, 0 => exact lane_23_0 p
  | 23, 1 => exact lane_23_1 p
  | 23, 2 => exact lane_23_2 p
  | 23, 3 => exact lane_23_3 p
  | 24, 0 => exact lane_24_0 p
  | 24, 1 => exact lane_24_1 p
  | 24, 2 => exact lane_24_2 p
  | 24, 3 => exact lane_24_3 p
  | 25, 0 => exact lane_25_0 p
  | 25, 1 => exact lane_25_1 p
  | 25, 2 => exact lane_25_2 p
  | 25, 3 => exact lane_25_3 p
  | 26, 0 => exact lane_26_0 p
  | 26, 1 => exact lane_26_1 p
  | 26, 2 => exact lane_26_2 p
  | 26, 3 => exact lane_26_3 p
  | 27, 0 => exact lane_27_0 p
  | 27, 1 => exact lane_27_1 p
  | 27, 2 => exact lane_27_2 p
  | 27, 3 => exact lane_27_3 p
  | 28, 0 => exact lane_28_0 p
  | 28, 1 => exact lane_28_1 p
  | 28, 2 => exact lane_28_2 p
  | 28, 3 => exact lane_28_3 p
  | 29, 0 => exact lane_29_0 p
  | 29, 1 => exact lane_29_1 p
  | 29, 2 => exact lane_29_2 p
  | 29, 3 => exact lane_29_3 p
  | 30, 0 => exact lane_30_0 p
  | 30, 1 => exact lane_30_1 p
  | 30, 2 => exact lane_30_2 p
  | 30, 3 => exact lane_30_3 p
  | 31, 0 => exact lane_31_0 p
  | 31, 1 => exact lane_31_1 p
  | 31, 2 => exact lane_31_2 p
  | 31, 3 => exact lane_31_3 p
  | 32, 0 => exact lane_32_0 p
  | 32, 1 => exact lane_32_1 p
  | 32, 2 => exact lane_32_2 p
  | 32, 3 => exact lane_32_3 p
  | 33, 0 => exact lane_33_0 p
  | 33, 1 => exact lane_33_1 p
  | 33, 2 => exact lane_33_2 p
  | 33, 3 => exact lane_33_3 p
  | 34, 0 => exact lane_34_0 p
  | 34, 1 => exact lane_34_1 p
  | 34, 2 => exact lane_34_2 p
  | 34, 3 => exact lane_34_3 p
  | 35, 0 => exact lane_35_0 p
  | 35, 1 => exact lane_35_1 p
  | 35, 2 => exact lane_35_2 p
  | 35, 3 => exact lane_35_3 p
  | 36, 0 => exact lane_36_0 p
  | 36, 1 => exact lane_36_1 p
  | 36, 2 => exact lane_36_2 p
  | 36, 3 => exact lane_36_3 p
  | 37, 0 => exact lane_37_0 p
  | 37, 1 => exact lane_37_1 p
  | 37, 2 => exact lane_37_2 p
  | 37, 3 => exact lane_37_3 p
  | 38, 0 => exact lane_38_0 p
  | 38, 1 => exact lane_38_1 p
  | 38, 2 => exact lane_38_2 p
  | 38, 3 => exact lane_38_3 p
  | 39, 0 => exact lane_39_0 p
  | 39, 1 => exact lane_39_1 p
  | 39, 2 => exact lane_39_2 p
  | 39, 3 => exact lane_39_3 p
  | 40, 0 => exact lane_40_0 p
  | 40, 1 => exact lane_40_1 p
  | 40, 2 => exact lane_40_2 p
  | 40, 3 => exact lane_40_3 p
  | 41, 0 => exact lane_41_0 p
  | 41, 1 => exact lane_41_1 p
  | 41, 2 => exact lane_41_2 p
  | 41, 3 => exact lane_41_3 p
  | 42, 0 => exact lane_42_0 p
  | 42, 1 => exact lane_42_1 p
  | 42, 2 => exact lane_42_2 p
  | 42, 3 => exact lane_42_3 p
  | 43, 0 => exact lane_43_0 p
  | 43, 1 => exact lane_43_1 p
  | 43, 2 => exact lane_43_2 p
  | 43, 3 => exact lane_43_3 p
  | 44, 0 => exact lane_44_0 p
  | 44, 1 => exact lane_44_1 p
  | 44, 2 => exact lane_44_2 p
  | 44, 3 => exact lane_44_3 p
  | 45, 0 => exact lane_45_0 p
  | 45, 1 => exact lane_45_1 p
  | 45, 2 => exact lane_45_2 p
  | 45, 3 => exact lane_45_3 p
  | 46, 0 => exact lane_46_0 p
  | 46, 1 => exact lane_46_1 p
  | 46, 2 => exact lane_46_2 p
  | 46, 3 => exact lane_46_3 p
  | 47, 0 => exact lane_47_0 p
  | 47, 1 => exact lane_47_1 p
  | 47, 2 => exact lane_47_2 p
  | 47, 3 => exact lane_47_3 p
  | 48, 0 => exact lane_48_0 p
  | 48, 1 => exact lane_48_1 p
  | 48, 2 => exact lane_48_2 p
  | 48, 3 => exact lane_48_3 p
  | 49, 0 => exact lane_49_0 p
  | 49, 1 => exact lane_49_1 p
  | 49, 2 => exact lane_49_2 p
  | 49, 3 => exact lane_49_3 p
  | 50, 0 => exact lane_50_0 p
  | 50, 1 => exact lane_50_1 p
  | 50, 2 => exact lane_50_2 p
  | 50, 3 => exact lane_50_3 p
  | 51, 0 => exact lane_51_0 p
  | 51, 1 => exact lane_51_1 p
  | 51, 2 => exact lane_51_2 p
  | 51, 3 => exact lane_51_3 p
  | 52, 0 => exact lane_52_0 p
  | 52, 1 => exact lane_52_1 p
  | 52, 2 => exact lane_52_2 p
  | 52, 3 => exact lane_52_3 p
  | 53, 0 => exact lane_53_0 p
  | 53, 1 => exact lane_53_1 p
  | 53, 2 => exact lane_53_2 p
  | 53, 3 => exact lane_53_3 p
  | 54, 0 => exact lane_54_0 p
  | 54, 1 => exact lane_54_1 p
  | 54, 2 => exact lane_54_2 p
  | 54, 3 => exact lane_54_3 p
  | 55, 0 => exact lane_55_0 p
  | 55, 1 => exact lane_55_1 p
  | 55, 2 => exact lane_55_2 p
  | 55, 3 => exact lane_55_3 p
  | 56, 0 => exact lane_56_0 p
  | 56, 1 => exact lane_56_1 p
  | 56, 2 => exact lane_56_2 p
  | 56, 3 => exact lane_56_3 p
  | 57, 0 => exact lane_57_0 p
  | 57, 1 => exact lane_57_1 p
  | 57, 2 => exact lane_57_2 p
  | 57, 3 => exact lane_57_3 p
  | 58, 0 => exact lane_58_0 p
  | 58, 1 => exact lane_58_1 p
  | 58, 2 => exact lane_58_2 p
  | 58, 3 => exact lane_58_3 p
  | 59, 0 => exact lane_59_0 p
  | 59, 1 => exact lane_59_1 p
  | 59, 2 => exact lane_59_2 p
  | 59, 3 => exact lane_59_3 p
  | 60, 0 => exact lane_60_0 p
  | 60, 1 => exact lane_60_1 p
  | 60, 2 => exact lane_60_2 p
  | 60, 3 => exact lane_60_3 p
  | 61, 0 => exact lane_61_0 p
  | 61, 1 => exact lane_61_1 p
  | 61, 2 => exact lane_61_2 p
  | 61, 3 => exact lane_61_3 p
  | 62, 0 => exact lane_62_0 p
  | 62, 1 => exact lane_62_1 p
  | 62, 2 => exact lane_62_2 p
  | 62, 3 => exact lane_62_3 p
  | 63, 0 => exact lane_63_0 p
  | 63, 1 => exact lane_63_1 p
  | 63, 2 => exact lane_63_2 p
  | 63, 3 => exact lane_63_3 p
  | _, l + 4 => omega
  | s + 64, _ => omega

def laneCount (s : Nat) (lane : Fin 4) (p : Word) : BitVec 32 :=
  (LineTable.countFlip (lanePos s lane) (laneIndex s lane p)).setWidth 32

def count (s : Nat) (p : Word) : BitVec 32 :=
  laneCount s 0 p + laneCount s 1 p + laneCount s 2 p +
    (if (params s).maskD9 = 0 then 0 else laneCount s 3 p)

theorem laneCount_bound (s : Nat) (lane : Fin 4) (p : Word) :
    (laneCount s lane p).toNat ≤ 12 := by
  simpa [laneCount, BitVec.toNat_setWidth, Nat.mod_eq_of_lt
    (Nat.lt_of_le_of_lt (LineTable.countFlip_bound (lanePos s lane) (laneIndex s lane p))
      (by decide : 12 < 2^32))] using
    LineTable.countFlip_bound (lanePos s lane) (laneIndex s lane p)

theorem laneCount_toNat (s : Fin 64) (lane : Fin 4) (p : Word) :
    (laneCount s.val lane p).toNat = 2 * laneReference s.val lane p := by
  simp only [laneCount, BitVec.toNat_setWidth, LineTable.countFlip_correct,
    lane_correct]
  have h := LineTable.reference_bound (lanePos s.val lane) (laneIndex s.val lane p)
  rw [lane_correct] at h
  omega

theorem laneReference_sum (s : Nat) (p : Word) :
    laneReference s 0 p + laneReference s 1 p + laneReference s 2 p +
      laneReference s 3 p = Geometry.reference p s := by
  by_cases h : (params s).maskD9 = 0#64
  · simp [laneReference, Geometry.reference, h] <;> omega
  · simp [laneReference, Geometry.reference, h] <;> omega

theorem count_toNat (s : Fin 64) (p : Word) :
    (count s.val p).toNat = 2 * Geometry.reference p s.val := by
  have h0 := laneCount_bound s.val 0 p
  have h1 := laneCount_bound s.val 1 p
  have h2 := laneCount_bound s.val 2 p
  have h3 := laneCount_bound s.val 3 p
  have e0 := laneCount_toNat s 0 p
  have e1 := laneCount_toNat s 1 p
  have e2 := laneCount_toNat s 2 p
  have e3 := laneCount_toNat s 3 p
  have hs := laneReference_sum s.val p
  have ab : (laneCount s.val 0 p + laneCount s.val 1 p).toNat =
      (laneCount s.val 0 p).toNat + (laneCount s.val 1 p).toNat := by
    rw [BitVec.toNat_add]
    exact Nat.mod_eq_of_lt (by omega)
  have abc : (laneCount s.val 0 p + laneCount s.val 1 p + laneCount s.val 2 p).toNat =
      (laneCount s.val 0 p).toNat + (laneCount s.val 1 p).toNat + (laneCount s.val 2 p).toNat := by
    rw [BitVec.toNat_add, ab]
    exact Nat.mod_eq_of_lt (by omega)
  unfold count
  split
  · have hz : laneReference s.val 3 p = 0 := by simp [laneReference, *]
    change (laneCount s.val 0 p + laneCount s.val 1 p + laneCount s.val 2 p + 0#32).toNat = _
    rw [BitVec.add_zero, abc]
    omega
  · rw [BitVec.toNat_add, abc, Nat.mod_eq_of_lt (by omega)]
    omega

end NeuralReversi.Solve1.Scalar

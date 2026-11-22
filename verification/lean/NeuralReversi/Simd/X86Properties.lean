import NeuralReversi.Simd.X86Model

namespace NeuralReversi.Simd.X86

theorem unpack256_pack256 (a : Vec256) : unpack256 (pack256 a) = a := by
  funext i
  rcases i with ⟨i,hi⟩
  have cases : i=0 ∨ i=1 ∨ i=2 ∨ i=3 := by omega
  rcases cases with rfl|rfl|rfl|rfl
  all_goals apply BitVec.eq_of_getLsbD_eq
  all_goals intro j hj
  all_goals simp only [unpack256,pack256]
  all_goals rw [BitVec.getLsbD_extractLsb']
  all_goals simp only [BitVec.getLsbD_append]
  all_goals simp +arith [hj]

theorem pack256_unpack256 (a : BitVec 256) : pack256 (unpack256 a) = a := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [pack256,unpack256,BitVec.getLsbD_append,BitVec.getLsbD_extractLsb']
  by_cases h0 : i < 64
  · simp [h0]
  · by_cases h1 : i < 128
    · have b : i-64 < 64 := by omega
      have eq : 64+(i-64)=i := by omega
      simp [h0,b,eq]
    · by_cases h2 : i < 192
      · have b : ¬i-64 < 64 := by omega
        have c : i-64-64 < 64 := by omega
        have eq : 128+(i-64-64)=i := by omega
        simp [h0,b,c,eq]
      · have b : ¬i-64 < 64 := by omega
        have c : ¬i-64-64 < 64 := by omega
        have d : i-64-64-64 < 64 := by omega
        have eq : 192+(i-64-64-64)=i := by omega
        simp only [show (3 : Fin 4).val = 3 from rfl, Nat.reduceMul, h0,b,c,d,decide_true,Bool.true_and,eq,↓reduceIte]

theorem shuffleEpi32_4e (a : Vec128) : shuffleEpi32 a 0x4e = fun i => a ⟨1-i.val,by omega⟩ := by
  funext i
  rcases i with ⟨i,hi⟩
  have cases : i=0 ∨ i=1 := by omega
  rcases cases with rfl|rfl
  · change (a 1).extractLsb' 32 32 ++ (a 1).extractLsb' 0 32 = a 1
    exact BitVec.extractLsb'_append_extractLsb'
  · change (a 0).extractLsb' 32 32 ++ (a 0).extractLsb' 0 32 = a 0
    exact BitVec.extractLsb'_append_extractLsb'

theorem horizontalOrU64_exact (a : Vec256) :
    horizontalOrU64 a = a 0 ||| a 1 ||| a 2 ||| a 3 := by
  rw [horizontalOrU64,shuffleEpi32_4e]
  change (a 0 ||| a 2) ||| (a 1 ||| a 3) = a 0 ||| a 1 ||| a 2 ||| a 3
  simp only [BitVec.or_assoc]
  rw [← BitVec.or_assoc (a 2) (a 1) (a 3),BitVec.or_comm (a 2) (a 1),BitVec.or_assoc]

end NeuralReversi.Simd.X86

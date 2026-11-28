module
public import NeuralReversi.Simd.ZmmModel
public import NeuralReversi.Simd.X86Properties

@[expose] public section

namespace NeuralReversi.Simd.X86
theorem ternary512_eq (a b c : Vec512) (imm : BitVec 8) :
    ternary512 a b c imm = fun i => ternaryWord (a i) (b i) (c i) imm := by
  funext i
  exact ternaryWordFast_exact (a i) (b i) (c i) imm

theorem horizontalOr2Zmm_exact (a b : Vec512) :
    horizontalOr2Zmm a b =
      (a 0 ||| a 1 ||| a 2 ||| a 3 ||| a 4 ||| a 5 ||| a 6 ||| a 7,
       b 0 ||| b 1 ||| b 2 ||| b 3 ||| b 4 ||| b 5 ||| b 6 ||| b 7) := by
  have flip (x y z : Word) : x ||| (y ||| z) = y ||| (x ||| z) := by
    rw [←BitVec.or_assoc,BitVec.or_comm x y,BitVec.or_assoc]
  rw [horizontalOr2Zmm,shuffleEpi32_4e]
  change (((a 0 ||| a 1) ||| (a 4 ||| a 5)) ||| ((a 2 ||| a 3) ||| (a 6 ||| a 7)),
          ((b 0 ||| b 1) ||| (b 4 ||| b 5)) ||| ((b 2 ||| b 3) ||| (b 6 ||| b 7))) = _
  simp only [BitVec.or_assoc,flip]
theorem unpack512_pack512 (a : Vec512) : unpack512 (pack512 a) = a := by
  funext i
  by_cases h : i.val < 4
  · simp only [unpack512,pack512,h,↓reduceDIte,BitVec.extractLsb'_append_eq_right,
      unpack256_pack256,cast512Si256]
  · simp only [unpack512,pack512,h,↓reduceDIte,BitVec.extractLsb'_append_eq_left,
      unpack256_pack256,extractI64x4]
    congr 1
    apply Fin.ext
    simp only [show (1:Fin 2).val=1 from rfl,Nat.mul_one]
    omega

theorem pack512_unpack512 (a : BitVec 512) : pack512 (unpack512 a) = a := by
  have low : cast512Si256 (unpack512 a) = unpack256 (a.extractLsb' 0 256) := by
    funext i
    simp only [cast512Si256,unpack512,show i.val < 4 from i.isLt,↓reduceDIte]
  have high : extractI64x4 (unpack512 a) 1 = unpack256 (a.extractLsb' 256 256) := by
    funext i
    simp only [extractI64x4,unpack512,show (1:Fin 2).val=1 from rfl,Nat.mul_one,
      show ¬4+i.val < 4 by omega,↓reduceDIte]
    congr 1
    apply Fin.ext
    change 4+i.val-4=i.val
    omega
  rw [pack512,low,high,pack256_unpack256,pack256_unpack256]
  exact BitVec.extractLsb'_append_extractLsb'


end NeuralReversi.Simd.X86

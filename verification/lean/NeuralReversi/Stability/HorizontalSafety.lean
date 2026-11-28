module
public import NeuralReversi.Stability.Horizontal

@[expose] public section

namespace NeuralReversi.Stability

theorem horizontal_product_bound (p : Word) :
    (p &&& 0x0101010101010101).toNat * 255 < 2^64 := by
  have hm : (p &&& 0x0101010101010101).toNat ≤ 0x0101010101010101 := by
    exact Nat.and_le_right
  omega

theorem horizontal_multiply_safe (p : Word) :
    BitVec.umulOverflow (p &&& 0x0101010101010101) (255 : Word) = false := by
  have h := horizontal_product_bound p
  simp only [BitVec.umulOverflow, decide_eq_false_iff_not]
  change ¬ (p &&& 0x0101010101010101).toNat * 255 ≥ 2^64
  omega

end NeuralReversi.Stability

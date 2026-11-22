import NeuralReversi.Simd.PairModel
import NeuralReversi.Simd.X86Properties

namespace NeuralReversi.Simd.X86

theorem horizontalOr2U64_exact (a b : Vec256) :
    horizontalOr2U64 a b = (a 0 ||| a 1 ||| a 2 ||| a 3,b 0 ||| b 1 ||| b 2 ||| b 3) := by
  rw [horizontalOr2U64,shuffleEpi32_4e]
  change ((a 0 ||| a 1) ||| (a 2 ||| a 3),(b 0 ||| b 1) ||| (b 2 ||| b 3)) = _
  simp only [BitVec.or_assoc]

end NeuralReversi.Simd.X86

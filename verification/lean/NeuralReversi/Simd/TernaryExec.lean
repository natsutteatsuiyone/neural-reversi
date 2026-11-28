module
public import Lean.Compiler.CSimpAttr
public import NeuralReversi.Simd.TernaryFast

@[expose] public section

namespace NeuralReversi.Simd.X86

@[csimp] theorem ternaryWord_exec : ternaryWord = ternaryWordFast := by
  funext a b c imm
  exact (ternaryWordFast_exact a b c imm).symm

@[csimp] theorem ternarylogicEpi64_exec : ternarylogicEpi64 = ternarylogicEpi64Fast := by
  funext a b c imm i
  exact (ternaryWordFast_exact (a i) (b i) (c i) imm).symm

end NeuralReversi.Simd.X86

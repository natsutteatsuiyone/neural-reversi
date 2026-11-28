module
public import Lean.Compiler.CSimpAttr
public import NeuralReversi.Flip.ScalarProperties

@[expose] public section

namespace NeuralReversi.Flip.Scalar

@[csimp] theorem shiftCount32_exec : shiftCount32 = shiftCount32Fast := by
  funext x
  exact (shiftCount32Fast_eq x).symm

end NeuralReversi.Flip.Scalar

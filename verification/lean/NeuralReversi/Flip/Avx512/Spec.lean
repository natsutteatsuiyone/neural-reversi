module
public import NeuralReversi.Flip.Spec

@[expose] public section

namespace NeuralReversi.Flip.Avx512
open NeuralReversi.Flip

def IndexCoordinateBit (pos : BitVec 64) (player opponent : Word) (i : Nat) : Prop :=
  if h : pos.toNat < 64 then CoordinateBit ⟨pos.toNat,h⟩ player opponent i else False

end NeuralReversi.Flip.Avx512

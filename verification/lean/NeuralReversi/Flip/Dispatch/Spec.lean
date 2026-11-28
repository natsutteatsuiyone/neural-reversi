module
public import NeuralReversi.Flip.Dispatch.Types
public import NeuralReversi.Flip.Spec
public import NeuralReversi.Flip.Avx2.Spec

@[expose] public section

namespace NeuralReversi.Flip.Dispatch

def RawCoordinate (config : X86Config) (sq : Square) (player opponent : Word) (i : Nat) : Prop :=
  if config.avx512cd && config.avx512vl then CoordinateBit sq player opponent i
  else if config.avx2 then Avx2.RawSpec sq player opponent i
  else CoordinateBit sq player opponent i

end NeuralReversi.Flip.Dispatch

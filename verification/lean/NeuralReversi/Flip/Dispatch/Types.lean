module
public import NeuralReversi.Flip.Types

@[expose] public section

namespace NeuralReversi.Flip.Dispatch

structure X86Config where
  avx512cd : Bool
  avx512vl : Bool
  avx2 : Bool

inductive TargetConfig where
  | x86_64 (config : X86Config)
  | aarch64 (neon : Bool) (sha3 : Bool)
  | other

end NeuralReversi.Flip.Dispatch

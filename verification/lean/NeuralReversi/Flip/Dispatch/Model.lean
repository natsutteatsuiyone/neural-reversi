import NeuralReversi.Flip.Dispatch.Types
import NeuralReversi.Flip.Model
import NeuralReversi.Flip.Avx2.Model
import NeuralReversi.Flip.Avx512.Model
import NeuralReversi.Flip.Neon.Model
import NeuralReversi.Flip.WasmSimd.Model

namespace NeuralReversi.Flip.Dispatch

def flip (config : X86Config) (checks : Bool) (sq : Square) (player opponent : Word) :
    Except String Word :=
  if config.avx512cd && config.avx512vl then .ok (Avx512.flip sq player opponent)
  else if config.avx2 then .ok (Avx2.flip sq player opponent)
  else NeuralReversi.Flip.flip checks sq player opponent

def flipTarget (config : TargetConfig) (checks : Bool) (sq : Square) (player opponent : Word) :
    Except String Word :=
  match config with
  | .x86_64 x86 => flip x86 checks sq player opponent
  | .aarch64 neon sha3 =>
      if neon then .ok (Neon.flip sha3 sq player opponent)
      else NeuralReversi.Flip.flip checks sq player opponent
  | .wasm32 simd128 =>
      if simd128 then .ok (WasmSimd.flip sq player opponent)
      else NeuralReversi.Flip.flip checks sq player opponent
  | .other => NeuralReversi.Flip.flip checks sq player opponent

end NeuralReversi.Flip.Dispatch

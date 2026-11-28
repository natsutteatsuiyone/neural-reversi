module
public import NeuralReversi.Simd.X86Model

@[expose] public section

namespace NeuralReversi.Simd.X86

def unpackloEpi64 (a b : Vec256) : Vec256 := fun i =>
  match i.val with
  | 0 => a 0 | 1 => b 0 | 2 => a 2 | _ => b 2
def unpackhiEpi64 (a b : Vec256) : Vec256 := fun i =>
  match i.val with
  | 0 => a 1 | 1 => b 1 | 2 => a 3 | _ => b 3
def horizontalOr2U64 (a b : Vec256) : Word × Word :=
  let lo := unpackloEpi64 a b
  let hi := unpackhiEpi64 a b
  let pairs := or256 lo hi
  let m128 := or128 (cast256Si128 pairs) (extractI128 pairs 1)
  (cvtsi128Si64 m128,cvtsi128Si64 (shuffleEpi32 m128 0x4e))

end NeuralReversi.Simd.X86

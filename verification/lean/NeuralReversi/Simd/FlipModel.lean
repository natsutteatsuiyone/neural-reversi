import NeuralReversi.Simd.X86Model

namespace NeuralReversi.Simd.X86

def xor256 (a b : Vec256) : Vec256 := fun i => a i ^^^ b i

def andnot256 (a b : Vec256) : Vec256 := fun i => ~~~(a i) &&& b i

def cmpeqEpi64 (a b : Vec256) : Vec256 := fun i =>
  if a i = b i then BitVec.allOnes 64 else 0

def cmpgtEpi64 (a b : Vec256) : Vec256 := fun i =>
  if (b i).toInt < (a i).toInt then BitVec.allOnes 64 else 0

end NeuralReversi.Simd.X86

import NeuralReversi.Simd.FlipModel
import NeuralReversi.Simd.TernaryModel

namespace NeuralReversi.Simd.X86

def subEpi64 (a b : Vec256) : Vec256 := fun i => a i - b i

def lzcntEpi64 (a : Vec256) : Vec256 := fun i => (a i).clz

def rawSrlvEpi64 (a counts : Vec256) : Vec256 := fun i =>
  if (counts i).toNat < 64 then a i >>> (counts i).toNat else 0

def minEpi64 (a b : Vec256) : Vec256 := fun i =>
  if (a i).toInt ≤ (b i).toInt then a i else b i

end NeuralReversi.Simd.X86

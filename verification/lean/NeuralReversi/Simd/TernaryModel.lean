import NeuralReversi.Simd.X86Model

namespace NeuralReversi.Simd.X86

def truthIndex (a b c : Bool) : Nat := 4*a.toNat + 2*b.toNat + c.toNat

def ternaryBit (immediate : BitVec 8) (a b c : Bool) : Bool :=
  immediate.getLsbD (truthIndex a b c)

def wordFromBits (bits : Fin 64 → Bool) : Word :=
  (BitVec.ofBoolListLE (List.ofFn bits)).cast (by simp)

def ternaryWord (a b c : Word) (immediate : BitVec 8) : Word :=
  wordFromBits fun i => ternaryBit immediate (a.getLsbD i) (b.getLsbD i) (c.getLsbD i)

def ternarylogicEpi64 (a b c : Vec256) (immediate : BitVec 8) : Vec256 :=
  fun lane => ternaryWord (a lane) (b lane) (c lane) immediate

end NeuralReversi.Simd.X86

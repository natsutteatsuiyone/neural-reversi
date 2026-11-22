import NeuralReversi.Simd.ZmmModel
import NeuralReversi.Simd.Fixture

namespace NeuralReversi.Simd.X86

def zmmOutput (a : Vec512) : String :=
  String.intercalate "," ((List.ofFn a).map fun word => toString word.toNat)

def zmmPrimitiveOutput (a b : Vec512) : String :=
  let (left,right) := horizontalOr2Zmm a b
  String.intercalate "|" [
    zmmOutput (setEpi64 (a 0) (a 1) (a 2) (a 3) (a 4) (a 5) (a 6) (a 7)),
    zmmOutput (set1Epi64 (a 0)),zmmOutput (and512 a b),zmmOutput (or512 a b),
    zmmOutput (add512 a b),zmmOutput (rolv512 a b),
    wordsOutput (cast512Si256 a),wordsOutput (extractI64x4 a 1),
    zmmOutput (unpacklo512 a b),zmmOutput (unpackhi512 a b),
    toString left.toNat,toString right.toNat,
    zmmOutput (ternary512 a b (rolv512 a b) 0xf8)]

def zmmPrimitiveRequest (fields : List String) : Except String String := do
  let [_,a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7] := fields |
    throw "expected sixteen ZMM words"
  let words ← [a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7].mapM parseWord
  let [a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7] := words |
    throw "invalid ZMM word count"
  pure (zmmPrimitiveOutput (setEpi64 a7 a6 a5 a4 a3 a2 a1 a0)
    (setEpi64 b7 b6 b5 b4 b3 b2 b1 b0))

def zmmTernaryRequest (fields : List String) : Except String String := do
  let [_,imm,a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7,c0,c1,c2,c3,c4,c5,c6,c7] := fields |
    throw "expected IMM8 and twenty-four ZMM words"
  let some imm := imm.toNat? | throw "invalid IMM8"
  if imm ≥ 256 then throw "immediate outside IMM8 domain"
  let words ← [a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7,c0,c1,c2,c3,c4,c5,c6,c7].mapM parseWord
  let [a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7,c0,c1,c2,c3,c4,c5,c6,c7] := words |
    throw "invalid ZMM ternary word count"
  pure (zmmOutput (ternary512 (setEpi64 a7 a6 a5 a4 a3 a2 a1 a0)
    (setEpi64 b7 b6 b5 b4 b3 b2 b1 b0) (setEpi64 c7 c6 c5 c4 c3 c2 c1 c0) (BitVec.ofNat 8 imm)))


end NeuralReversi.Simd.X86

import NeuralReversi.Flip.Avx2.Model
import NeuralReversi.Simd.Fixture

namespace NeuralReversi.Flip.Avx2
open NeuralReversi.Simd.X86

def preparedRequest (fields : List String) : Except String String := do
  let [_,position,p0,p1,p2,p3,o0,o1,o2,o3] := fields |
    throw "expected position and eight independent AVX2 words"
  let pos ← parseWord position
  let words ← [p0,p1,p2,p3,o0,o1,o2,o3].mapM parseWord
  let [p0,p1,p2,p3,o0,o1,o2,o3] := words | throw "invalid prepared word count"
  let pp := setEpi64x p3 p2 p1 p0
  let oo := setEpi64x o3 o2 o1 o0
  if valid : pos.toNat < 66 then
    let result := mmFlipPrepared pp oo pos valid
    return s!"ok:{(result 0).toNat}|{(result 1).toNat}"
  else throw "prepared position outside 0..65"

def flipPrimitiveRequest (fields : List String) : Except String String := do
  let [_,a0,a1,a2,a3,b0,b1,b2,b3] := fields | throw "expected eight flip intrinsic words"
  let words ← [a0,a1,a2,a3,b0,b1,b2,b3].mapM parseWord
  let [a0,a1,a2,a3,b0,b1,b2,b3] := words | throw "invalid flip intrinsic word count"
  let a := setEpi64x a3 a2 a1 a0
  let b := setEpi64x b3 b2 b1 b0
  return String.intercalate "|" [wordsOutput (xor256 a b),wordsOutput (andnot256 a b),
    wordsOutput (cmpeqEpi64 a b),wordsOutput (cmpgtEpi64 a b)]

end NeuralReversi.Flip.Avx2

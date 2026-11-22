import NeuralReversi.Flip.Avx512.Model
import NeuralReversi.Simd.Fixture

namespace NeuralReversi.Flip.Avx512
open NeuralReversi.Simd.X86

def publicRequest (fields : List String) : Except String String := do
  let [_,player,opponent,reserved,square] := fields |
    throw "expected player opponent reserved-zero square"
  let player ← parseWord player
  let opponent ← parseWord opponent
  let reserved ← parseWord reserved
  if reserved != 0 then throw "AVX512 flip reserves the fourth field as zero"
  let some square := square.toNat? | throw "invalid unsigned square"
  if valid : square < 64 then
    return s!"ok:{(flip ⟨square,valid⟩ player opponent).toNat}"
  else throw "invalid square discriminant"

def indexRequest (fields : List String) : Except String String := do
  let [_,position,player,opponent] := fields |
    throw "expected position player opponent"
  let position ← parseWord position
  let player ← parseWord player
  let opponent ← parseWord opponent
  if valid : position.toNat < 66 then
    return s!"ok:{(flipIndex position valid player opponent).toNat}"
  else throw "AVX512 flip index outside 0..65"

def primitiveRequest (fields : List String) : Except String String := do
  let [_,a0,a1,a2,a3,b0,b1,b2,b3] := fields | throw "expected eight AVX512 flip words"
  let words ← [a0,a1,a2,a3,b0,b1,b2,b3].mapM parseWord
  let [a0,a1,a2,a3,b0,b1,b2,b3] := words | throw "invalid AVX512 flip word count"
  let a := setEpi64x a3 a2 a1 a0
  let b := setEpi64x b3 b2 b1 b0
  return String.intercalate "|" [wordsOutput (lzcntEpi64 a),
    wordsOutput (rawSrlvEpi64 a b),wordsOutput (minEpi64 a b),wordsOutput (subEpi64 a b)]

end NeuralReversi.Flip.Avx512

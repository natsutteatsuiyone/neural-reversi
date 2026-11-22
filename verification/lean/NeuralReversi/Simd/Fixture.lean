import NeuralReversi.Simd.X86Model
import NeuralReversi.Simd.TernaryExec
import NeuralReversi.Simd.PairModel

namespace NeuralReversi.Simd.X86

def wordsOutput (a : Vec256) : String :=
  String.intercalate "," ([a 0,a 1,a 2,a 3].map fun word => toString word.toNat)

def halfOutput (a : Vec128) : String := s!"{(a 0).toNat},{(a 1).toNat}"

def primitiveOutput (a b : Vec256) : String :=
  let low := cast256Si128 a
  let high := extractI128 a 1
  let m128 := or128 low high
  let swapped := shuffleEpi32 m128 0x4e
  String.intercalate "|" [
    wordsOutput (setEpi64x (a 0) (a 1) (a 2) (a 3)),
    wordsOutput (set1Epi64x (a 0)),
    wordsOutput (and256 a b), wordsOutput (or256 a b), wordsOutput (addEpi64 a b),
    wordsOutput (sllvEpi64 a b), wordsOutput (srlvEpi64 a b),
    halfOutput low, halfOutput high, halfOutput swapped,
    toString (horizontalOrU64 a).toNat, toString (cvtsi128Si64 swapped).toNat]

def parseWord (text : String) : Except String Word := do
  let some value := text.toNat? | throw "invalid intrinsic word"
  if value ≥ 2^64 then throw "intrinsic word outside u64 domain"
  pure (BitVec.ofNat 64 value)

def pairReductionOutput (a b : Vec256) : String :=
  let (left,right) := horizontalOr2U64 a b
  s!"{wordsOutput (unpackloEpi64 a b)}|{wordsOutput (unpackhiEpi64 a b)}|{left.toNat}|{right.toNat}"

def primitiveRequest (fields : List String) : Except String String := do
  let [_,a0,a1,a2,a3,b0,b1,b2,b3] := fields | throw "expected eight intrinsic words"
  let a0 ← parseWord a0
  let a1 ← parseWord a1
  let a2 ← parseWord a2
  let a3 ← parseWord a3
  let b0 ← parseWord b0
  let b1 ← parseWord b1
  let b2 ← parseWord b2
  let b3 ← parseWord b3
  let a := setEpi64x a3 a2 a1 a0
  let b := setEpi64x b3 b2 b1 b0
  pure (if fields.head? == some "avx2-pair-reduction" then pairReductionOutput a b
    else primitiveOutput a b)

def ternaryRequest (fields : List String) : Except String String := do
  let [_,immediate,a0,a1,a2,a3,b0,b1,b2,b3,c0,c1,c2,c3] := fields |
    throw "expected IMM8 and twelve ternary words"
  let some immediate := immediate.toNat? | throw "invalid IMM8"
  if immediate ≥ 256 then throw "immediate outside IMM8 domain"
  let words ← [a0,a1,a2,a3,b0,b1,b2,b3,c0,c1,c2,c3].mapM parseWord
  let [a0,a1,a2,a3,b0,b1,b2,b3,c0,c1,c2,c3] := words | throw "invalid ternary word count"
  pure (wordsOutput (ternarylogicEpi64 (setEpi64x a3 a2 a1 a0)
    (setEpi64x b3 b2 b1 b0) (setEpi64x c3 c2 c1 c0) (BitVec.ofNat 8 immediate)))

end NeuralReversi.Simd.X86

import Std

namespace NeuralReversi.Flip.Scalar

abbrev Word := BitVec 64

def leadingZeros32 (x : Word) : BitVec 32 := BitVec.ofNat 32 x.clz.toNat

def shiftCount32 (x : Word) : BitVec 32 := 64#32 - 1#32 - leadingZeros32 x

def flipLeft (mask player notOpponent : Word) : Word :=
  let x := notOpponent &&& mask
  let flank := x &&& -x
  if flank &&& player ≠ 0 then mask &&& (flank - 1#64) else 0

def checkedSub32 (checks : Bool) (a b : BitVec 32) : Except String (BitVec 32) :=
  if checks ∧ a.toNat < b.toNat then .error "attempt to subtract with overflow"
  else .ok (a-b)

def rustShl64 (checks : Bool) (a : Word) (count : BitVec 32) : Except String Word :=
  if checks ∧ 64 ≤ count.toNat then .error "attempt to shift left with overflow"
  else .ok (a <<< (count.toNat % 64))

def flipRight (checks : Bool) (mask player notOpponent : Word) : Except String Word :=
  let x := notOpponent &&& mask
  if x = 0 then .ok 0
  else do
    let high ← checkedSub32 checks 64#32 1#32
    let count ← checkedSub32 checks high (leadingZeros32 x)
    let flank ← rustShl64 checks 1#64 count
    pure (if flank &&& player ≠ 0 then mask &&& ~~~(flank ||| (flank-1#64)) else 0)


end NeuralReversi.Flip.Scalar

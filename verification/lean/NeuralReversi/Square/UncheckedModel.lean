import NeuralReversi.Square.Model

namespace NeuralReversi.Square

inductive UncheckedResult where
  | value (square : Square)
  | panic (input : Nat)
  | invalidDiscriminant (byte : Byte)
  deriving DecidableEq, Repr

def UncheckedResult.message (input : Nat) : String :=
  s!"Index out of bounds for Square enum. index: {input}"

def transmuteByte (value : Byte) : UncheckedResult :=
  if valid : value.toNat < 64 then .value ⟨value.toNat, valid⟩
  else .invalidDiscriminant value

def fromU8Unchecked (checked : Bool) (value : Byte) : UncheckedResult :=
  if checked && decide (64 ≤ value.toNat) then .panic value.toNat
  else transmuteByte value

def fromU32Unchecked (checked : Bool) (value : BitVec 32) : UncheckedResult :=
  if checked && decide (64 ≤ value.toNat) then .panic value.toNat
  else fromU8Unchecked checked (value.setWidth 8)

def fromUsizeUnchecked64 (checked : Bool) (value : BitVec 64) : UncheckedResult :=
  if checked && decide (64 ≤ value.toNat) then .panic value.toNat
  else fromU8Unchecked checked (value.setWidth 8)

end NeuralReversi.Square

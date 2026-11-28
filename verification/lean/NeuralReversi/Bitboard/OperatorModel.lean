module
public import NeuralReversi.Bitboard.Model

@[expose] public section

namespace NeuralReversi.Bitboard

def bitAnd (lhs rhs : Bits) : Bits := lhs &&& rhs
def bitOr (lhs rhs : Bits) : Bits := lhs ||| rhs
def bitXor (lhs rhs : Bits) : Bits := lhs ^^^ rhs
def bitNot (bits : Bits) : Bits := ~~~bits

def bitAndAssign (state rhs : Bits) : Unit × Bits := ((), state &&& rhs)
def bitOrAssign (state rhs : Bits) : Unit × Bits := ((), state ||| rhs)
def bitXorAssign (state rhs : Bits) : Unit × Bits := ((), state ^^^ rhs)

def shiftLeft (checked : Bool) (bits : Bits) (count : BitVec 32) : Except String Bits :=
  if checked && decide (64 ≤ count.toNat) then .error "attempt to shift left with overflow"
  else .ok (bits <<< (count.toNat % 64))

def shiftRight (checked : Bool) (bits : Bits) (count : BitVec 32) : Except String Bits :=
  if checked && decide (64 ≤ count.toNat) then .error "attempt to shift right with overflow"
  else .ok (bits >>> (count.toNat % 64))

def shiftLeftAssign (checked : Bool) (bits : Bits) (count : BitVec 32) : Except String Unit × Bits :=
  match shiftLeft checked bits count with
  | .ok result => (.ok (), result)
  | .error reason => (.error reason, bits)

def shiftRightAssign (checked : Bool) (bits : Bits) (count : BitVec 32) : Except String Unit × Bits :=
  match shiftRight checked bits count with
  | .ok result => (.ok (), result)
  | .error reason => (.error reason, bits)

def leftShiftBit (bits : Bits) (count i : Nat) : Bool :=
  if count ≤ i then bits.getLsbD (i - count) else false

def rightShiftBit (bits : Bits) (count i : Nat) : Bool :=
  bits.getLsbD (i + count)

end NeuralReversi.Bitboard

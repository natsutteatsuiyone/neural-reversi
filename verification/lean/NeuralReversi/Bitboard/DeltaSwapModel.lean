module
public import NeuralReversi.Bitboard.GeometryModel

@[expose] public section

namespace NeuralReversi.Bitboard

def deltaSwap (checked : Bool) (bits mask : Bits) (delta : BitVec 32) : Except String Bits :=
  if checked && decide (64 ≤ delta.toNat) then .error "attempt to shift left with overflow"
  else .ok (deltaSwapRaw bits mask (delta.toNat % 64))

def deltaShiftedBit (bits : Bits) (shift i : Nat) : Bool :=
  if i < 64 ∧ shift ≤ i then bits.getLsbD (i - shift) else false

def deltaTemporaryBit (bits mask : Bits) (shift i : Nat) : Bool :=
  mask.getLsbD i && (bits.getLsbD i ^^ deltaShiftedBit bits shift i)

def deltaOutputBit (bits mask : Bits) (shift i : Nat) : Bool :=
  bits.getLsbD i ^^ deltaTemporaryBit bits mask shift i ^^
    deltaTemporaryBit bits mask shift (shift + i)

end NeuralReversi.Bitboard

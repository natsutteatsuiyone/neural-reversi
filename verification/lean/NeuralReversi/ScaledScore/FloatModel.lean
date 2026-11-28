module
public import NeuralReversi.ScaledScore.Model

@[expose] public section

namespace NeuralReversi.ScaledScore

abbrev FloatBits := BitVec 32

def packF32 (negative : Bool) (exponent fraction : Nat) : FloatBits :=
  BitVec.ofBool negative ++ (BitVec.ofNat 8 exponent ++ BitVec.ofNat 23 fraction)

def f32Sign (bits : FloatBits) : Bool := bits.msb
def f32Exponent (bits : FloatBits) : Nat := (bits.extractLsb' 23 8).toNat
def f32Fraction (bits : FloatBits) : Nat := (bits.extractLsb' 0 23).toNat

def precisionShift (magnitude : Nat) : Nat := magnitude.log2 - 23

def roundEvenQuotient (magnitude shift : Nat) : Nat :=
  let unit := 2 ^ shift
  let quotient := magnitude / unit
  let remainder := magnitude % unit
  if unit < 2 * remainder ∨ (unit = 2 * remainder ∧ quotient % 2 = 1) then
    quotient + 1
  else quotient

def integerCastBits (raw : Word) : FloatBits :=
  let negative := decide (raw.toInt < 0)
  let magnitude := raw.toInt.natAbs
  if magnitude = 0 then 0#32
  else
    let exponent := magnitude.log2
    if exponent ≤ 23 then
      packF32 negative (exponent + 127) (magnitude * 2 ^ (23 - exponent) - 2 ^ 23)
    else
      let significand := roundEvenQuotient magnitude (precisionShift magnitude)
      if 2 ^ 24 ≤ significand then packF32 negative (exponent + 128) 0
      else packF32 negative (exponent + 127) (significand - 2 ^ 23)

def scaleIntegerCastBy256 (bits : FloatBits) : FloatBits :=
  if bits = 0#32 then 0#32
  else packF32 (f32Sign bits) (f32Exponent bits - 8) (f32Fraction bits)

def toDiscDiffF32 (score : ScaledScore) : FloatBits :=
  scaleIntegerCastBy256 (integerCastBits score.raw)

end NeuralReversi.ScaledScore

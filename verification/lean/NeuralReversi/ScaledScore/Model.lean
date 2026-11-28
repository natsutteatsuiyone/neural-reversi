module

@[expose] public section

namespace NeuralReversi.ScaledScore

abbrev Word := BitVec 32

structure ScaledScore where
  raw : Word
  deriving DecidableEq, Repr

def value (score : ScaledScore) : Word := score.raw

def fromRaw (checked : Bool) (raw : Word) : Except String ScaledScore :=
  if checked && decide (raw.toInt < -32767 ∨ raw.toInt > 32767) then
    .error "assertion failed: raw_value >= -Self::INF.0 && raw_value <= Self::INF.0"
  else .ok ⟨raw⟩

def fromDiscDiff (checked : Bool) (discs : Word) : Except String ScaledScore :=
  if checked && decide (discs.toInt < -127 ∨ discs.toInt > 127) then
    .error "assertion failed: disc_diff >= -SCORE_INF && disc_diff <= SCORE_INF"
  else .ok ⟨discs <<< 8⟩

def toDiscDiff (score : ScaledScore) : Word := score.raw.sshiftRight 8

end NeuralReversi.ScaledScore

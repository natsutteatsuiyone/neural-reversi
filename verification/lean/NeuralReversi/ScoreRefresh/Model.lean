module
public import NeuralReversi.ScaledScore.ArithmeticModel
public import NeuralReversi.ScaledScore.FloatModel
public import NeuralReversi.ScaledScore.DisplayModel

@[expose] public section

namespace NeuralReversi.ScoreRefresh
abbrev Word := BitVec 32

structure DiscScore where
  raw : Word
  deriving DecidableEq, Repr

structure Score where
  raw : Word
  deriving DecidableEq, Repr

namespace DiscScore
def new (raw : Word) : DiscScore := ⟨raw⟩
def value (score : DiscScore) : Word := score.raw

def negate (checked : Bool) (a : DiscScore) : Except String DiscScore :=
  if checked && a.raw.negOverflow then .error "attempt to negate with overflow"
  else .ok ⟨-a.raw⟩
def addInt (checked : Bool) (a : DiscScore) (b : Word) : Except String DiscScore :=
  if checked && a.raw.saddOverflow b then .error "attempt to add with overflow"
  else .ok ⟨a.raw + b⟩
def subInt (checked : Bool) (a : DiscScore) (b : Word) : Except String DiscScore :=
  if checked && a.raw.ssubOverflow b then .error "attempt to subtract with overflow"
  else .ok ⟨a.raw - b⟩

def lt (a b : DiscScore) : Bool := decide (a.raw.toInt < b.raw.toInt)
def le (a b : DiscScore) : Bool := decide (a.raw.toInt ≤ b.raw.toInt)

def display {Result : Type} (score : DiscScore) (options : ScaledScore.FormatOptions)
    (formatI32 : Word → ScaledScore.FormatOptions → Result) : Result :=
  formatI32 score.raw options
end DiscScore

def discMax : DiscScore := ⟨64⟩
def discMin : DiscScore := ⟨-64⟩
def discInf : DiscScore := ⟨127⟩
def scaleBits : Nat := 8
def scale : Word := 256
def scoreZero : Score := ⟨0⟩
def scoreMax : Score := ⟨discMax.raw <<< 8⟩
def scoreMin : Score := ⟨discMin.raw <<< 8⟩
def scoreInf : Score := ⟨32767⟩

def value (score : Score) : Word := score.raw

def rawAssertion : String :=
  "assertion failed: raw_value >= -Self::INF.0 && raw_value <= Self::INF.0"
def discAssertion : String :=
  "assertion failed: disc_diff.value() >= -DISC_SCORE_INF.value() &&\n    disc_diff.value() <= DISC_SCORE_INF.value()"

def fromRaw (checked : Bool) (raw : Word) : Except String Score :=
  if checked && decide (raw.toInt < -32767 ∨ raw.toInt > 32767) then .error rawAssertion
  else .ok ⟨raw⟩

def fromDiscDiff (checked : Bool) (discs : DiscScore) : Except String Score :=
  if checked && decide (discs.raw.toInt < -127 ∨ discs.raw.toInt > 127) then .error discAssertion
  else .ok ⟨discs.raw <<< 8⟩

def toDiscDiff (score : Score) : DiscScore := ⟨score.raw.sshiftRight 8⟩

def legacy (score : Score) : ScaledScore.ScaledScore := ⟨score.raw⟩
def ofLegacy (score : ScaledScore.ScaledScore) : Score := ⟨score.raw⟩

def toDiscDiffF32 (score : Score) : ScaledScore.FloatBits :=
  ScaledScore.scaleIntegerCastBy256 (ScaledScore.integerCastBits score.raw)

def display {State Payload : Type} (score : Score) (options : ScaledScore.FormatOptions)
    (write : State → String → ScaledScore.WriteResult State Payload) (state : State) :
    ScaledScore.WriteResult State Payload × ScaledScore.FormatOptions :=
  ScaledScore.display score.raw options write state

def addWords (checked : Bool) (a b : Word) : Except String Score :=
  if checked && a.saddOverflow b then .error "attempt to add with overflow"
  else .ok ⟨a + b⟩

def subWords (checked : Bool) (a b : Word) : Except String Score :=
  if checked && a.ssubOverflow b then .error "attempt to subtract with overflow"
  else .ok ⟨a - b⟩

def addScore (checked : Bool) (a b : Score) : Except String Score :=
  addWords checked a.raw b.raw

def subScore (checked : Bool) (a b : Score) : Except String Score :=
  subWords checked a.raw b.raw

def negate (checked : Bool) (a : Score) : Except String Score :=
  if checked && a.raw.negOverflow then .error "attempt to negate with overflow"
  else .ok ⟨-a.raw⟩

def addInt (checked : Bool) (a : Score) (b : Word) : Except String Score :=
  addWords checked a.raw b

def subInt (checked : Bool) (a : Score) (b : Word) : Except String Score :=
  subWords checked a.raw b

def divInt (_checked : Bool) (a : Score) (b : Word) : Except String Score :=
  if b = 0#32 then .error "attempt to divide by zero"
  else if a.raw = 0x80000000#32 ∧ b = 0xffffffff#32 then
    .error "attempt to divide with overflow"
  else .ok ⟨a.raw.sdiv b⟩

def assignResult (before : Score) (outcome : Except String Score) :
    Except String Unit × Score :=
  match outcome with
  | .ok after => (.ok (), after)
  | .error reason => (.error reason, before)

def addAssign (checked : Bool) (before rhs : Score) : Except String Unit × Score :=
  assignResult before (addScore checked before rhs)

def subAssign (checked : Bool) (before rhs : Score) : Except String Unit × Score :=
  assignResult before (subScore checked before rhs)


end NeuralReversi.ScoreRefresh

module
public import NeuralReversi.ScaledScore.Model

@[expose] public section

namespace NeuralReversi.ScaledScore

def addWords (checked : Bool) (a b : Word) : Except String ScaledScore :=
  if checked && a.saddOverflow b then .error "attempt to add with overflow"
  else .ok ⟨a + b⟩

def subWords (checked : Bool) (a b : Word) : Except String ScaledScore :=
  if checked && a.ssubOverflow b then .error "attempt to subtract with overflow"
  else .ok ⟨a - b⟩

def addScore (checked : Bool) (a b : ScaledScore) : Except String ScaledScore :=
  addWords checked a.raw b.raw

def subScore (checked : Bool) (a b : ScaledScore) : Except String ScaledScore :=
  subWords checked a.raw b.raw

def negate (checked : Bool) (a : ScaledScore) : Except String ScaledScore :=
  if checked && a.raw.negOverflow then .error "attempt to negate with overflow"
  else .ok ⟨-a.raw⟩

def addInt (checked : Bool) (a : ScaledScore) (b : Word) : Except String ScaledScore :=
  addWords checked a.raw b

def subInt (checked : Bool) (a : ScaledScore) (b : Word) : Except String ScaledScore :=
  subWords checked a.raw b

def divInt (_checked : Bool) (a : ScaledScore) (b : Word) : Except String ScaledScore :=
  if b = 0#32 then .error "attempt to divide by zero"
  else if a.raw = 0x80000000#32 ∧ b = 0xffffffff#32 then
    .error "attempt to divide with overflow"
  else .ok ⟨a.raw.sdiv b⟩

def assignResult (before : ScaledScore) (outcome : Except String ScaledScore) :
    Except String Unit × ScaledScore :=
  match outcome with
  | .ok after => (.ok (), after)
  | .error reason => (.error reason, before)

def addAssign (checked : Bool) (before rhs : ScaledScore) : Except String Unit × ScaledScore :=
  assignResult before (addScore checked before rhs)

def subAssign (checked : Bool) (before rhs : ScaledScore) : Except String Unit × ScaledScore :=
  assignResult before (subScore checked before rhs)

end NeuralReversi.ScaledScore

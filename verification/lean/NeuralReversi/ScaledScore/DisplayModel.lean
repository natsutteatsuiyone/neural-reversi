import NeuralReversi.ScaledScore.FloatRounding

namespace NeuralReversi.ScaledScore

structure FormatOptions where
  fill : Char := ' '
  alignment : Option (Fin 3) := none
  sign : Option Bool := none
  alternate : Bool := false
  zeroPad : Bool := false
  debugHex : Option Bool := none
  width : Option (BitVec 16) := none
  precision : Option (BitVec 16) := none
  deriving DecidableEq

def displayCents (raw : Word) : Nat :=
  roundEvenQuotient (roundedMagnitude raw.toInt.natAbs * 25) 6

def twoDigits (n : Nat) : String :=
  if n < 10 then "0" ++ n.repr else n.repr

def magnitudeChunks (cents : Nat) : List String :=
  if cents = 0 then ["0.", "00"]
  else if cents < 10 then ["0.", "0", cents.repr]
  else if cents < 100 then ["0.", cents.repr]
  else [(cents / 100).repr, ".", twoDigits (cents % 100)]

def displayChunks (raw : Word) : List String :=
  (if raw.toInt < 0 then ["-"] else []) ++ magnitudeChunks (displayCents raw)

inductive WriteResult (State Payload : Type) where
  | ok (state : State)
  | error (state : State)
  | panic (payload : Payload) (state : State)
  deriving DecidableEq, Repr

def writeChunks {State Payload : Type} (write : State → String → WriteResult State Payload)
    (state : State) : List String → WriteResult State Payload
  | [] => .ok state
  | chunk :: rest => match write state chunk with
    | .ok next => writeChunks write next rest
    | .error next => .error next
    | .panic payload next => .panic payload next

def display {State Payload : Type} (raw : Word) (options : FormatOptions)
    (write : State → String → WriteResult State Payload) (state : State) :
    WriteResult State Payload × FormatOptions :=
  (writeChunks write state (displayChunks raw), options)

end NeuralReversi.ScaledScore

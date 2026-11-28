module
public import NeuralReversi.EndgameTiming
public import NeuralReversi.PvsMoveLoop

@[expose] public section

namespace NeuralReversi.EndgameProbCut

open PvsMoveLoop (scoreMax)

structure Thresholds where
  verification : Int
  screen : Int
  deriving DecidableEq

def clamp (lo hi n : Int) : Int :=
  min hi (max lo n)

def satI32 (n : Int) : Int :=
  clamp (-2147483648) 2147483647 n

def roundedUnits (up : Bool) (units : Int) : Int :=
  let quantum : Int := EndgameTiming.quantum
  satI32 (if up then -((-units) / quantum) else units / quantum)

def roundedMargin (up : Bool) (bits : BitVec 64) : Int :=
  let magnitude := bits &&& 0x7fffffffffffffff#64
  match EndgameTiming.decode magnitude with
  | some (.finite units) => roundedUnits up (if bits.msb then -(Int.ofNat units) else Int.ofNat units)
  | some .infinity => if bits.msb then -2147483648 else 2147483647
  | some .nan | none => 0

def thresholds (beta verification screen : Int) : Thresholds :=
  ⟨min scoreMax (satI32 (beta + verification)), clamp (-scoreMax) scoreMax (satI32 (beta + screen))⟩

def nativeThresholds (beta : Int) (verification screen : BitVec 64) : Thresholds :=
  thresholds beta (roundedMargin true verification) (roundedMargin false screen)

end NeuralReversi.EndgameProbCut

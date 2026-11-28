module
public import NeuralReversi.PvsParallel.RootHeaderMir.Program

@[expose] public section

namespace NeuralReversi.PvsParallel.GeneralOwnerMir

def minSplit : Nat := 12
def maxSpeculative : Nat := 24

def canSplit (idx count depth : Nat) (cut available : Bool) : Bool :=
  (decide (1 ≤ idx) || (!cut && decide (depth ≤ maxSpeculative))) &&
    decide (minSplit ≤ depth) && decide (2 ≤ count - idx) && available

def header (idx count fresh depth : Nat) (cut available busy : Bool) (scout : Int) : HeaderAction :=
  if idx < count then
    if canSplit idx count depth cut available then .split
    else if busy && decide (idx ≠ 0) && decide (idx + 1 < fresh) then .postpone idx (fresh - 1)
    else .child (idx + 1) true false (-scout)
  else .finish

def after (alpha best score : Int) : Int × Bool × Bool :=
  if best < score then
    if alpha < score then (score, true, true) else (score, false, false)
  else (best, false, false)

end NeuralReversi.PvsParallel.GeneralOwnerMir

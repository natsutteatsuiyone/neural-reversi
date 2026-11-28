module
public import NeuralReversi.Threading.AtomicBitSet

@[expose] public section

namespace NeuralReversi.PvsParallel.SplitRebookingMir

def cap : Nat := 4

def nested (pool depth : Nat) (opened : Bool) (count : Nat) : Bool :=
  (if decide (pool ≤ 1) then false else (if opened then (if decide (pool > cap) then (if decide (count = cap) then decide (depth < 8) else false) else false) else decide (depth < 8)))

def room (late : Bool) (count : Nat) : Bool :=
  if late then decide (count < cap) else decide (count < cap)

end NeuralReversi.PvsParallel.SplitRebookingMir

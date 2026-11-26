import NeuralReversi.MoveList.Concurrent
import NeuralReversi.Move.Model

namespace NeuralReversi.PvsParallel.NativeWorkerFeed

def site (pv : Bool) : Nat × Nat := if pv then (37, 1) else (26, 0)

def ticket (rc : List Move.Move) (old : BitVec w) : Option (Move.Move × Nat) := do
  let (index, ordinal) ← MoveList.Concurrent.next rc.length old
  let mv ← rc[index]?
  return (mv,ordinal.toNat)

end NeuralReversi.PvsParallel.NativeWorkerFeed

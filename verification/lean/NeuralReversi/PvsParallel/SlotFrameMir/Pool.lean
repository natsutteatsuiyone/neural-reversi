import NeuralReversi.PvsParallel.SlotFrameMir.Syntax

namespace NeuralReversi.PvsParallel.SlotFrameMir

def poolIndices (size : Nat) : List Nat := 0 :: List.range' 1 (size-1)
def lateTarget (worker : Worker) : Worker := worker

end NeuralReversi.PvsParallel.SlotFrameMir

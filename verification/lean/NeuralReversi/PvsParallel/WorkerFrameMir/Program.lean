import NeuralReversi.PvsParallel.WorkerFrameMir.Syntax

namespace NeuralReversi.PvsParallel.WorkerFrameMir

def callers : Kind → Option (List Kind)
  | .abandon => some [.other]
  | .detach => some [.abandon, .work]
  | .helper => some [.idle]
  | .helperDrop => some [.helper]
  | .idle => some [.split, .workerEntry]
  | .mainEntry => some []
  | .mainLoop => some [.mainEntry]
  | .split => some [.other]
  | .work => some [.idle]
  | .workerEntry => some []
  | _ => none

def searchingStores : List (Kind × Bool) := [
  (.book, true),
  (.bookJob, true),
  (.detach, false),
  (.finalize, true),
  (.helperDrop, false),
  (.mainLoop, false),
  (.mainLoop, true)
]

def failedStores : List Kind := [.abandon, .helperDrop]

def joinCheck (searching failed : Bool) : Bool := !searching && !failed

def canJoin (searching failed hasSlot deepest tested : Bool) : Bool :=
  joinCheck searching failed && (!hasSlot || (deepest && tested))

end NeuralReversi.PvsParallel.WorkerFrameMir

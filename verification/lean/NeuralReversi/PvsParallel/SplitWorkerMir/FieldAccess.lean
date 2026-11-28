module
public import NeuralReversi.PvsParallel.SplitWorkerMir.Syntax

@[expose] public section

namespace NeuralReversi.PvsParallel.SplitWorkerMir.FieldStores

inductive Access where
  | readAlpha (value : Word)
  | readBest (value : Word)
  | writeAlpha (value : Word)
  | writeBest (value : Word)
  | writeMove (index : Nat)
  deriving DecidableEq

end NeuralReversi.PvsParallel.SplitWorkerMir.FieldStores

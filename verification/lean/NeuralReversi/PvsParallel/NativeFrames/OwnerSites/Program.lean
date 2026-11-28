module
public import NeuralReversi.EndgameSearch.CacheOwner.Program

@[expose] public section

namespace NeuralReversi.PvsParallel.NativeFrames.OwnerSites

inductive Role where
  | passChild
  | presearch
  | scout
  | reducedRetry
  | wide
  deriving DecidableEq

def Role.site : Role → Nat
  | .passChild => 2
  | .presearch => 5
  | .scout => 7
  | .reducedRetry => 8
  | .wide => 9

def Role.block : Role → Nat
  | .passChild => 59
  | .presearch => 257
  | .scout => 356
  | .reducedRetry => 365
  | .wide => 382

end NeuralReversi.PvsParallel.NativeFrames.OwnerSites

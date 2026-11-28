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
  | .presearch => 4
  | .scout => 6
  | .reducedRetry => 7
  | .wide => 8

def Role.block : Role → Nat
  | .passChild => 59
  | .presearch => 244
  | .scout => 343
  | .reducedRetry => 352
  | .wide => 369

end NeuralReversi.PvsParallel.NativeFrames.OwnerSites

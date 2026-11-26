import NeuralReversi.EndgameSearch.CacheOwner.Program

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
  | .passChild => 60
  | .presearch => 246
  | .scout => 345
  | .reducedRetry => 354
  | .wide => 371

end NeuralReversi.PvsParallel.NativeFrames.OwnerSites

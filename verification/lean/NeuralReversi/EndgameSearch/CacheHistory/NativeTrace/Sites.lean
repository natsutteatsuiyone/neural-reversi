import NeuralReversi.EndgameSearch.CacheHistory.Boundary
import NeuralReversi.EndgameSearch.CacheOwner.Syntax

namespace NeuralReversi.EndgameSearch.CacheHistory.NativeTrace

def expectedSites : Arm → Option (List (Nat × CacheOwner.Node))
  | .failed => none
  | .table _ => some []
  | .fixed _ => some [(0,.access)]
  | .shallow => some [(0,.access),(3,.shallow)]
  | .deep poll => some [(0,.access),if poll then (1,.deep) else (2,.deep)]

end NeuralReversi.EndgameSearch.CacheHistory.NativeTrace

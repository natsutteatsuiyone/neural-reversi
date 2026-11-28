module
public import NeuralReversi.EndgameSearch.CacheHistory.Boundary
public import NeuralReversi.EndgameSearch.CacheOwner.Syntax

@[expose] public section

namespace NeuralReversi.EndgameSearch.CacheHistory.NativeTrace

def expectedSites : Arm → Option (List (Nat × CacheOwner.Node))
  | .failed => none
  | .table _ => some []
  | .fixed _ => some [(0,.access)]
  | .shallow six => some [(0,.access),if six then (4,.shallow) else (3,.shallow)]
  | .deep poll => some [(0,.access),if poll then (1,.deep) else (2,.deep)]

end NeuralReversi.EndgameSearch.CacheHistory.NativeTrace

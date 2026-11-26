import NeuralReversi.EndgameSearch.CacheMir

namespace NeuralReversi.EndgameSearch.CacheMir

def budget (n : Nat) : EndgameTreeCache.Table → Nat
  | .deep => (max (2097152 / (max n 1)) 131072)
  | .shallow => (min (max (262144 / (max n 1)) 32768) 131072)

end NeuralReversi.EndgameSearch.CacheMir

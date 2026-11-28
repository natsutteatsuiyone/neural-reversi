module

@[expose] public section

namespace NeuralReversi.PvsParallel

inductive RootEntry where
  | wipeout
  | header (sort : Bool) (idx fresh : Nat) (best : Int)
deriving DecidableEq, Repr

def rootEntry (count : Nat) (cacheHit : Bool) (_ : 0 < count) : RootEntry :=
  if cacheHit then .wipeout else .header (decide (1 < count)) 0 count (-32767)

def rootEpilogue (pvIdx : Nat) (best : Int) : Option Int × Int :=
  (if pvIdx = 0 then some best else none, best)

end NeuralReversi.PvsParallel

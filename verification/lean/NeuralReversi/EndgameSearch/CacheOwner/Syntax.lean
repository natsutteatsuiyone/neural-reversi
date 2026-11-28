module

@[expose] public section

namespace NeuralReversi.EndgameSearch.CacheOwner

inductive EntryPoint where
  | main | worker
  deriving DecidableEq

inductive PairSource where
  | fresh | shared
  deriving DecidableEq

inductive Handle where
  | newArc | returned | captured | foreign
  deriving DecidableEq

structure Boot where
  pair : PairSource
  registered : Handle
  returned : Handle
  capture : Handle
  enter : Handle
  kind : EntryPoint

inductive Node where
  | main | idle | job | active | dispatch | split | root | endRoot | solve | endWindow | midRoot
  | iterate | midWindow | midClosure | midProb | helper | helperClosure | pvs | ecCutoff | splitSearch
  | midStrategyProb | boundary | access | progress | failLow | deep | deepChild | shallow | shallowMoves
  | shallowChild | deepSlot | deepProbe | deepStore | shallowSlot | shallowProbe | shallowStore
  deriving DecidableEq

inductive Flow where
  | forward | borrowPair | borrowShallow | deepField | shallowField | callback | foreign
  deriving DecidableEq

structure Edge where
  caller : Node
  site : Nat
  callee : Node
  flow : Flow
  deriving DecidableEq

end NeuralReversi.EndgameSearch.CacheOwner


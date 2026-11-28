module
public import NeuralReversi.TranspositionTable.Snapshot

@[expose] public section

namespace NeuralReversi.TranspositionTable.ReaderMir

inductive Ordering where
  | relaxed | acquire
  deriving DecidableEq

inductive AtomicAction where
  | load (location : SeqLock.Location) (mode : Ordering)
  | fence (mode : Ordering)
  deriving DecidableEq

structure Outcome where
  actions : List AtomicAction
  returned : Option SeqLock.Snapshot

end NeuralReversi.TranspositionTable.ReaderMir

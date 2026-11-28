module
public import NeuralReversi.TranspositionTable.ProbeRetry

@[expose] public section

namespace NeuralReversi.TranspositionTable.ReaderMir

inductive ProbeEffect where
  | read (slot : Fin 2)
  | generation
  | spin
  deriving DecidableEq, Repr

inductive ProbeExit where
  | done (result : ProbeRetry.Result)
  | retry (lastStable : Option (Fin 2))

structure Pass where
  effects : List ProbeEffect
  exit : ProbeExit

end NeuralReversi.TranspositionTable.ReaderMir

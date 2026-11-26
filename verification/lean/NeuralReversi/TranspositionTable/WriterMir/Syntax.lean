import NeuralReversi.TranspositionTable.Save
import NeuralReversi.TranspositionTable.WriterTrace

namespace NeuralReversi.TranspositionTable.WriterMir

open SeqLock (Word Location)

inductive Ordering where
  | relaxed | acquire | release | acqRel
  deriving DecidableEq

inductive Atom where
  | cas (location : Location) (expected replacement : Word) (success failure : Ordering)
  | fence (order : Ordering)
  | store (location : Location) (value : Word) (order : Ordering)
  deriving DecidableEq

end NeuralReversi.TranspositionTable.WriterMir

module
public import NeuralReversi.TranspositionTable.WriterMir.Syntax

@[expose] public section

namespace NeuralReversi.TranspositionTable.WriterMir

open SeqLock (Word Location)

structure Request where
  expected : Word
  player : Word
  opponent : Word
  entry : EntryData
  deriving DecidableEq

inductive SourceOp where
  | load (location : Location) (order : Ordering)
  | snapshot
  | fence (order : Ordering)
  | spin
  | claim (request : Request)
  | exact (player opponent : Word) (entry : EntryData)
  deriving DecidableEq

structure SourceStep where
  ops : List SourceOp
  retry : Bool
  deriving DecidableEq

end NeuralReversi.TranspositionTable.WriterMir

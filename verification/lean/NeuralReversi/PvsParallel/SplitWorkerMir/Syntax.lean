import NeuralReversi.PvsParallel.ConcurrentReplay

namespace NeuralReversi.PvsParallel.SplitWorkerMir

abbrev Word := BitVec 32

structure Input where
  index : Nat
  depth : Nat
  pv : Bool
  root : Bool
  cut : Bool
  beta : Word
  alpha : Word
  fresh : Word
  scout : Word
  wide : Word

structure Query where
  index : Nat
  depth : Nat
  pv : Bool
  alpha : Word
  beta : Word
  cut : Bool
  result : Word
  deriving DecidableEq

inductive Step where
  | load (again : Bool)
  | returned (call : Query)
  | lock
  | poll
  | unlock
  deriving DecidableEq

structure Searched where
  steps : List Step
  searchedAlpha : Word
  chosen : Word

structure Locked where
  stopped : Bool
  alpha : Word
  best : Word

inductive Exit where
  | stopped | again | cutoff
  deriving DecidableEq

structure Commit where
  exit : Exit
  rootRow : Option (Word × Bool)
  best : Option Word
  moveSet : Bool
  alpha : Option Word
  cut : Bool
  deriving DecidableEq

end NeuralReversi.PvsParallel.SplitWorkerMir

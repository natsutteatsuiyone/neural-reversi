module
public import NeuralReversi.Threading.AtomicWord

@[expose] public section

namespace NeuralReversi.PvsParallel.WorkerFrameMir

inductive Kind
  | work | helper | helperDrop | idle | mainLoop | split | abandon | detach
  | mainEntry | workerEntry | book | bookJob | finalize | other
  deriving DecidableEq

structure Frame where
  kind : Kind
  entered : Threading.AtomicWord.Event
  deriving DecidableEq

end NeuralReversi.PvsParallel.WorkerFrameMir

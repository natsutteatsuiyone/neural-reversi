module

@[expose] public section

namespace NeuralReversi.PvsParallel.SplitActivationMir

inductive Op
  | slotLock | slotUnlock | threadLock | threadUnlock | clear | add | accept
  | initialCandidates | assign | wake | take | sleep | detach | close | remove
  deriving DecidableEq

end NeuralReversi.PvsParallel.SplitActivationMir

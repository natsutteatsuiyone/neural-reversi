import NeuralReversi.PvsParallel.SplitActivationMir.Syntax

namespace NeuralReversi.PvsParallel.SplitActivationMir

def initialOps : List Op := [.slotLock, .clear, .add, .accept, .assign, .initialCandidates, .slotUnlock]
def enrollOps : List Op := [.add, .assign, .wake]
def releaseOps : List Op := [.slotLock, .threadLock, .detach, .close, .remove, .threadUnlock]
def detachOps : List Op := [.take, .sleep]

def gate (late available idle : Bool) : Bool :=
  if late then available && idle else idle

def entersWorker (searching hasHelperJob : Bool) : Bool := searching && !hasHelperJob

end NeuralReversi.PvsParallel.SplitActivationMir

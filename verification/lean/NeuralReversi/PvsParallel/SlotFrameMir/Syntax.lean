module
public import NeuralReversi.PvsParallel.WorkerFrameMir.Syntax
public import NeuralReversi.Threading.SplitCompletion

@[expose] public section

namespace NeuralReversi.PvsParallel.SlotFrameMir

abbrev Worker := Threading.SplitCompletion.Worker
abbrev Slot := Worker × Nat

end NeuralReversi.PvsParallel.SlotFrameMir

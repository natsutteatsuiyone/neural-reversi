module

@[expose] public section

namespace NeuralReversi.PvsParallel.SlotFrameMir

def dispatchWork (searching job : Bool) := searching && !job
def completedHelperBusy := false
def postHelperJob := true

end NeuralReversi.PvsParallel.SlotFrameMir

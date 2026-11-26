import NeuralReversi.PvsParallel.SlotFrameMir.Live

namespace NeuralReversi.PvsParallel.SlotFrameMir

def dispatchWork (searching job : Bool) := searching && !job
def completedHelperBusy := false
def postHelperJob := true

end NeuralReversi.PvsParallel.SlotFrameMir

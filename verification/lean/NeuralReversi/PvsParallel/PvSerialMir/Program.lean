module
public import NeuralReversi.PvsParallel.RootHeaderMir
public import NeuralReversi.PvsParallel.RootUpdateMir

@[expose] public section

namespace NeuralReversi.PvsParallel.PvSerialMir

def minimumClaimDepth : Nat := 19

abbrev action := headerAction
abbrev decision := RootUpdateMir.decision
def writesPv (d : RootUpdateMir.Changes) : Bool := d.takesMove

end NeuralReversi.PvsParallel.PvSerialMir

module
public import NeuralReversi.PvsParallel.SlotFrameMir.Syntax

@[expose] public section

namespace NeuralReversi.PvsParallel.SlotFrameMir

def slotsPerThread : Nat := 8
def ownedSlot (u : Worker) (depth : Nat) : Slot := (u,depth)
def savedParent (active : Option Slot) := active
def installed (slot : Slot) := some slot
def restored (taskParent : Option Slot) := taskParent
def workerReset (captured _taken : Option Slot) := captured
def unwindReset (_captured taken : Option Slot) := taken
def unwindAborts (who actualOwner : Worker) : Bool := decide (actualOwner ≠ who)

end NeuralReversi.PvsParallel.SlotFrameMir

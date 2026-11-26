import NeuralReversi.PvsParallel.NativeStopReturns.Syntax

namespace NeuralReversi.PvsParallel.NativeStopReturns

def boundaryPcs : List Nat := [26, 28, 442, 443, 444, 445]
def deepPollPcs : List Nat := [54, 107]
def movePollPcs : List Nat := [0]
def boundaryPollPc : Nat := 71

def nodesBeforePc : Nat := 20
def nodesAfterPc : Nat := 68

def passSite : Nat := 2

def routes : List Route := [
  ⟨.cached, 251, [(248, [.poll]), (249, [.taken]), (250, [.zero]), (434, [.releaseClaim]), (435, []), (436, []), (437, []), (439, []), (441, []), (444, []), (445, [.ret])]⟩,
  ⟨.postJoin, 308, [(305, [.poll]), (306, [.taken]), (307, [.zero]), (433, []), (436, []), (437, []), (439, []), (441, []), (444, []), (445, [.ret])]⟩,
  ⟨.cached, 379, [(376, [.poll]), (377, [.taken]), (378, [.zero, .releaseClaim]), (432, []), (433, []), (436, []), (437, []), (439, []), (441, []), (444, []), (445, [.ret])]⟩,
  ⟨.passChild, 0, [(59, [.child]), (60, [.negate]), (61, [.undoPass]), (62, [.forward]), (440, []), (441, []), (444, []), (445, [.ret])]⟩,
  ⟨.terminal, 0, [(63, [.solve]), (64, []), (440, []), (441, []), (444, []), (445, [.ret])]⟩]

end NeuralReversi.PvsParallel.NativeStopReturns

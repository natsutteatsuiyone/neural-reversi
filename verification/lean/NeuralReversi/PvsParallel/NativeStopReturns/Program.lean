import NeuralReversi.PvsParallel.NativeStopReturns.Syntax

namespace NeuralReversi.PvsParallel.NativeStopReturns

def boundaryPcs : List Nat := [27, 29, 444, 445, 446, 447]
def deepPollPcs : List Nat := [46, 98]
def movePollPcs : List Nat := [0]
def boundaryPollPc : Nat := 67

def nodesBeforePc : Nat := 20
def nodesAfterPc : Nat := 64

def passSite : Nat := 2

def routes : List Route := [
  ⟨.cached, 253, [(250, [.poll]), (251, [.taken]), (252, [.zero]), (436, [.releaseClaim]), (437, []), (438, []), (439, []), (441, []), (443, []), (446, []), (447, [.ret])]⟩,
  ⟨.postJoin, 310, [(307, [.poll]), (308, [.taken]), (309, [.zero]), (435, []), (438, []), (439, []), (441, []), (443, []), (446, []), (447, [.ret])]⟩,
  ⟨.cached, 381, [(378, [.poll]), (379, [.taken]), (380, [.zero, .releaseClaim]), (434, []), (435, []), (438, []), (439, []), (441, []), (443, []), (446, []), (447, [.ret])]⟩,
  ⟨.passChild, 0, [(60, [.child]), (61, [.negate]), (62, [.undoPass]), (63, [.forward]), (442, []), (443, []), (446, []), (447, [.ret])]⟩,
  ⟨.terminal, 0, [(65, [.solve]), (66, []), (442, []), (443, []), (446, []), (447, [.ret])]⟩]

end NeuralReversi.PvsParallel.NativeStopReturns

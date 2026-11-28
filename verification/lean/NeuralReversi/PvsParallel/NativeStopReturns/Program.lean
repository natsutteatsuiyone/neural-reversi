module
public import NeuralReversi.PvsParallel.NativeStopReturns.Syntax

@[expose] public section

namespace NeuralReversi.PvsParallel.NativeStopReturns

def boundaryPcs : List Nat := [26, 28, 456, 457, 458, 459]
def deepPollPcs : List Nat := [54, 107]
def movePollPcs : List Nat := [0]
def boundaryPollPc : Nat := 71

def nodesBeforePc : Nat := 20
def nodesAfterPc : Nat := 68

def passSite : Nat := 2

def routes : List Route := [
  ⟨.cached, 264, [(261, [.poll]), (262, [.taken]), (263, [.zero]), (447, [.releaseClaim]), (448, []), (449, []), (450, []), (453, []), (455, []), (458, []), (459, [.ret])]⟩,
  ⟨.postJoin, 321, [(318, [.poll]), (319, [.taken]), (320, [.zero]), (446, []), (449, []), (450, []), (453, []), (455, []), (458, []), (459, [.ret])]⟩,
  ⟨.cached, 392, [(389, [.poll]), (390, [.taken]), (391, [.zero, .releaseClaim]), (445, []), (446, []), (449, []), (450, []), (453, []), (455, []), (458, []), (459, [.ret])]⟩,
  ⟨.passChild, 0, [(59, [.child]), (60, [.negate]), (61, [.undoPass]), (62, [.forward]), (454, []), (455, []), (458, []), (459, [.ret])]⟩,
  ⟨.terminal, 0, [(63, [.solve]), (64, []), (454, []), (455, []), (458, []), (459, [.ret])]⟩]

end NeuralReversi.PvsParallel.NativeStopReturns

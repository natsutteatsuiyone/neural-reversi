import NeuralReversi.Stability.EdgeCapture
namespace NeuralReversi.Stability.Edge

def MoveProgress (p o : Byte) (x : Fin 8) : Prop :=
  p &&& o = 0 → emptyAt p o x = true →
  let a := applyEdgeMove x p o
  a.1 &&& a.2 = 0 ∧ (a.1 ||| a.2).cpop.toNat = (p ||| o).cpop.toNat + 1

def MoveCorrect (p o : Byte) (x : Fin 8) : Prop :=
  p &&& o = 0 → emptyAt p o x = true →
  let a := applyEdgeMove x p o
  a.1 &&& a.2 = 0 ∧ (a.1 ||| a.2).cpop.toNat = (p ||| o).cpop.toNat + 1 ∧
    a = moveSpec p o x

end NeuralReversi.Stability.Edge

module

@[expose] public section

namespace NeuralReversi.PvsParallel.RootUpdateMir

structure Changes where
  visible : Bool
  raisesAlpha : Bool
  raisesBest : Bool
  takesMove : Bool
  cuts : Bool

def changes (first higherAlpha higherBest belowBeta atMax : Bool) : Changes :=
  ⟨(if first then true else higherAlpha),
   (if higherAlpha then (if higherBest then belowBeta else false) else false),
   higherBest,
   (if higherAlpha then higherBest else false),
   (if higherAlpha then (if higherBest then (if belowBeta then atMax else true) else false) else false)⟩

end NeuralReversi.PvsParallel.RootUpdateMir

module
public import NeuralReversi.PvsParallel.SplitWorkerMir.FieldAccess

@[expose] public section

namespace NeuralReversi.PvsParallel.SplitWorkerMir.FieldStores

def accesses (i : Input) (s : Locked) (chosen : Word) :=
  let branch0 : List Access := if chosen.toInt < i.beta.toInt then [.readAlpha s.alpha, .readBest s.best, .writeBest chosen, .readAlpha s.alpha, .writeMove i.index, .writeAlpha chosen] else [.readAlpha s.alpha, .readBest s.best, .writeBest chosen, .readAlpha s.alpha, .writeMove i.index]
  let branch1 : List Access := if i.pv then branch0 else [.readAlpha s.alpha, .readBest s.best, .writeBest chosen, .readAlpha s.alpha, .writeMove i.index]
  let branch2 : List Access := if chosen.toInt > s.alpha.toInt then branch1 else [.readAlpha s.alpha, .readBest s.best, .writeBest chosen, .readAlpha s.alpha]
  let branch3 : List Access := if chosen.toInt > s.best.toInt then branch2 else [.readAlpha s.alpha, .readBest s.best]
  let branch4 : List Access := if chosen.toInt < i.beta.toInt then [.readBest s.best, .writeBest chosen, .readAlpha s.alpha, .writeMove i.index, .writeAlpha chosen] else [.readBest s.best, .writeBest chosen, .readAlpha s.alpha, .writeMove i.index]
  let branch5 : List Access := if i.pv then branch4 else [.readBest s.best, .writeBest chosen, .readAlpha s.alpha, .writeMove i.index]
  let branch6 : List Access := if chosen.toInt > s.alpha.toInt then branch5 else [.readBest s.best, .writeBest chosen, .readAlpha s.alpha]
  let branch7 : List Access := if chosen.toInt > s.best.toInt then branch6 else [.readBest s.best]
  let branch8 : List Access := if i.root then branch3 else branch7
  let branch9 : List Access := if s.stopped then [] else branch8
  branch9

end NeuralReversi.PvsParallel.SplitWorkerMir.FieldStores

module
public import NeuralReversi.PvsParallel.SplitWorkerMir.Syntax

@[expose] public section

namespace NeuralReversi.PvsParallel.SplitWorkerMir

def maximum : Word := 16384

def emptyExit : Threading.SplitReturn.Departure := .exhausted

def lockedSteps : List Step := [.poll, .unlock]

def search (i : Input) : Option Searched :=
  let branch16 : Option Searched := if i.depth = 0 then none else some ⟨[.load false, .returned ⟨i.index, (i.depth - 1), false, (-(i.alpha + 1)), (-i.alpha), (!i.cut), i.scout⟩, .load true, .returned ⟨i.index, (i.depth - 1), true, (-i.beta), (-i.fresh), false, i.wide⟩, .lock], i.fresh, (-i.wide)⟩
  let branch17 : Option Searched := if (-i.scout).toInt > i.alpha.toInt then branch16 else some ⟨[.load false, .returned ⟨i.index, (i.depth - 1), false, (-(i.alpha + 1)), (-i.alpha), (!i.cut), i.scout⟩, .lock], i.alpha, (-i.scout)⟩
  let branch18 : Option Searched := if i.pv then branch17 else some ⟨[.load false, .returned ⟨i.index, (i.depth - 1), false, (-(i.alpha + 1)), (-i.alpha), (!i.cut), i.scout⟩, .lock], i.alpha, (-i.scout)⟩
  let branch19 : Option Searched := if i.depth = 0 then none else branch18
  let branch20 : Option Searched := if 1 < i.index + 1 then branch19 else none
  let branch21 : Option Searched := if i.pv then branch20 else branch19
  branch21

def commit (i : Input) (s : Locked) (chosen : Word) : Commit :=
  let branch0 : Commit := if chosen.toInt ≥ maximum.toInt then ⟨.cutoff, (some (chosen, decide (chosen.toInt > s.alpha.toInt))), (some chosen), true, (some chosen), true⟩ else ⟨.again, (some (chosen, decide (chosen.toInt > s.alpha.toInt))), (some chosen), true, (some chosen), true⟩
  let branch1 : Commit := if chosen.toInt ≥ maximum.toInt then ⟨.cutoff, (some (chosen, decide (chosen.toInt > s.alpha.toInt))), (some chosen), true, (some chosen), false⟩ else ⟨.again, (some (chosen, decide (chosen.toInt > s.alpha.toInt))), (some chosen), true, (some chosen), false⟩
  let branch2 : Commit := if chosen.toInt ≥ maximum.toInt then branch0 else branch1
  let branch3 : Commit := if chosen.toInt < i.beta.toInt then branch2 else ⟨.cutoff, (some (chosen, decide (chosen.toInt > s.alpha.toInt))), (some chosen), true, none, true⟩
  let branch4 : Commit := if i.pv then branch3 else ⟨.cutoff, (some (chosen, decide (chosen.toInt > s.alpha.toInt))), (some chosen), true, none, true⟩
  let branch5 : Commit := if chosen.toInt > s.alpha.toInt then branch4 else ⟨.again, (some (chosen, decide (chosen.toInt > s.alpha.toInt))), (some chosen), false, none, false⟩
  let branch6 : Commit := if chosen.toInt > s.best.toInt then branch5 else ⟨.again, (some (chosen, decide (chosen.toInt > s.alpha.toInt))), none, false, none, false⟩
  let branch7 : Commit := if chosen.toInt ≥ maximum.toInt then ⟨.cutoff, none, (some chosen), true, (some chosen), true⟩ else ⟨.again, none, (some chosen), true, (some chosen), true⟩
  let branch8 : Commit := if chosen.toInt ≥ maximum.toInt then ⟨.cutoff, none, (some chosen), true, (some chosen), false⟩ else ⟨.again, none, (some chosen), true, (some chosen), false⟩
  let branch9 : Commit := if chosen.toInt ≥ maximum.toInt then branch7 else branch8
  let branch10 : Commit := if chosen.toInt < i.beta.toInt then branch9 else ⟨.cutoff, none, (some chosen), true, none, true⟩
  let branch11 : Commit := if i.pv then branch10 else ⟨.cutoff, none, (some chosen), true, none, true⟩
  let branch12 : Commit := if chosen.toInt > s.alpha.toInt then branch11 else ⟨.again, none, (some chosen), false, none, false⟩
  let branch13 : Commit := if chosen.toInt > s.best.toInt then branch12 else ⟨.again, none, none, false, none, false⟩
  let branch14 : Commit := if i.root then branch6 else branch13
  let branch15 : Commit := if s.stopped then ⟨.stopped, none, none, false, none, false⟩ else branch14
  branch15

end NeuralReversi.PvsParallel.SplitWorkerMir

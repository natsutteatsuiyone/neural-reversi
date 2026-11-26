import NeuralReversi.PvsParallel.GeneralOwnerMir.Calls

namespace NeuralReversi.PvsParallel.GeneralOwnerMir.Regression

private def busy : Schedule Nat := ⟨fun _ _ => false,fun _ _ => true⟩
private def idleHelper : Schedule Nat := ⟨fun _ _ => true,fun _ _ => false⟩
private def noSplit : SplitReturn Nat := fun _ _ _ => none

/-- After two rotations fresh_end is behind idx while the deferred moves return. -/
private def deferred (pre : Int) :=
  Calls.body busy (fun i _ => Int.ofNat (i+3)) 20 true [7,2,3,4] 2 4 (some 7) 0 1 pre noSplit

#guard (deferred 4).calls == [(7,4),(4,5),(2,6),(3,7)]
#guard (deferred 4).value == some (-4)
#guard (deferred (-1)).calls == [(7,-1)]
#guard (deferred (-1)).value == some 1

private def beforeFirst :=
  Calls.body idleHelper (fun _ _ => 0) 20 false [2,4] 4 2 none 0 1 0 fun _ _ r => do
    let .handoff s := r.outcome | none
    if r.remaining.idx == 0 && r.remaining.pending == [2,4] && s.best == -32767 then some 5 else none

#guard beforeFirst.calls.isEmpty
#guard beforeFirst.value == some 5

end NeuralReversi.PvsParallel.GeneralOwnerMir.Regression

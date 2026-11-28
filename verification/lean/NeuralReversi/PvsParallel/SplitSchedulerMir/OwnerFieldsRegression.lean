module
public import NeuralReversi.PvsParallel.SplitSchedulerMir.OwnerFields
meta import NeuralReversi.PvsParallel.SplitSchedulerMir.OwnerFields

@[expose] public section

namespace NeuralReversi.PvsParallel.SplitSchedulerMir.OwnerFieldsRegression

open Threading.AtomicWord (Write)
open OwnerFields

private def owner : Clock := ⟨2, fun step => 4 + 2 * step, by omega⟩
private def store (score : BitVec 32) : Write := ⟨⟨1, 5⟩, score.zeroExtend 64, false, false⟩

/- Two successive initializations of slot 0 in the twenty-empty native solve.
   The first scored -6144 and the next one -2048. Both retained the absent move. -/
private def first : Initialization := ⟨0, ⟨6, by decide⟩, -32767, 255, -32767, [store (-6144)], [], []⟩
private def second : Initialization := ⟨0, ⟨6, by decide⟩, -6144, 255, -6144, [store (-2048)], [], []⟩
private def writes (field : Field) := laterWrites owner 0 field 10

#guard ((writes .score [first, second]).map fun w => (w.value.setWidth 32).toInt) == [-32767, -6144, -6144, -2048]
#guard ((writes .move [first, second]).map fun w => (w.event.step, w.value.toNat)) == [(30, 255), (42, 255)]

/- A different slot consumes owner instructions. A prefix ending just after the
   score seed exposes that store, while the move field still has no later write. -/
private def partialInit : Initialization := {second with count := ⟨3, by decide⟩, scoreStores := []}
#guard ((writes .score [{first with index := 1}, partialInit]).map (·.event.step)) == [40]
#guard (writes .move [{first with index := 1}, partialInit]).isEmpty
#guard (writes .score [{second with count := ⟨2, by decide⟩}]).isEmpty

/- Alpha is the fifth initializer instruction. A partial reuse may seed the
   score and move while leaving the alpha word from the previous use intact. -/
#guard (writes .alpha [{second with count := ⟨4, by decide⟩}]).isEmpty
#guard ((writes .alpha [{first with index := 1}, {second with count := ⟨5, by decide⟩}]).map (·.event.step)) == [44]

end NeuralReversi.PvsParallel.SplitSchedulerMir.OwnerFieldsRegression

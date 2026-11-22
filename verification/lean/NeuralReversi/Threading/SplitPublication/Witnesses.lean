import NeuralReversi.Threading.SplitPublication

namespace NeuralReversi.Threading.SplitPublication.Witnesses

open AtomicWord (Event Write Read ReadWindow)

private def searching : List Write :=
  [⟨⟨0, 0⟩, 1, true, false⟩, ⟨⟨1, 1⟩, 0, true, false⟩,
   ⟨⟨0, 2⟩, 1, true, false⟩, ⟨⟨1, 4⟩, 0, true, false⟩,
   ⟨⟨0, 5⟩, 1, true, false⟩]

private def idle : Fin searching.length :=
  ⟨1, by decide⟩

private def tryRead (source : Fin searching.length) : Read searching :=
  ⟨⟨1, 3⟩, source, true⟩

private def old : Fin searching.length := ⟨0, by decide⟩
private def current : Fin searching.length := ⟨2, by decide⟩
private def future : Fin searching.length := ⟨4, by decide⟩

def WakeWitnessContract : Prop :=
  (searching.get old).value = 1 ∧
  (searching.get current).value = 1 ∧
  (searching.get future).value = 1 ∧
  ∀ before : Event → Event → Prop,
    ¬ ReadWindow (tryRead old) before idle current.val ∧
    ¬ ReadWindow (tryRead future) before idle current.val

theorem wakeWitnessContract_proven : WakeWitnessContract := by
  refine ⟨rfl, rfl, rfl, fun before => ⟨?_, ?_⟩⟩
  · intro seen
    have lower := seen.bounds.1
    change 1 ≤ 0 at lower
    omega
  · intro seen
    have upper := seen.bounds.2
    change 4 ≤ 2 at upper
    omega

end NeuralReversi.Threading.SplitPublication.Witnesses

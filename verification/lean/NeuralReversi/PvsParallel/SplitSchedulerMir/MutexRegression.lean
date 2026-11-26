import NeuralReversi.PvsParallel.SplitSchedulerMir.Mutex

namespace NeuralReversi.PvsParallel.SplitSchedulerMir.MutexRegression

open Threading.AtomicWord (Event)
open Threading.SpinLock (Operation)

/- A four-thread exact nine-empty split. There were unreported acquisitions of
   the same mutex between ticket reports. Thread 2 wrote alpha at stamp 237. -/
private def ops : List Operation := [
  .acquire ⟨5,185⟩, .release ⟨5,189⟩, .acquire ⟨5,193⟩, .release ⟨5,196⟩,
  .acquire ⟨4,199⟩, .release ⟨4,203⟩, .acquire ⟨4,207⟩, .release ⟨4,210⟩,
  .acquire ⟨6,216⟩, .release ⟨6,220⟩, .acquire ⟨6,224⟩, .release ⟨6,227⟩,
  .acquire ⟨2,230⟩, .release ⟨2,238⟩]
private def rows : List (Event × Event) := [
  (⟨5,185⟩,⟨5,189⟩),(⟨4,199⟩,⟨4,203⟩),(⟨6,216⟩,⟨6,220⟩),(⟨2,230⟩,⟨2,238⟩)]

#guard Mutex.covers ops rows
#guard !(Mutex.covers ops rows.reverse)
#guard !(Mutex.covers ops [(⟨5,185⟩,⟨5,196⟩)])
#guard !(Mutex.covers ops [(⟨5,185⟩,⟨4,203⟩)])
#guard Mutex.covers (ops.take 13) (rows.take 3)
#guard !(Mutex.covers (ops.take 13) rows)
#guard (Mutex.guards [.acquire ⟨1,4⟩,.release ⟨1,3⟩]).isNone
#guard (Mutex.guards [.acquire ⟨1,4⟩,.release ⟨2,5⟩]).isNone
#guard (Mutex.guards [.acquire ⟨1,4⟩,.acquire ⟨2,5⟩]).isNone

end NeuralReversi.PvsParallel.SplitSchedulerMir.MutexRegression

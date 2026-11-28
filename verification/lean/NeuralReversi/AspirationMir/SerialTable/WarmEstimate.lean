module
public import NeuralReversi.AspirationMir.SerialTable.InitialEstimate
public import NeuralReversi.PvsParallel.SerialTable.ExactScores

@[expose] public section

namespace NeuralReversi.AspirationMir.SerialTable.InitialEstimate

open PvsParallel
open PvsParallel.SerialTable (Table ExactScores)
open PvsMoveLoop (scoreMax)
open PvsBoardTree (Position)
open TranspositionTable

/-- Either slot may hold a weak exact-bound word or an exact search result.
    The other bounds take the evaluator path. -/
theorem warm_bounded (t : Table l) (mc fc) (raw : Position → Int) (b : Position)
    (support : t.image.Supported t.history) (seeds : ExactScores t.history) :
    let value := estimate t.image mc fc raw b
    (-scoreMax ≤ value.toInt ∧ value.toInt ≤ scoreMax) := by
  have represent (n : Int) (h : -scoreMax ≤ n ∧ n ≤ scoreMax) :
      -scoreMax ≤ (BitVec.ofInt 32 n).toInt ∧ (BitVec.ofInt 32 n).toInt ≤ scoreMax := by
    rw [BitVec.toInt_ofInt_eq_self (by decide : 0 < 32) (by unfold scoreMax at h; omega)
      (by unfold scoreMax at h; omega)]
    exact h
  have onMiss := represent _ (fallback_bounded mc fc raw b)
  dsimp only [estimate]
  cases hit : t.image.probe b with
  | miss => exact onMiss
  | hit slot e =>
    obtain ⟨tried,reached,read,occupied⟩ := ProbeRetry.contract_proven.2 _ _ _ _ hit
    have member := (t.image.probe_reads support b).published.matches slot e tried reached read occupied
    by_cases bound : e.bound = Packing.boundByte .exact
    · simpa [Option.filter,bound] using represent _ (seeds b e member bound)
    · simpa [Option.filter,bound] using onMiss

end NeuralReversi.AspirationMir.SerialTable.InitialEstimate

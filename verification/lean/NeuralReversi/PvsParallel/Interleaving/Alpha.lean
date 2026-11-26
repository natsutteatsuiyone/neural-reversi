import NeuralReversi.Threading.SplitWindow.Stores

namespace NeuralReversi.PvsParallel.Interleaving

open Aspiration (Window)
open Threading
open SplitWindow (State scan)

/-- At either unlocked alpha load, the observed word comes from some prefix of the
    already committed reports. The scout and its re-search can see different prefixes. -/
def alphaAt (pv : Bool) (beta : Int) (entry : State) (scores : List Int) (count : Nat) : Int :=
  (scan pv beta entry (scores.take count)).alpha

theorem alphaAt_valid (pv : Bool) (original beta : Int) (entry : State) (scores : List Int)
    (count : Nat) (w : Window entry.alpha beta) (cap : entry.alpha ≤ max original entry.best) :
    Window (alphaAt pv beta entry scores count) beta ∧
    alphaAt pv beta entry scores count ≤ (scan pv beta entry scores).alpha ∧
    (pv = false → alphaAt pv beta entry scores count = entry.alpha) := by
  let initial := scan pv beta entry (scores.take count)
  obtain ⟨_,valid,capped,fixed⟩ := SplitWindow.stateContract_proven pv original beta entry (scores.take count) w cap
  have later := (SplitWindow.stateContract_proven pv original beta initial (scores.drop count) valid capped).1
  have same : scan pv beta initial (scores.drop count) = scan pv beta entry scores := by
    simp only [initial,scan,← List.foldl_append,List.take_append_drop]
  exact ⟨valid,same ▸ later,fixed⟩

/-- The Rust relaxed load may read an older value. Its coherence interval and the
    stores actually issued in that slot incarnation supply the schedule's prefix. -/
theorem alphaAt_of_load {history : CutoffVisibility.History} {writes : List AtomicWord.Write}
    {load : AtomicWord.Read writes} {first last : Fin writes.length}
    (read : SplitWindow.AlphaRead history load first last)
    (stores : SplitWindow.StoreWindow writes first last pv beta entry scores)
    (w : Window entry.alpha beta) (cap : entry.alpha ≤ max original entry.best) :
    ∃ count, count ≤ scores.length ∧
      AtomicWord.signed32At writes load.source = alphaAt pv beta entry scores count := by
  obtain ⟨lo,hi⟩ := read.window.bounds
  obtain ⟨early,tail,rfl,word⟩ := SplitWindow.storeContract_proven _ _ _ pv original beta entry scores stores w cap _ lo hi
  exact ⟨early.length,by simp,by simpa [alphaAt] using word⟩

end NeuralReversi.PvsParallel.Interleaving

import NeuralReversi.PvsParallel.NativeFrames.CacheCalls.Shallow

namespace NeuralReversi.PvsParallel.NativeFrames.CacheCalls.Deep

open EndgameSearch (Result SafeThreshold FlipConfig)
open EndgameSearch.CacheHistory (Pool)
open EndgameSearch.Cancellation (Outcome)
open EndgameSearch.CacheOwner (Edge transfers)
open PvsBoardTree (Position)
open Bitboard.Moves (X86Config)
open Flip.Dispatch (TargetConfig)

private def finish (site : Nat) (edge : Edge.mk .deep site .deepStore .deepField ∈ transfers)
    (b : Position) (a value : Int) (safe : SafeThreshold a) (s : EndgameSearch.Cancellation.State) :
    Program .deep s.history (EndgameSearch.DeepRecursionMir.finish b a value s).state.history := by
  unfold EndgameSearch.DeepRecursionMir.finish
  dsimp only
  split
  · exact .done
  · exact .access site (.store .deep b.val a value safe) .deepField edge .done

private def ordered (sub : M → EndgameSearch.Cancellation.State → Outcome) (issued : ∀ m s, Program .deepChild s.history (sub m s).state.history)
    (beta : Int) (ms : List M) (best : Int) (s : EndgameSearch.Cancellation.State) :
    Program .deep s.history (EndgameSearch.DeepRecursionMir.ordered sub beta ms best s).2.history := by
  cases ms with
  | nil => exact .done
  | cons m rest =>
    refine .call 7 .deepChild .forward (by decide) rfl (issued m s) ?_
    simp only [EndgameSearch.DeepRecursionMir.ordered]
    split
    · exact .done
    · exact ordered sub issued beta rest _ (sub m s).state
termination_by ms.length

def moves (b : Position) (a : Int) (safe : SafeThreshold a) (s : EndgameSearch.Cancellation.State) (first : M) (rest ranked : List M)
    (sub : M → EndgameSearch.Cancellation.State → Outcome) (issued : ∀ m s, Program .deepChild s.history (sub m s).state.history) :
    Program .deep s.history (EndgameSearch.DeepRecursionMir.moves b a s first rest ranked sub).state.history := by
  unfold EndgameSearch.DeepRecursionMir.moves
  split
  · exact .call 3 .deepChild .forward (by decide) rfl (issued first s)
      (finish 4 (by decide) b a _ safe (sub first s).state)
  · exact (ordered sub issued (a+1) ranked (-127) s).append (finish 8 (by decide) b a _ safe _)

theorem switched (safe : SafeThreshold a) : SafeThreshold (-(a+1)) := by
  obtain ⟨lo,hi⟩ := safe
  exact ⟨by omega,by omega⟩

/-- Poll before removing the square. The shallow child receives the shallow
    field of this pair, while a deeper child receives the pair itself. -/
def child (p : Pool) (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : FlipConfig)
    (next : Position) (sq : Bitboard.Square) (a : Int) (safe : SafeThreshold a) (s : EndgameSearch.Cancellation.State)
    (rec : Int → EndgameSearch.Cancellation.State → Outcome) (issued : ∀ s, Program .deep s.history (rec (-(a+1)) s).state.history) :
    Program .deepChild s.history (EndgameSearch.DeepRecursionMir.executeChild sq (a+1) s
      (EndgameSearch.ShallowMir.run target bmi2 mc fc p.address next) rec).state.history := by
  unfold EndgameSearch.DeepRecursionMir.executeChild
  dsimp only
  split
  · exact .done
  · split
    · exact .call 0 .shallow .shallowField (by decide) rfl
        (Shallow.program p target bmi2 mc fc next (s.empties.remove sq) _ (switched safe) s.history) .done
    · let placed := {s with polls := s.polls.after, empties := s.empties.remove sq}
      exact .call 1 .deep .forward (by decide) rfl (issued placed) .done

end NeuralReversi.PvsParallel.NativeFrames.CacheCalls.Deep

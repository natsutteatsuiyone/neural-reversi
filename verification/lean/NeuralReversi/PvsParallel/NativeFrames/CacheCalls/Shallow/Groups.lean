module
public import NeuralReversi.PvsParallel.NativeFrames.CacheCalls.Program

@[expose] public section

namespace NeuralReversi.PvsParallel.NativeFrames.CacheCalls.Shallow

open EndgameSearch (Result SafeThreshold)
open EndgameSearch.ShallowMir (Stage)
open EndgameSearch.CacheOwner (Edge transfers)
open EndgameSearch.CacheHistory (Pool)
open EndgameTreeCache (Store)
open PvsBoardTree (Position)

def stageHistory : Stage → List Store
  | .cutoff r => r.history
  | .next _ h => h

/-- One native shallow_search_moves activation. The child returns before the
    comparison, and only the cut branch stores at the forwarded parent slot. -/
def group (b : Position) (a : Int) (safe : SafeThreshold a) (sub : M → List Store → Result)
    (child : ∀ m h, Program .shallowChild h (sub m h).history) (ms : List M) (best : Int) (h : List Store) :
    Program .shallowMoves h (stageHistory (EndgameSearch.ShallowMir.stage b (a+1) sub ms best h)) := by
  cases ms with
  | nil => exact .done
  | cons m rest =>
    unfold EndgameSearch.ShallowMir.stage
    refine .call 0 .shallowChild .forward (by decide) rfl (child m h) ?_
    dsimp only
    split
    · let raw := (sub m h).score
      exact .access 1 (.store .shallow b.val (a+1-1) raw (by simpa using safe)) .forward (by simp only [Access.node]; decide) .done
    · exact group b a safe sub child rest (max best (sub m h).score) (sub m h).history
termination_by ms.length

theorem groupEdge (index : Nat) (small : index < 5) :
    Edge.mk .shallow (5+index) .shallowMoves .forward ∈ transfers :=
  (show ∀ k : Fin 5, Edge.mk .shallow (5+k.val) .shallowMoves .forward ∈ transfers by decide) ⟨index,small⟩

/-- Each group, even an empty group, has its own MIR call site. A cutoff skips
    later groups and leaves the store in shallow_search_moves. -/
def groups (b : Position) (a : Int) (safe : SafeThreshold a) (sub : M → List Store → Result)
    (child : ∀ m h, Program .shallowChild h (sub m h).history) (ms : List (List M)) (index : Nat)
    (fits : index + ms.length ≤ 5) (best : Int) (h : List Store) :
    Program .shallow h (EndgameSearch.ShallowMir.groups b a sub ms best h).history := by
  cases ms with
  | nil => exact .access 10 (.store .shallow b.val a best safe) .forward (by simp only [Access.node]; decide) .done
  | cons first later =>
    have size : index + (later.length+1) ≤ 5 := fits
    let block := group b a safe sub child first best h
    refine .call (5+index) .shallowMoves .forward (groupEdge _ (by omega)) rfl block ?_
    unfold EndgameSearch.ShallowMir.groups
    cases stage : EndgameSearch.ShallowMir.stage b (a+1) sub first best h with
    | cutoff r => exact .done
    | next improved past => exact groups b a safe sub child later (index+1) (by omega) improved past
termination_by ms.length

end NeuralReversi.PvsParallel.NativeFrames.CacheCalls.Shallow

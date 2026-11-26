import NeuralReversi.PvsParallel.NativeFrames.CacheCalls.Deep.Moves

namespace NeuralReversi.PvsParallel.NativeFrames.CacheCalls.Deep

open EndgameSearch (SafeThreshold FlipConfig)
open EndgameSearch.CacheHistory (Pool Input Arm Local)
open EndgameTreeCache (Store entryFor)
open PvsBoardTree (Position legalSquares pass play)
open Bitboard.Moves (X86Config)
open Flip.Dispatch (TargetConfig)

private def afterEarly (h last : List Store) : Option Int → List Store
  | some _ => h | none => last

/-- The generated-order loop probes each child, including a hit that is too
    large for the enhanced cutoff. It stops at the first usable upper bound. -/
private def enhanced (p : Pool) (a : Int) (safe : SafeThreshold a) (h last : List Store)
    (children : List Board.Board) (onMiss : Unit → Program .deep h last) :
    Program .deep h (afterEarly h last (EndgameCache.enhancedCutoff (entryFor p.address h .deep) a children)) := by
  cases children with
  | nil => exact onMiss ()
  | cons b rest =>
    refine .access 5 (.slot .deep (p.key b)) .deepField (by simp only [Access.node]; decide)
      (.access 6 (.probe .deep b (-(a+1)) (switched safe)) .deepField (by simp only [Access.node]; decide) ?_)
    simp only [EndgameCache.enhancedCutoff,List.findSome?_cons]
    cases hit : EndgameCache.probe (entryFor p.address h .deep b) b (-(a+1)) with
    | none => exact enhanced p a safe h last rest onMiss
    | some score =>
      by_cases cut : score ≤ -(a+1)
      · simp [cut,afterEarly]; exact .done
      · simpa [cut,EndgameCache.enhancedCutoff,Access.after] using enhanced p a safe h last rest onMiss
termination_by children.length

private def early (p : Pool) (fc : FlipConfig) (paired : Bool) (b : Position) (s : EndgameSearch.Cancellation.State)
    (a : Int) (safe : SafeThreshold a) (first : Bitboard.Square) (rest : List Bitboard.Square) (last : List Store)
    (onMiss : Unit → Program .deep s.history last) :
    Program .deep s.history (afterEarly s.history last (EndgameSearch.DeepRecursionMir.early fc paired b s p.address a first rest)) := by
  refine .access 1 (.slot .deep (p.key b.val)) .deepField (by simp only [Access.node]; decide)
    (.access 2 (.probe .deep b.val a safe) .deepField (by simp only [Access.node]; decide) ?_)
  unfold EndgameSearch.DeepRecursionMir.early
  split
  · exact .done
  · split
    · exact .done
    · dsimp only
      split
      · split
        · exact .done
        · exact onMiss ()
      · split
        · split
          · exact enhanced p a safe s.history last _ onMiss
          · exact onMiss ()
        · exact .done

/-- Emit the complete worker-local call sequence for null_window_search.
    The same poll state controls child admission and the parent's final store. -/
def program (p : Pool) (target : TargetConfig) (bmi2 paired : Bool) (mc : X86Config) (fc : FlipConfig)
    (b : Position) (a : Int) (safe : SafeThreshold a) (s : EndgameSearch.Cancellation.State) :
    Program .deep s.history (EndgameSearch.DeepRecursionMir.run target bmi2 paired mc fc p.address b a s).state.history := by
  rw [EndgameSearch.DeepRecursionMir.run.eq_def]
  split
  · split
    · exact .call 0 .deep .forward (by decide) rfl
        (program p target bmi2 paired mc fc (pass b) (-(a+1)) (switched safe) s) .done
    · exact .done
  · rename_i m ms movesNow
    let sub := fun (sq : PvsBoardTree.GeneratedSquare mc b.val) input =>
      let next := (play mc fc b sq).val
      EndgameSearch.DeepRecursionMir.executeChild sq.val (a+1) input
        (EndgameSearch.ShallowMir.run target bmi2 mc fc p.address next)
        (EndgameSearch.DeepRecursionMir.run target bmi2 paired mc fc p.address next)
    let following := EndgameSearch.DeepRecursionMir.moves b a s m ms (EndgameSearch.deepOrder mc fc b) sub
    let onMiss : Unit → Program .deep s.history following.state.history := fun _ =>
      moves b a safe s m ms _ sub fun sq input =>
        child p target bmi2 mc fc (play mc fc b sq).val sq.val a safe input _
          (fun nextState => program p target bmi2 paired mc fc (play mc fc b sq).val (-(a+1)) (switched safe) nextState)
    have block := early p fc paired b s a safe m.val (ms.map Subtype.val) following.state.history onMiss
    cases result : EndgameSearch.DeepRecursionMir.early fc paired b s p.address a m.val (ms.map Subtype.val) <;>
      simpa only [result,afterEarly] using block
termination_by Board.playRank mc b.val
decreasing_by
  · exact Board.legalCheckedStep_strict mc (.pass ⟨PvsBoardTree.stuck_of_nil (by assumption),by assumption⟩) b.property
  · exact (play mc fc b _).property

def boundaryProgram (p : Pool) (i : Input) (mc : X86Config) (fc : FlipConfig) (s : Local)
    (arm : i.arm = .deep poll) : Program .deep s.history (i.run mc fc p.address s.history).history := by
  simp only [Input.run,arm,Input.eval]
  have code := program p i.target i.bmi2 i.paired mc fc i.board _ (EndgameSearch.ShallowMir.scoreThreshold i.alpha)
    (DeepLeafMir.pollInput (i.state s.history) poll)
  cases poll <;> exact code

end NeuralReversi.PvsParallel.NativeFrames.CacheCalls.Deep

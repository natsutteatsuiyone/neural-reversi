import NeuralReversi.PvsParallel.NativeFrames.CacheCalls.Shallow.Groups

namespace NeuralReversi.PvsParallel.NativeFrames.CacheCalls.Shallow

open EndgameSearch (Result SafeThreshold FlipConfig)
open EndgameSearch.Cancellation (EmptyState)
open EndgameSearch.CacheHistory (Pool Input Good)
open EndgameTreeCache (Store entryFor)
open PvsBoardTree (Position GeneratedSquare legalSquares play pass)
open Bitboard.Moves (X86Config)
open Flip.Dispatch (TargetConfig)

private theorem switched (safe : SafeThreshold a) : SafeThreshold (-(a+1)) := by
  obtain ⟨lo,hi⟩ := safe
  exact ⟨by omega,by omega⟩

private def readNode (p : Pool) (b : Position) (a : Int) (safe : SafeThreshold a) (next : Program .shallow h last) :
    Program .shallow h last :=
  .access 1 (.slot .shallow (p.key b.val)) .forward (by simp only [Access.node]; decide)
    (.access 2 (.probe .shallow b.val a safe) .forward (by simp only [Access.node]; decide) next)

private def readChild (p : Pool) (b : Position) (a : Int) (safe : SafeThreshold a) (next : Program .shallowChild h last) :
    Program .shallowChild h last :=
  .access 0 (.slot .shallow (p.key b.val)) .forward (by simp only [Access.node]; decide)
    (.access 1 (.probe .shallow b.val a safe) .forward (by simp only [Access.node]; decide) next)

/-- At four empties the child can store only after the fixed solver returns.
    On a cache hit or stability cutoff it returns without that store. -/
def child (p : Pool) (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : FlipConfig)
    (b : Position) (empty : EmptyState) (m : GeneratedSquare mc b.val) (a : Int) (safe : SafeThreshold a)
    (h : List Store) (rec : EmptyState → Int → List Store → Result)
    (issued : Unit → Program .shallow h (rec (empty.remove m.val) (-(a+1)) h).history) :
    Program .shallowChild h (EndgameSearch.ShallowMir.executeChild target bmi2 mc fc p.address b empty m (a+1) h rec).history := by
  let n := (play mc fc b m).val
  let placed := empty.remove m.val
  let threshold := -(a+1)
  unfold EndgameSearch.ShallowMir.executeChild EndgameSearch.ShallowMir.child
  dsimp only
  split
  · refine readChild p n threshold (switched safe) ?_
    split
    · exact .done
    · split
      · exact .done
      · let value := EndgameSearch.ShallowMir.four target bmi2 n placed threshold
        exact .access 2 (.store .shallow n.val threshold value (switched safe)) .forward
          (by simp only [Access.node]; decide) .done
  · exact .call 3 .shallow .forward (by decide) rfl (issued ()) .done

private def legalNode (p : Pool) (b : Position) (a : Int) (safe : SafeThreshold a)
    (h : List Store) (first : M) (rest : List M) (stable : Option Int) (stages : List (List M))
    (five : stages.length = 5) (sub : M → List Store → Result)
    (issued : ∀ m h, Program .shallowChild h (sub m h).history) :
    let cached := EndgameCache.probe (entryFor p.address h .shallow b.val) b.val a
    Program .shallow h (EndgameSearch.ShallowMir.node b (first::rest) a h false 0 (fun _ => ⟨0,h⟩)
      cached stable stages sub).history := by
  refine readNode p b a safe ?_
  simp only [EndgameSearch.ShallowMir.node]
  split
  · exact .done
  · split
    · exact .done
    · split
      · exact .call 3 .shallowChild .forward (by decide) rfl (issued first h)
          (.access 4 (.store .shallow b.val a (sub first h).score safe) .forward
            (by simp only [Access.node]; decide) .done)
      · exact groups b a safe sub issued stages 0 (by omega) (-127) h

/-- Construct all cache calls issued by shallow_search, including forced pass,
    four-empty children, singly legal moves and all five native move groups. -/
def program (p : Pool) (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : FlipConfig)
    (b : Position) (empty : EmptyState) (a : Int) (safe : SafeThreshold a) (h : List Store) :
    Program .shallow h (EndgameSearch.ShallowMir.run target bmi2 mc fc p.address b empty a h).history := by
  rw [EndgameSearch.ShallowMir.run.eq_def]
  split
  · rename_i moves
    split
    · exact .call 0 .shallow .forward (by decide) rfl
        (program p target bmi2 mc fc (pass b) empty (-(a+1)) (switched safe) h) .done
    · exact .done
  · rename_i first rest moves
    apply legalNode p b a safe h first rest _ _ (by rfl)
    intro m past
    apply child p target bmi2 mc fc b empty m a safe past
    intro _
    exact program p target bmi2 mc fc (play mc fc b m).val (empty.remove m.val) (-(a+1)) (switched safe) past
termination_by Board.playRank mc b.val
decreasing_by
  · exact Board.legalCheckedStep_strict mc (.pass ⟨PvsBoardTree.stuck_of_nil (by assumption),by assumption⟩) b.property
  · exact (play mc fc b m).property

def boundaryProgram (p : Pool) (i : Input) (mc : X86Config) (fc : FlipConfig) (s : EndgameSearch.CacheHistory.Local)
    (arm : i.arm = .shallow) : Program .shallow s.history (i.run mc fc p.address s.history).history := by
  simp only [Input.run,arm,Input.eval]
  exact program p i.target i.bmi2 mc fc i.board i.empties _ (EndgameSearch.ShallowMir.scoreThreshold i.alpha) s.history

theorem audited (p : Pool) (i : Input) (arm : i.arm = .shallow) (g : Good mc fc p.width p.sizes p.key s) :
    let ops := ((boundaryProgram p i mc fc s arm).accesses.map Access.instruction)
    let actual := EndgameSearch.CacheHistory.Payloads.observeRun p s g.shape ops
    EndgameSearch.CacheHistory.Payloads.check p true s (i.run mc fc p.address s.history).history actual =
      some ⟨actual,i.after mc fc p.address s⟩ := by
  let code := boundaryProgram p i mc fc s arm
  have sound := (EndgameSearch.CacheHistory.after_call i g (EndgameSearch.CacheMir.budget_fits p.size p.width p.word)).1.valid
  have audit := code.audit sound p g.shape
  have same := (EndgameSearch.CacheHistory.Payloads.complete_state i g _ audit).1
  exact same ▸ audit

end NeuralReversi.PvsParallel.NativeFrames.CacheCalls.Shallow

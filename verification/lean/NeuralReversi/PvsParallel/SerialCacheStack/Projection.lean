import NeuralReversi.PvsParallel.SerialCacheStack.Program

namespace NeuralReversi.PvsParallel.SerialCacheStack

open SerialCalls (Request Program Executes Tree)
open SerialExecution (Frame Inputs body push Invariant)
open EndgameTreeCache (soundHistory)

/-- Replaying the exact operand load at each numbered activation preserves the
    whole nested stack execution, including the return word of every child. -/
private theorem change_source {c : SerialScout.Config} {inputs : Inputs mc fc}
    {before after : Nat → History}
    (ran : Executes (body c (source inputs before)) (push mc fc) i ctx p k v forest)
    (eq : ∀ j, i < j → j ≤ k → before j = after j) :
    Executes (body c (source inputs after)) (push mc fc) i ctx p k v forest := by
  induction ran with
  | pure => exact .pure
  | @call caller q child start mid word inside resume last value tail entered sub next down later =>
    have sm := sub.openings.2
    have ml := next.openings.2
    have own := body_congr (inputs := source inputs before) (other := source inputs after) (i := start+1)
      (ctx := child) c (by simp only [source,eq (start+1) (by omega) (by omega)])
    have childRun := down (fun j lo hi => eq j (by omega) (by omega))
    rw [own] at childRun
    exact .call entered childRun (later fun j lo hi => eq j (by omega) hi)

structure SourceFrame (c : SerialScout.Config) (inputs : Inputs mc fc) (i : Nat) (ctx : Frame)
    (past : History) (k : Nat) (value : Int) : Prop where
  execution : ∃ loads forest, (∀ j, soundHistory mc fc (loads j)) ∧ loads i = past ∧
    Executes (body c (source inputs loads)) (push mc fc) i ctx (body c (source inputs loads) i ctx) k value forest

structure SourceProgram (c : SerialScout.Config) (inputs : Inputs mc fc) (i : Nat) (ctx : Frame)
    (p : Program Request α) (k : Nat) (value : α) : Prop where
  execution : ∃ loads forest, (∀ j, soundHistory mc fc (loads j)) ∧ Program.Runs p (forest.map Tree.direct) value ∧
    (∀ r ∈ forest.map Tree.direct, ∃ child, push mc fc ctx r.1 = some child ∧ Invariant mc fc child r.2) ∧
    ∀ f : α → Int, Executes (body c (source inputs loads)) (push mc fc) i ctx (p.map f) k (f value) forest

theorem interpret_returns {inputs : Inputs mc fc} {c : SerialScout.Config} {ctx : Frame}
    {p : Program Request α} (live : Program.Ready allowed p) (descend : Child)
    (childRun : ∀ q, allowed q → ∀ i past, soundHistory mc fc past →
      ∃ child k v after, push mc fc ctx q = some child ∧ descend i child past = some (k,v,after) ∧
        soundHistory mc fc after ∧ past <+: after ∧ Invariant mc fc child v ∧ SourceFrame c inputs i child past k v)
    (i : Nat) (past : History) (ok : soundHistory mc fc past) :
    ∃ k v after, interpret (mc := mc) (fc := fc) descend i ctx past p = some (k,v,after) ∧
      soundHistory mc fc after ∧ past <+: after ∧ SourceProgram c inputs i ctx p k v := by
  induction live generalizing i past with
  | pure =>
    refine ⟨_,_,past,rfl,ok,List.prefix_refl _,⟨fun _ => past,[],fun _ => ok,.pure,by simp,fun _ => .pure⟩⟩
  | @call q resume allowed _ ih =>
    obtain ⟨child, mid, w, nextHistory, pushed, childValue, nextOk, earlier, inv, ⟨left,inside,leftOk,_,below⟩⟩ :=
      childRun q allowed (i+1) past ok
    obtain ⟨k, v, after, returned, finalOk, extension, ⟨right,tail,rightOk,issued,earlierChildren,later⟩⟩ :=
      ih w mid nextHistory nextOk
    let loads := fun j => if j ≤ mid then left j else right j
    have first := below.openings.2
    have eq : body c (source inputs left) (i+1) child = body c (source inputs loads) (i+1) child :=
      body_congr _ (by simp [source,loads,first])
    have leftRun := change_source (after := loads) below (fun j _ upper => by simp [loads,upper])
    rw [eq] at leftRun
    refine ⟨k,v,after,by simp [interpret,pushed,childValue,returned],finalOk,earlier.trans extension,
      ⟨loads,.node (i+1) q w inside :: tail,?_,.call issued,?_,?_⟩⟩
    · intro j; dsimp [loads]; split <;> first | apply leftOk | apply rightOk
    · intro item h
      rcases List.mem_cons.mp h with rfl | laterMember
      · exact ⟨child,pushed,inv⟩
      · exact earlierChildren _ laterMember
    · intro f
      exact .call pushed leftRun (change_source (after := loads) (later f)
        (fun j bound _ => by simp [loads,Nat.not_le_of_lt bound]))

private theorem map_id (p : Program Req Int) : p.map id = p := by
  induction p with
  | pure | fail => rfl
  | call req k ih => exact congrArg (Program.call req) (funext ih)

theorem SourceProgram.asFrame {inputs : Inputs mc fc} {c : SerialScout.Config}
    (replayed : SourceProgram c inputs i ctx (body c (source inputs (fun _ => past)) i ctx) k value)
    (ok : soundHistory mc fc past) : SourceFrame c inputs i ctx past k value := by
  obtain ⟨loads, forest, good, _, _, all_⟩ := replayed.execution
  let fixed := fun j => if j = i then past else loads j
  have completed := all_ id
  rw [map_id] at completed
  have changeAt := change_source (after := fixed) completed (fun j beyond _ => by simp [fixed,Nat.ne_of_gt beyond])
  have eq : body c (source inputs (fun _ => past)) i ctx = body c (source inputs fixed) i ctx :=
    body_congr _ (by simp [source, fixed])
  rw [eq] at changeAt
  refine ⟨fixed,forest,?_,by simp [fixed],changeAt⟩
  intro j; dsimp [fixed]; split <;> first | exact ok | apply good

end NeuralReversi.PvsParallel.SerialCacheStack

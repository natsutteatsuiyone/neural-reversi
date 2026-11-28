module
public import NeuralReversi.PvsParallel.SerialCalls.Stack

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialCalls

namespace Program

/-- Every possible return word from an issued child leaves the caller able to
    continue. A failure instruction has no constructor. -/
inductive Ready {Req α : Type} (allowed : Req → Prop) : Program Req α → Prop
  | pure : Ready allowed (.pure v)
  | call (validRequest : allowed req) (more : ∀ word : Int, Ready allowed (resume word)) :
      Ready allowed (.call req resume)

theorem Ready.hasRun (live : Ready allowed p) : ∃ calls out, Runs p calls out := by
  induction live with
  | pure => exact ⟨[], _, .pure⟩
  | call _ _ ih =>
    obtain ⟨cs, v, ran⟩ := ih 0
    exact ⟨_, v, .call ran⟩

theorem Ready.of_runs {allowed : Req → Prop} {p : Program Req α} (live : Ready (fun _ => True) p)
    (issued : ∀ cs v, Runs p cs v → ∀ t ∈ cs, allowed t.1) : Ready allowed p := by
  induction live with
  | pure => exact .pure
  | @call q resume _ more ih =>
    obtain ⟨cs, v, ran⟩ := (more 0).hasRun
    refine .call (issued _ _ (.call ran) _ List.mem_cons_self) ?_
    intro word
    exact ih word fun _ _ tail t h => issued _ _ (.call tail) t (.tail _ h)

theorem Ready.weaken {p : Program Req α} {small large : Req → Prop}
    (live : Ready small p) (implies : ∀ q, small q → large q) : Ready large p := by
  induction live with
  | pure => exact .pure
  | call ok _ ih => exact .call (implies _ ok) ih

theorem Ready.map {p : Program Req α} (live : Ready allowed p) (f : α → β) : Ready allowed (Program.map f p) := by
  induction live with
  | pure => exact .pure
  | call ok _ ih => exact .call ok ih

theorem Ready.mapRequests {p : Program Req α} (f : Req → Req') (live : Ready (allowed ∘ f) p) :
    Ready allowed (p.mapRequests f) := by
  induction live with
  | pure => exact .pure
  | call ok _ ih => exact .call ok ih

end Program

/-- Execute the current caller, carrying its updated issue counter from a child
    into the continuation. The induction ranges over this caller's instructions. -/
theorem executes_of_ready {body : Body Ctx Req} {push : Push Ctx Req} {ctx : Ctx} {p : Program Req Int}
    (live : Program.Ready allowed p)
    (enter : ∀ q, allowed q → ∀ i, ∃ child last word forest, push ctx q = some child ∧
      Executes body push (i+1) child (body (i+1) child) last word forest) (start : Nat) :
    ∃ last value forest, Executes body push start ctx p last value forest := by
  induction live generalizing start with
  | pure => exact ⟨_,_,_,.pure⟩
  | @call q resume validRequest _ ih =>
    obtain ⟨child,k,v,forest,entered,sub⟩ := enter q validRequest start
    obtain ⟨last,out,rest,after⟩ := ih v k
    exact ⟨_,_,_,.call entered sub after⟩

theorem Machine.steps_append (body : Body Ctx Req) (push : Push Ctx Req) (n m : Nat)
    (s : Machine Ctx Req) :
    s.steps body push (n+m) = (s.steps body push n).bind (Machine.steps body push m) := by
  induction n generalizing s with
  | zero => simp [Machine.steps]
  | succ n ih =>
    rw [Nat.succ_add, Machine.steps, Machine.steps, Option.bind_assoc]
    exact congrArg (Option.bind (s.step body push)) (funext (ih ·))

private theorem Executes.toPure {body : Body Ctx Req} {push : Push Ctx Req}
    (ran : Executes body push start ctx code last word forest) (stack : List (Saved Ctx Req)) :
    ∃ count, (Machine.running start ctx code stack).steps body push count =
      some (.running last ctx (.pure word) stack) := by
  induction ran generalizing stack with
  | pure => exact ⟨0, rfl⟩
  | @call parent req child i k v inside resume last out rest entered _ _ sub next =>
    obtain ⟨n, execute⟩ := sub (⟨parent,resume⟩ :: stack)
    obtain ⟨m, after⟩ := next stack
    refine ⟨(n+(m+1))+1, ?_⟩
    rw [Machine.steps]
    simp only [Machine.step, entered]
    change (Machine.running (i+1) child (body (i+1) child) (⟨parent,resume⟩ :: stack)).steps body push (n+(m+1)) = _
    rw [Machine.steps_append, execute]
    exact after

theorem Executes.to_steps (ran : Executes body push start ctx code last word forest) :
    ∃ count, (Machine.running start ctx code []).steps body push count = some (.finished last word) := by
  obtain ⟨n, upto⟩ := ran.toPure []
  refine ⟨n+1, ?_⟩
  rw [Machine.steps_append, upto]
  rfl

end NeuralReversi.PvsParallel.SerialCalls

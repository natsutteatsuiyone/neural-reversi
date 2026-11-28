module
public import NeuralReversi.PvsParallel.SerialCalls.Ready

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialCalls

/-- Step until the last activation returns, retaining the final child-opening counter. -/
def Machine.drive (body : Body Ctx Req) (push : Push Ctx Req) (s : Machine Ctx Req) :
    Option (Nat × Int) :=
  match s with
  | .finished last value => some (last, value)
  | .running .. =>
    match s.step body push with
    | some next => Machine.drive body push next
    | none => none
partial_fixpoint

theorem Machine.drive_of_steps {body : Body Ctx Req} {push : Push Ctx Req} {s : Machine Ctx Req}
    (ran : s.steps body push count = some (.finished last word)) : s.drive body push = some (last,word) := by
  induction count generalizing s with
  | zero =>
    cases Option.some.inj ran
    rw [drive]
  | succ n ih =>
    obtain ⟨next, edge, tail⟩ := Option.bind_eq_some_iff.mp ran
    cases s with
    | finished => contradiction
    | running =>
      rw [drive, edge]
      exact ih tail

theorem Executes.drives (ran : Executes body push start ctx p last word forest) :
    (Machine.running start ctx p []).drive body push = some (last, word) := by
  obtain ⟨count, finite⟩ := ran.to_steps
  exact Machine.drive_of_steps finite

/-- Preserve the outer frame's result type while each recursive child returns a signed score. -/
def outer (body : Body Ctx Req) (push : Push Ctx Req) (i : Nat) (ctx : Ctx) :
    Program Req α → Option (Nat × α)
  | .pure answer => some (i, answer)
  | .fail => none
  | .call q resume => do
    let child ← push ctx q
    let (after, word) ← (Machine.running (i+1) child (body (i+1) child) []).drive body push
    outer body push after ctx (resume word)

theorem Executes.outer_returns {p : Program Req α} {f : α → Int}
    (ran : Executes body push start ctx (p.map f) last value forest) :
    ∃ answer, outer body push start ctx p = some (last, answer) ∧ f answer = value ∧
      Program.Runs p (forest.map Tree.direct) answer := by
  induction p generalizing start last value forest with
  | fail => cases ran
  | pure answer => cases ran; exact ⟨answer, rfl, rfl, .pure⟩
  | call q resume ih =>
    cases ran with
    | call entered sub next =>
      obtain ⟨answer, after, observed, issued⟩ := ih _ next
      exact ⟨answer, by simp [outer, entered, sub.drives, after], observed, .call issued⟩

end NeuralReversi.PvsParallel.SerialCalls

module
public import NeuralReversi.PvsParallel.SerialCalls

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialCalls

/-- A child call is identified by its issue number on the single worker. The
    same board, window or move can appear in any number of distinct activations. -/
inductive Tree (Req : Type) where
  | node (opening : Nat) (request : Req) (word : Int) (children : List (Tree Req))

def Tree.direct : Tree Req → Req × Int
  | .node _ req word _ => (req, word)

abbrev Body (Ctx Req : Type) := Nat → Ctx → Program Req Int
abbrev Push (Ctx Req : Type) := Ctx → Req → Option Ctx

/-- The worker suspends the parent at a call and starts the child's body. The
    only operation supplying the continuation's word is the child's return. -/
structure Saved (Ctx Req : Type) where
  parent : Ctx
  resume : Int → Program Req Int

inductive Machine (Ctx Req : Type) where
  | running (clock : Nat) (context : Ctx) (code : Program Req Int) (stack : List (Saved Ctx Req))
  | finished (clock : Nat) (value : Int)

def Machine.step (body : Body Ctx Req) (push : Push Ctx Req) :
    Machine Ctx Req → Option (Machine Ctx Req)
  | .running n ctx (.call req resume) stack => do
    let child ← push ctx req
    return .running (n+1) child (body (n+1) child) (⟨ctx, resume⟩ :: stack)
  | .running n _ (.pure word) (saved :: stack) => some (.running n saved.parent (saved.resume word) stack)
  | .running n _ (.pure word) [] => some (.finished n word)
  | _ => none

def Machine.steps (body : Body Ctx Req) (push : Push Ctx Req) :
    Nat → Machine Ctx Req → Option (Machine Ctx Req)
  | 0, s => some s
  | n+1, s => (s.step body push).bind (steps body push n)

/-- Call trees are output by the instructions. Each child tree has the exact
    word passed to its caller and all of that child's earlier descendants. -/
inductive Executes (body : Body Ctx Req) (push : Push Ctx Req) :
    Nat → Ctx → Program Req Int → Nat → Int → List (Tree Req) → Prop
  | pure : Executes body push n ctx (.pure value) n value []
  | call (entered : push ctx req = some child)
      (sub : Executes body push (n+1) child (body (n+1) child) middle word inside)
      (next : Executes body push middle ctx (resume word) final value rest) :
      Executes body push n ctx (.call req resume) final value (.node (n+1) req word inside :: rest)

mutual
  def Tree.openings : Tree Req → List Nat
    | .node i _ _ children => i :: openingNumbers children
  def openingNumbers : List (Tree Req) → List Nat
    | [] => []
    | t :: rest => t.openings ++ openingNumbers rest
end

theorem Executes.openings (ran : Executes body push n ctx p final value forest) :
    openingNumbers forest = List.range' (n+1) (final-n) ∧ n ≤ final := by
  induction ran with
  | pure => simp [openingNumbers]
  | @call ctx req child n middle word inside resume last value rest _ _ _ ih tail =>
    obtain ⟨sub, earlier⟩ := ih
    obtain ⟨after, later⟩ := tail
    refine ⟨?_, by omega⟩
    change (_ :: openingNumbers _) ++ openingNumbers _ = _
    rw [sub, after]
    have steps : last - n = 1 + (middle - (n+1)) + (last - middle) := by omega
    rw [steps, ← List.range'_append (s := n+1) (m := 1+(middle-(n+1))) (step := 1), Nat.one_add, List.range'_succ]
    simp only [List.cons_append, List.cons.injEq, true_and]
    congr 2 <;> omega

theorem Executes.directCalls (ran : Executes body push n ctx p final value forest) :
    Program.Runs p (forest.map Tree.direct) value := by
  induction ran with
  | pure => exact .pure
  | call _ _ _ _ ih => exact .call ih

private def Unwind (body : Body Ctx Req) (push : Push Ctx Req) (last : Nat) (value : Int) :
    List (Saved Ctx Req) → Nat → Int → Prop
  | [], n, word => n = last ∧ word = value
  | saved :: stack, n, word => ∃ mid v forest,
    Executes body push n saved.parent (saved.resume word) mid v forest ∧ Unwind body push last value stack mid v

private def Completion (body : Body Ctx Req) (push : Push Ctx Req) (last : Nat) (value : Int) :
    Machine Ctx Req → Prop
  | .finished n v => n = last ∧ v = value
  | .running n ctx p stack => ∃ mid v forest,
    Executes body push n ctx p mid v forest ∧ Unwind body push last value stack mid v

private theorem step_back {body : Body Ctx Req} {push : Push Ctx Req} {s t : Machine Ctx Req}
    (edge : s.step body push = some t) (result : Completion body push k out t) :
    Completion body push k out s := by
  cases s with
  | finished => contradiction
  | running n ctx p stack =>
    cases p with
    | fail => contradiction
    | pure word =>
      cases stack with
      | nil =>
        cases Option.some.inj edge
        exact ⟨n, word, [], .pure, result⟩
      | cons saved stack =>
        cases Option.some.inj edge
        exact ⟨n, word, [], .pure, result⟩
    | call req resume =>
      cases enter : push ctx req with
      | none => simp [Machine.step, enter] at edge
      | some child =>
        simp [Machine.step, enter] at edge
        subst t
        obtain ⟨middle, word, inside, sub, final, v, rest, next, after⟩ := result
        exact ⟨final, v, _, .call enter sub next, after⟩

private theorem steps_back {body : Body Ctx Req} {push : Push Ctx Req} (count : Nat)
    (s : Machine Ctx Req) (ran : s.steps body push count = some (.finished k out)) :
    Completion body push k out s := by
  induction count generalizing s with
  | zero =>
    cases Option.some.inj ran
    exact ⟨rfl, rfl⟩
  | succ n ih =>
    obtain ⟨t, edge, rest⟩ := Option.bind_eq_some_iff.mp ran
    exact step_back edge (ih t rest)

/-- Every finite normal stack execution supplies a well-nested return record.
    The input specifies the entry state, the number of instructions, and the
    final observable word. No per-child result or checked record is needed. -/
theorem record_program_of_steps (body : Body Ctx Req) (push : Push Ctx Req) (ctx : Ctx) {p : Program Req Int}
    (ran : (Machine.running start ctx p []).steps body push count = some (.finished k out)) :
    ∃ forest, Executes body push start ctx p k out forest := by
  obtain ⟨m, word, forest, exec, times, values⟩ := steps_back count _ ran
  exact ⟨forest, times ▸ values ▸ exec⟩

/-- An induction over completed calls certifies precisely the children issued
    by a frame. The statement permits a fresh set of operands at every opening. -/
theorem Executes.children {body : Body Ctx Req} {push : Push Ctx Req}
    (invariant : Ctx → Int → Prop)
    (localRule : ∀ i ctx children word, Program.Runs (body i ctx) children word →
      (∀ r ∈ children, ∃ child, push ctx r.1 = some child ∧ invariant child r.2) → invariant ctx word)
    (ran : Executes body push n ctx p final v forest) :
    ∀ r ∈ forest.map Tree.direct, ∃ child, push ctx r.1 = some child ∧ invariant child r.2 := by
  induction ran with
  | pure => simp
  | call entered sub _ ih more =>
    intro r mem
    rcases List.mem_cons.mp mem with rfl | rest
    · exact ⟨_, entered, localRule _ _ _ _ sub.directCalls ih⟩
    · exact more r rest

end NeuralReversi.PvsParallel.SerialCalls

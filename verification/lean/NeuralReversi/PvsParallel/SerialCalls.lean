import NeuralReversi.PvsParallel.ExactDepth

namespace NeuralReversi.PvsParallel.SerialCalls

structure Request where
  board : PvsBoardTree.Position
  depth : ExactDepth.Depth
  pv : Bool
  cut : Bool
  alpha : Int
  beta : Int
  deriving DecidableEq

structure MoveRequest (Move : Type) where
  move : Move
  pv : Bool
  alpha : Int
  beta : Int
  deriving DecidableEq

/-- One suspended caller. Its continuation receives the return word from the
    requested child before it can decide which instruction to execute next. -/
inductive Program (Req α : Type) where
  | pure (value : α)
  | fail
  | call (request : Req) (resume : Int → Program Req α)

namespace Program

def eval (reply : Req → Option Int) : Program Req α → Option α
  | .pure value => some value
  | .fail => none
  | .call req resume => do eval reply (resume (← reply req))

/-- Direct child returns, in issue order. After receiving a word the caller
    continues from that instruction's destination. -/
inductive Runs : Program Req α → List (Req × Int) → α → Prop
  | pure : Runs (.pure value) [] value
  | call (more : Runs (resume word) rest value) :
      Runs (.call req resume) ((req, word) :: rest) value

theorem Runs.eval {p : Program Req α} (ran : Runs p children value) (reply : Req → Option Int)
    (available : ∀ r ∈ children, reply r.1 = some r.2) : p.eval reply = some value := by
  induction ran with
  | pure => rfl
  | call _ ih =>
    rw [Program.eval, available _ List.mem_cons_self]
    exact ih fun r h => available r (List.mem_cons_of_mem _ h)

theorem runs_of_eval (p : Program Req α) (reply : Req → Option Int) (got : p.eval reply = some v) :
    ∃ children, Runs p children v ∧ ∀ r ∈ children, reply r.1 = some r.2 := by
  induction p with
  | fail => contradiction
  | pure => exact ⟨[], Option.some.inj got ▸ .pure, by simp⟩
  | call req resume ih =>
    cases h : reply req with
    | none => simp [eval, h] at got
    | some word =>
      obtain ⟨rest, more, supplied⟩ := ih word (by simpa [eval, h] using got)
      refine ⟨(req, word) :: rest, .call more, ?_⟩
      intro r mem
      rcases List.mem_cons.mp mem with rfl | later
      · exact h
      · exact supplied r later

def ofOption : Option α → Program Req α
  | some value => .pure value
  | none => .fail

theorem runs_ofOption (ran : Runs (ofOption x : Program Req α) children v) :
    x = some v ∧ children = [] := by
  cases x <;> cases ran
  exact ⟨rfl, rfl⟩

def map (f : α → β) : Program Req α → Program Req β
  | .pure value => .pure (f value)
  | .fail => .fail
  | .call req resume => .call req fun word => (resume word).map f

theorem eval_map (p : Program Req α) (f : α → β) (reply : Req → Option Int) :
    (p.map f).eval reply = (p.eval reply).map f := by
  induction p with
  | pure | fail => rfl
  | call req resume ih =>
    simp only [Program.map, eval]
    cases h : reply req with
    | none => rfl
    | some word => exact ih word

theorem runs_map {p : Program Req α} {f : α → β} (ran : Runs (p.map f) children v) :
    ∃ x, Runs p children x ∧ f x = v := by
  induction p generalizing children v with
  | fail => cases ran
  | pure x => cases ran; exact ⟨x, .pure, rfl⟩
  | call req resume ih =>
    cases ran with
    | call more =>
      obtain ⟨x, run, rfl⟩ := ih _ more
      exact ⟨x, .call run, rfl⟩

def mapRequests (f : Req → Req') : Program Req α → Program Req' α
  | .pure value => .pure value
  | .fail => .fail
  | .call req resume => .call (f req) fun word => (resume word).mapRequests f

theorem runs_mapRequests {p : Program Req α} {f : Req → Req'} (ran : Runs (p.mapRequests f) children v) :
    ∃ original, Runs p original v ∧ original.map (fun r => (f r.1, r.2)) = children := by
  induction p generalizing children v with
  | fail => cases ran
  | pure => cases ran; exact ⟨[], .pure, rfl⟩
  | call req resume ih =>
    cases ran with
    | call more =>
      obtain ⟨rest, run, rfl⟩ := ih _ more
      exact ⟨_ :: rest, .call run, rfl⟩

theorem lookup_mem [DecidableEq Req] {calls : List (Req × α)} (unique : (calls.map Prod.fst).Nodup)
    (present : (q, w) ∈ calls) : calls.lookup q = some w := by
  induction calls with
  | nil => cases present
  | cons h tail ih =>
    rcases List.mem_cons.mp present with rfl | later
    · simp [List.lookup]
    · have rest := List.nodup_cons.mp unique
      have ne : h.1 ≠ q := by
        intro same
        exact rest.1 (same ▸ List.mem_map_of_mem (f := Prod.fst) later)
      simpa [List.lookup, beq_eq_false_iff_ne.mpr (Ne.symm ne)] using ih rest.2 later

def returnAt [DecidableEq Key] (key : Req → Key) (calls : List (Req × Int)) (q : Key) : Int :=
  ((calls.map fun r => (key r.1, r.2)).lookup q).getD 0

theorem returnAt_mem [DecidableEq Key] (key : Req → Key) (calls : List (Req × Int))
    (unique : (calls.map fun r => key r.1).Nodup) (r : Req × Int) (h : r ∈ calls) :
    returnAt key calls (key r.1) = r.2 := by
  unfold returnAt
  rw [lookup_mem (by simpa [List.map_map, Function.comp_def] using unique)
    (List.mem_map_of_mem (f := fun x => (key x.1, x.2)) h)]
  rfl

theorem eval_mapRequests (p : Program Req α) (f : Req → Req') (reply : Req' → Option Int) :
    (p.mapRequests f).eval reply = p.eval (reply ∘ f) := by
  induction p with
  | pure | fail => rfl
  | call req resume ih =>
    simp only [mapRequests, eval, Function.comp_apply]
    cases h : reply (f req) with
    | none => rfl
    | some word => exact ih word

end Program
end NeuralReversi.PvsParallel.SerialCalls

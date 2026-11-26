import Std

namespace NeuralReversi.PvsPostpone

variable {α : Type}

def shouldPostpone (busy : Bool) (idx freshEnd : Nat) : Bool :=
  busy && decide (idx ≠ 0) && decide (idx + 1 < freshEnd)

structure Queue (α : Type) where
  idx : Nat
  fresh : List α
  postponed : List α

def Queue.pending (q : Queue α) : List α := q.fresh ++ q.postponed
def Queue.budget (q : Queue α) : Nat := q.pending.length + q.fresh.length

def step (busy : Bool) (q : Queue α) : Option (Option α × Queue α) :=
  match q.fresh, q.postponed with
  | [], [] => none
  | [], move :: rest => some (some move, ⟨q.idx + 1, [], rest⟩)
  | move :: rest, postponed =>
    if shouldPostpone busy q.idx (q.idx + q.fresh.length) then
      some (none, ⟨q.idx, rest, postponed ++ [move]⟩)
    else
      some (some move, ⟨q.idx + 1, rest, postponed⟩)

def StepContract : Prop :=
  ∀ (α : Type),
  (∀ (q : Queue α) busy freshEnd, q.fresh.length = freshEnd - q.idx →
    (shouldPostpone busy q.idx freshEnd = true ↔
      busy = true ∧ q.idx ≠ 0 ∧ 2 ≤ q.fresh.length)) ∧
  (∀ (q : Queue α) busy, step busy q = none ↔ q.pending = []) ∧
  (∀ (q : Queue α) busy taken next, step busy q = some (taken, next) →
    List.Perm (taken.toList ++ next.pending) q.pending ∧
    next.budget < q.budget ∧
    (taken = none → q.idx = next.idx ∧ next.fresh.length + 1 = q.fresh.length ∧
      next.postponed.length = q.postponed.length + 1))

theorem stepContract_proven : StepContract := by
  intro α
  refine ⟨?_, ?_, ?_⟩
  · intro q busy freshEnd length
    simp only [shouldPostpone, Bool.and_eq_true, decide_eq_true_eq]
    constructor
    · rintro ⟨⟨hb, eldest⟩, fresh⟩
      exact ⟨hb, eldest, by omega⟩
    · rintro ⟨hb, eldest, fresh⟩
      exact ⟨⟨hb, eldest⟩, by omega⟩
  · rintro ⟨idx, fresh, postponed⟩ busy
    cases fresh with
    | nil => cases postponed <;> simp [step, Queue.pending]
    | cons move rest =>
      simp only [step, Queue.pending, List.cons_append, List.cons_ne_nil, iff_false]
      split <;> simp
  · rintro ⟨idx, fresh, postponed⟩ busy taken next h
    cases fresh with
    | nil =>
      cases postponed with
      | nil => simp [step] at h
      | cons move rest =>
        cases h
        simp [Queue.pending, Queue.budget]
    | cons move rest =>
      simp only [step] at h
      split at h
      · cases h
        refine ⟨?_, ?_, by simp⟩
        · simpa [Queue.pending, List.append_assoc] using
            List.perm_append_singleton move (rest ++ postponed)
        · simp [Queue.budget, Queue.pending]
          omega
      · cases h
        simp [Queue.budget, Queue.pending]
        omega

theorem step_idx (advance : step busy q = some (taken, next)) :
    next.idx = q.idx + taken.toList.length := by
  rcases q with ⟨idx, fresh, postponed⟩
  cases fresh with
  | nil =>
    cases postponed <;> simp [step] at advance
    cases advance
    subst taken
    subst next
    rfl
  | cons move rest =>
    simp only [step] at advance
    split at advance <;> cases advance <;> rfl

inductive Run (busy : Nat → Bool) : Nat → Queue α → List α → Nat → Prop
  | done (empty : q.pending = []) : Run busy iteration q [] 0
  | next (advance : step (busy iteration) q = some (taken, next)) :
      Run busy (iteration + 1) next visited steps →
      Run busy iteration q (taken.toList ++ visited) (steps + 1)

private theorem run_exists (busy : Nat → Bool) (iteration : Nat) (q : Queue α) :
    ∃ visited steps, Run busy iteration q visited steps := by
  cases h : step (busy iteration) q with
  | none => exact ⟨[], 0, .done ((stepContract_proven α).2.1 q _ |>.mp h)⟩
  | some result =>
    obtain ⟨taken, next⟩ := result
    have decreases := ((stepContract_proven α).2.2 q _ taken next h).2.1
    obtain ⟨visited, steps, run⟩ := run_exists busy (iteration + 1) next
    exact ⟨taken.toList ++ visited, steps + 1, .next h run⟩
termination_by q.budget

private theorem run_preserves (run : Run busy iteration q visited steps) :
    List.Perm visited q.pending ∧ steps ≤ q.budget := by
  induction run with
  | done empty =>
    rw [empty]
    exact ⟨List.Perm.refl _, Nat.zero_le _⟩
  | next advance _ ih =>
    obtain ⟨perm, decrease, _⟩ := (stepContract_proven _).2.2 _ _ _ _ advance
    exact ⟨(ih.1.append_left _).trans perm, by omega⟩

def CompletionContract : Prop :=
  ∀ (α : Type) (busy : Nat → Bool) idx (moves : List α),
    let q : Queue α := ⟨idx, moves, []⟩
    (∃ visited steps, Run busy 0 q visited steps) ∧
    ∀ visited steps, Run busy 0 q visited steps →
      List.Perm visited moves ∧ steps ≤ 2 * moves.length

theorem completionContract_proven : CompletionContract := by
  intro α busy idx moves q
  refine ⟨run_exists busy 0 q, ?_⟩
  intro visited steps run
  have result := run_preserves run
  exact ⟨by simpa [q, Queue.pending] using result.1,
    by simpa [q, Queue.budget, Queue.pending, Nat.two_mul] using result.2⟩

end NeuralReversi.PvsPostpone

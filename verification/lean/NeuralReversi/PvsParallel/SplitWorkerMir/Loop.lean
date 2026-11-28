module
public import NeuralReversi.PvsParallel.SplitWorkerMir.Dispatch

@[expose] public section

namespace NeuralReversi.PvsParallel.SplitWorkerMir

open Threading
open SplitReturn (Work Departure Loop)
open SplitResult (Report)
open MoveList.Concurrent (advance Claims)

structure Ticket where
  input : Input
  locked : Locked

def work : List Ticket → Option Work
  | [] => some ⟨[], [], emptyExit⟩
  | t :: rest => do
    let body ← search t.input
    let r := selected t.input body
    match (commit t.input t.locked body.chosen).exit with
    | .stopped => return ⟨[t.input.index], [], .cancelled⟩
    | .cutoff => return ⟨[t.input.index], [r], .cutoff⟩
    | .again =>
      let tail ← work rest
      return ⟨t.input.index :: tail.tickets, r :: tail.reports, tail.departure⟩

theorem finished (tickets : List Ticket) {actual : Work} (ran : work tickets = some actual)
    (len : Nat) (last : BitVec w)
    (exhausted : actual.departure = .exhausted →
      ∃ seen after, advance len seen = none ∧ Claims len seen after last) :
    Loop len last actual.tickets actual.reports actual.departure := by
  induction tickets generalizing actual with
  | nil =>
    cases ran
    obtain ⟨_, _, atEnd, intervening⟩ := exhausted rfl
    exact .exhausted atEnd intervening
  | cons t rest ih =>
    unfold work at ran
    cases searched : search t.input with
    | none => simp [searched] at ran
    | some body =>
      rw [searched] at ran
      cases ending : (commit t.input t.locked body.chosen).exit with
      | stopped =>
        simp [ending] at ran
        subst actual
        exact .cancelled (some t.input.index)
      | cutoff =>
        simp [ending] at ran
        subst actual
        exact .publish (selected t.input body) (.cutoff none)
      | again =>
        cases suffixResult : work rest with
        | none => simp [ending, suffixResult] at ran
        | some tail =>
          simp [ending, suffixResult] at ran
          subst actual
          exact .publish (selected t.input body) (ih suffixResult exhausted)

theorem cutoff_flag (i : Input) (s : Locked) (v : Word) (running : s.stopped = false) :
    (commit i s v).cut =
      SplitWindow.marksCutoff i.pv i.beta.toInt ⟨s.alpha.toInt, s.best.toInt⟩ v.toInt := by
  have scale : maximum.toInt = PvsMoveLoop.scoreMax := by decide
  rw [commit_eq]
  cases pv : i.pv <;> by_cases b : s.best.toInt < v.toInt <;> by_cases a : s.alpha.toInt < v.toInt <;>
    by_cases lo : v.toInt < i.beta.toInt <;> by_cases m : PvsMoveLoop.scoreMax ≤ v.toInt <;>
    simp [SplitWindow.marksCutoff, *] <;> omega

theorem cutoff_marks (i : Input) (s : Locked) (v : Word)
    (ends : (commit i s v).exit = .cutoff) :
    SplitWindow.marksCutoff i.pv i.beta.toInt ⟨s.alpha.toInt, s.best.toInt⟩ v.toInt = true := by
  cases h : s.stopped
  · rw [← cutoff_flag i s v h]
    rw [commit_eq] at ends ⊢
    simp only [h, Bool.false_eq_true, ite_false] at ends ⊢
    split at ends <;> first | assumption | contradiction
  · rw [commit_eq] at ends
    simp [h] at ends

theorem localCutoff {earlier later all : List Report} (i : Input) (s : Locked) (body : Searched)
    (initial : SplitWindow.State) (ordered : all = earlier ++ selected i body :: later)
    (visible : SplitReturn.stateAfter i.pv i.beta.toInt initial earlier = ⟨s.alpha.toInt, s.best.toInt⟩)
    (ends : (commit i s body.chosen).exit = .cutoff) :
    SplitReturn.LocalCutoff i.pv i.beta.toInt initial all := by
  exact ⟨earlier, selected i body, later, ordered, visible ▸ cutoff_marks i s body.chosen ends⟩

end NeuralReversi.PvsParallel.SplitWorkerMir

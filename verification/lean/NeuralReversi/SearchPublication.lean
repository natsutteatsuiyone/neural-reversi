module
public import NeuralReversi.Bitboard.Model

@[expose] public section

namespace NeuralReversi.SearchPublication

open Bitboard (Square)

structure Stamp where
  depth : Nat
  selectivity : Nat
  deriving DecidableEq

structure Row where
  square : Square
  score : Int
  pv : List Square
  deriving DecidableEq

structure Snapshot where
  rows : List Row
  stamp : Stamp
  endgame : Bool
  deriving DecidableEq

def capture (rows : List Row) (stamp : Stamp) (endgame : Bool) : Snapshot :=
  ⟨rows.mergeSort (fun a b => decide (b.score ≤ a.score)), stamp, endgame⟩

def CaptureContract : Prop :=
  ∀ (rows : List Row) (stamp : Stamp) (endgame : Bool),
    let saved := capture rows stamp endgame
    saved.stamp = stamp ∧ saved.endgame = endgame ∧
      saved.rows.Perm rows ∧ saved.rows.Pairwise (fun a b => b.score ≤ a.score) ∧
      (∀ a b, a.score = b.score → [a, b].Sublist rows → [a, b].Sublist saved.rows)

theorem captureContract_proven : CaptureContract := by
  intro rows stamp endgame saved
  let le := fun (a b : Row) => decide (b.score ≤ a.score)
  have trans (a b c : Row) (ab : le a b) (bc : le b c) : le a c := by
    simp only [le, decide_eq_true_eq] at ab bc ⊢
    omega
  have total (a b : Row) : le a b || le b a := by
    simp only [le, Bool.or_eq_true, decide_eq_true_eq]
    exact Int.le_total _ _
  exact ⟨rfl, rfl, List.mergeSort_perm rows le,
    (List.pairwise_mergeSort trans total rows).imp of_decide_eq_true,
    fun a b tied inOrder => List.pair_sublist_mergeSort trans total (by simp [tied]) inOrder⟩

structure Completed where
  marks : List (Option Stamp)
  last : Option Snapshot

def start (count : Nat) (inherited : Option Snapshot) : Completed :=
  ⟨List.replicate count none, inherited⟩

def beginPhase (state : Completed) : Completed :=
  start state.marks.length state.last

structure Attempt where
  index : Nat
  stamp : Stamp
  endgame : Bool
  rows : List Row
  aborted : Bool

def reportable (minimum : Nat) (attempt : Attempt) : Bool :=
  !attempt.aborted && decide (minimum ≤ attempt.stamp.selectivity)

def nextMarks (minimum : Nat) (marks : List (Option Stamp)) (attempt : Attempt) : List (Option Stamp) :=
  if reportable minimum attempt then marks.set attempt.index (some attempt.stamp) else marks

def ready (minimum : Nat) (marks : List (Option Stamp)) (attempt : Attempt) : Bool :=
  reportable minimum attempt &&
    (nextMarks minimum marks attempt).all (fun level => decide (level = some attempt.stamp))

def published (minimum : Nat) (marks : List (Option Stamp)) (attempt : Attempt) : Option Snapshot :=
  if ready minimum marks attempt then some (capture attempt.rows attempt.stamp attempt.endgame) else none

def step (minimum : Nat) (state : Completed) (attempt : Attempt) : Completed :=
  ⟨nextMarks minimum state.marks attempt, (published minimum state.marks attempt).or state.last⟩

def run (minimum : Nat) (state : Completed) (attempts : List Attempt) : Completed :=
  attempts.foldl (step minimum) state

def captures (minimum : Nat) : List (Option Stamp) → List Attempt → List Snapshot
  | _, [] => []
  | marks, attempt :: rest =>
    (published minimum marks attempt).toList ++ captures minimum (nextMarks minimum marks attempt) rest

private theorem ready_iff :
    ready minimum marks attempt = true ↔
      attempt.aborted = false ∧ minimum ≤ attempt.stamp.selectivity ∧
        ∀ mark ∈ marks.set attempt.index (some attempt.stamp), mark = some attempt.stamp := by
  by_cases counts : reportable minimum attempt = true
  · have facts : attempt.aborted = false ∧ minimum ≤ attempt.stamp.selectivity := by
      simpa only [reportable, Bool.and_eq_true, Bool.not_eq_true', decide_eq_true_eq] using counts
    simp [ready, nextMarks, counts, facts.1, facts.2]
  · constructor
    · simp [ready, counts]
    · rintro ⟨done, level, _⟩
      exact False.elim (counts (by simp [reportable, done, level]))

def ProgressContract : Prop :=
  ∀ (minimum : Nat) (state : Completed) (attempt : Attempt),
    attempt.index < state.marks.length →
    let next := step minimum state attempt
    next.marks.length = state.marks.length ∧
      (attempt.aborted = true ∨ attempt.stamp.selectivity < minimum → next = state) ∧
      (reportable minimum attempt = true →
        next.marks[attempt.index]? = some (some attempt.stamp) ∧
        ∀ i, i ≠ attempt.index → next.marks[i]? = state.marks[i]?) ∧
      ((∃ saved, published minimum state.marks attempt = some saved) ↔
        attempt.aborted = false ∧ minimum ≤ attempt.stamp.selectivity ∧
          ∀ mark ∈ next.marks, mark = some attempt.stamp) ∧
      (ready minimum state.marks attempt = false → next.last = state.last) ∧
      (ready minimum state.marks attempt = true →
        next.last = some (capture attempt.rows attempt.stamp attempt.endgame))

theorem progressContract_proven : ProgressContract := by
  intro minimum state attempt inside next
  have counts_iff : reportable minimum attempt = true ↔
      attempt.aborted = false ∧ minimum ≤ attempt.stamp.selectivity := by
    simp [reportable]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [next, step, nextMarks]
    split <;> simp
  · rintro (aborted | below)
    · cases state
      simp [next, step, published, ready, nextMarks, reportable, aborted]
    · cases state
      simp [next, step, published, ready, nextMarks, reportable, Nat.not_le_of_gt below]
  · intro counts
    change (nextMarks minimum state.marks attempt)[attempt.index]? = _ ∧
      ∀ i, i ≠ attempt.index → (nextMarks minimum state.marks attempt)[i]? = _
    simp only [nextMarks, counts, ite_true]
    exact ⟨List.getElem?_set_self inside, fun i different => List.getElem?_set_ne different.symm⟩
  · constructor
    · rintro ⟨saved, yes⟩
      have marked : ready minimum state.marks attempt = true := by
        cases h : ready minimum state.marks attempt <;> simp [published, h] at yes ⊢
      obtain ⟨done, level, all⟩ := ready_iff.mp marked
      exact ⟨done, level, by simpa only [next, step, nextMarks, counts_iff.mpr ⟨done, level⟩, ite_true] using all⟩
    · rintro ⟨done, level, all⟩
      have marked : ready minimum state.marks attempt = true := ready_iff.mpr
        ⟨done, level, by simpa only [next, step, nextMarks, counts_iff.mpr ⟨done, level⟩, ite_true] using all⟩
      exact ⟨capture attempt.rows attempt.stamp attempt.endgame, by simp [published, marked]⟩
  · intro no
    simp [next, step, published, no]
  · intro yes
    simp [next, step, published, yes]

def PhaseContract : Prop :=
  ∀ (state : Completed) (minimum : Nat) (attempt : Attempt),
    let phase := beginPhase state
    phase.marks = List.replicate state.marks.length none ∧ phase.last = state.last ∧
      (1 < state.marks.length → attempt.index < state.marks.length →
        (step minimum phase attempt).last = state.last)

theorem phaseContract_proven : PhaseContract := by
  intro state minimum attempt phase
  refine ⟨rfl, rfl, fun multiple valid => ?_⟩
  have size : phase.marks.length = state.marks.length := by simp [phase, beginPhase, start]
  have index : attempt.index < phase.marks.length := size.symm ▸ valid
  have another : ∃ i, i < phase.marks.length ∧ i ≠ attempt.index := by
    by_cases first : attempt.index = 0
    · exact ⟨1, size.symm ▸ multiple, by omega⟩
    · exact ⟨0, by omega, Ne.symm first⟩
  obtain ⟨i, inside, different⟩ := another
  have unchanged : phase.marks[i]? = some none := by
    have valid : i < state.marks.length := size ▸ inside
    simp [phase, beginPhase, start, valid]
  have progress := progressContract_proven minimum phase attempt index
  have blocked : ready minimum phase.marks attempt = false := by
    cases yes : ready minimum phase.marks attempt
    · rfl
    · obtain ⟨done, threshold, uniform⟩ := ready_iff.mp yes
      have updated := (progress.2.2.1 (by simp [reportable, done, threshold])).2 i different
      rw [unchanged] at updated
      have missing : none ∈ (step minimum phase attempt).marks := List.mem_of_getElem? updated
      have next := uniform none (by
        simpa only [step, nextMarks, show reportable minimum attempt = true by simp [reportable, done, threshold],
          ite_true] using missing)
      cases next
  exact progress.2.2.2.2.1 blocked

def LatestContract : Prop :=
  ∀ (minimum : Nat) (state : Completed) (attempts : List Attempt),
    (run minimum state attempts).last =
      (captures minimum state.marks attempts).getLast?.or state.last

theorem latestContract_proven : LatestContract := by
  intro minimum state attempts
  induction attempts generalizing state with
  | nil => rfl
  | cons attempt rest ih =>
    rw [run, List.foldl_cons, ← run, ih (step minimum state attempt), captures]
    simp only [step, List.getLast?_append, Option.or_assoc]
    cases published minimum state.marks attempt <;> rfl

inductive Exit where
  | interrupted
  | normal (stamp : Stamp) (rows : List Row)

def finish (minimum : Nat) (state : Completed) : Exit → Option Snapshot
  | .interrupted => state.last
  | .normal stamp rows => if minimum ≤ stamp.selectivity then some (capture rows stamp true) else state.last

def ExitContract : Prop :=
  ∀ (minimum : Nat) (state : Completed) (attempts : List Attempt) (stamp : Stamp) (rows : List Row),
    let recorded := run minimum state attempts
    let saved := (captures minimum state.marks attempts).getLast?.or state.last
    finish minimum recorded .interrupted = saved ∧
    (stamp.selectivity < minimum → finish minimum recorded (.normal stamp rows) = saved) ∧
    (minimum ≤ stamp.selectivity →
      finish minimum recorded (.normal stamp rows) = some (capture rows stamp true))

theorem exitContract_proven : ExitContract := by
  intro minimum state attempts stamp rows recorded saved
  have latest := latestContract_proven minimum state attempts
  exact ⟨latest, fun below => by simpa [finish, Nat.not_le_of_gt below] using latest,
    fun eligible => by simp [finish, eligible]⟩

def Unpublished (minimum : Nat) (state : Completed) (attempts : List Attempt) : Prop :=
  ∀ earlier attempt later, attempts = earlier ++ attempt :: later →
    ready minimum (run minimum state earlier).marks attempt = false

theorem unpublished_keeps_last (minimum : Nat) (state : Completed) (attempts : List Attempt)
    (noCapture : Unpublished minimum state attempts) : (run minimum state attempts).last = state.last := by
  induction attempts generalizing state with
  | nil => rfl
  | cons attempt rest ih =>
    have current : ready minimum state.marks attempt = false := noCapture [] attempt rest rfl
    have remaining : Unpublished minimum (step minimum state attempt) rest := by
      intro earlier next later partition
      exact noCapture (attempt :: earlier) next later (congrArg (attempt :: ·) partition)
    have last : (step minimum state attempt).last = state.last := by simp [step, published, current]
    exact (ih _ remaining).trans last

example :
    let initial := start 2 (some ⟨[⟨10, 5, [10, 18]⟩], ⟨7, 2⟩, false⟩)
    let pending : List Attempt := [
      ⟨0, ⟨9, 5⟩, true, [⟨10, -8, [10]⟩], false⟩,
      ⟨1, ⟨9, 2⟩, true, [⟨19, 1, [19]⟩], false⟩,
      ⟨1, ⟨9, 5⟩, true, [⟨19, 6, [19]⟩], true⟩]
    finish 2 (run 2 initial pending) .interrupted = initial.last := by decide

example :
    let inherited : Completed := ⟨[some ⟨9, 5⟩, some ⟨9, 5⟩], some ⟨[⟨10, 5, [10]⟩], ⟨9, 5⟩, false⟩⟩
    let first : Attempt := ⟨0, ⟨9, 5⟩, true, [⟨10, -8, [10]⟩], false⟩
    (step 2 (beginPhase inherited) first).last = inherited.last := by decide

end NeuralReversi.SearchPublication

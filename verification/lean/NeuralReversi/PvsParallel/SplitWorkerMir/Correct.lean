module
public import NeuralReversi.PvsParallel.SplitWorkerMir.Program
public import NeuralReversi.Solve2.Control

@[expose] public section

namespace NeuralReversi.PvsParallel.SplitWorkerMir

open Threading
open Threading.SplitResult (Report)

def scoutQuery (i : Input) : Query :=
  ⟨i.index, i.depth - 1, false, -(i.alpha + 1), -i.alpha, !i.cut, i.scout⟩

def wideQuery (i : Input) : Query :=
  ⟨i.index, i.depth - 1, true, -i.beta, -i.fresh, false, i.wide⟩

def retry (i : Input) : Bool := i.pv && decide (i.alpha.toInt < (-i.scout).toInt)

def issued (i : Input) : Searched :=
  if retry i then
    ⟨[.load false, .returned (scoutQuery i), .load true, .returned (wideQuery i), .lock], i.fresh, -i.wide⟩
  else ⟨[.load false, .returned (scoutQuery i), .lock], i.alpha, -i.scout⟩

theorem search_eq (i : Input) : search i =
    if i.depth = 0 ∨ (i.pv = true ∧ i.index = 0) then none else some (issued i) := by
  by_cases d : i.depth = 0 <;> cases pv : i.pv <;>
    cases idx : i.index <;> by_cases above : i.alpha.toInt < (-i.scout).toInt <;>
    simp [search, issued, retry, scoutQuery, wideQuery, *]

theorem issued_of_search {i : Input} {s : Searched} (hit : search i = some s) : s = issued i := by
  rw [search_eq] at hit
  split at hit
  · contradiction
  · exact (Option.some.inj hit).symm

structure Safe (i : Input) : Prop where
  alphaLow : -2147483648 < i.alpha.toInt
  alphaHigh : i.alpha.toInt < 2147483647
  scout : -2147483648 < i.scout.toInt
  beta : retry i = true → -2147483648 < i.beta.toInt
  fresh : retry i = true → -2147483648 < i.fresh.toInt
  wide : retry i = true → -2147483648 < i.wide.toInt

theorem retry_eq (safe : Safe i) : retry i = SplitWindow.reSearch i.pv i.alpha.toInt i.scout.toInt := by
  simp [retry, SplitWindow.reSearch, Solve2.neg_safe _ safe.scout]

def selected (i : Input) (s : Searched) : Report :=
  ⟨i.index, s.searchedAlpha.toInt, s.chosen.toInt⟩

theorem selected_eq (hit : search i = some s) (safe : Safe i) :
    selected i s = SplitWindow.selectReport i.pv i.index i.alpha.toInt i.fresh.toInt i.scout.toInt i.wide.toInt := by
  rw [issued_of_search hit, SplitWindow.selectReport, ← retry_eq safe]
  cases choice : retry i with
  | false => simp [issued, selected, choice, Solve2.neg_safe _ safe.scout]
  | true => simp [issued, selected, choice, Solve2.neg_safe _ (safe.wide choice)]

def lockedState (s : Locked) (c : Commit) : SplitWindow.State :=
  ⟨(c.alpha.getD s.alpha).toInt, (c.best.getD s.best).toInt⟩

theorem commit_eq (i : Input) (s : Locked) (chosen : Word) :
    commit i s chosen =
    if s.stopped then ⟨.stopped, none, none, false, none, false⟩ else
      let improves := decide (s.best.toInt < chosen.toInt)
      let above := decide (s.alpha.toInt < chosen.toInt)
      let picks := improves && above
      let raises := picks && i.pv && decide (chosen.toInt < i.beta.toInt)
      let cuts := picks && (!raises || decide (maximum.toInt ≤ chosen.toInt))
      ⟨if cuts then .cutoff else .again, if i.root then some (chosen, above) else none,
        if improves then some chosen else none, picks, if raises then some chosen else none, cuts⟩ := by
  cases stop : s.stopped <;> cases root : i.root <;> cases pv : i.pv <;>
    by_cases b : s.best.toInt < chosen.toInt <;> by_cases a : s.alpha.toInt < chosen.toInt <;>
    by_cases lo : chosen.toInt < i.beta.toInt <;> by_cases hi : maximum.toInt ≤ chosen.toInt <;>
    simp [commit, *, gt_iff_lt, ge_iff_le]

theorem publish_eq (i : Input) (s : Locked) (chosen : Word) (running : s.stopped = false) :
    lockedState s (commit i s chosen) =
      SplitWindow.publish i.pv i.beta.toInt ⟨s.alpha.toInt, s.best.toInt⟩ chosen.toInt := by
  rw [commit_eq]
  cases pv : i.pv <;> by_cases b : s.best.toInt < chosen.toInt <;>
    by_cases a : s.alpha.toInt < chosen.toInt <;> by_cases lo : chosen.toInt < i.beta.toInt <;>
    simp [lockedState, SplitWindow.publish, SplitWindow.raisesAlpha, *, Int.max_def] <;> omega

end NeuralReversi.PvsParallel.SplitWorkerMir

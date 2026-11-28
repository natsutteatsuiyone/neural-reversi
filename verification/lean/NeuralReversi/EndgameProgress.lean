module
public import NeuralReversi.SearchPublication

@[expose] public section

namespace NeuralReversi.EndgameProgress

open SearchPublication (Attempt Stamp Completed Row Snapshot capture finish nextMarks step run)

def ladder : List Nat := [1, 2, 3, 4, 5]

def fixedStepAllowed (cap : Nat) (depthFor : Nat → Nat) (empties selectivity : Nat) : Bool :=
  decide (selectivity ≤ cap ∧ empties ≤ depthFor selectivity)

def endSelectivity (cap : Nat) (depthFor : Nat → Nat) (empties : Nat) : Option Nat :=
  (ladder.takeWhile (fixedStepAllowed cap depthFor empties)).getLast?

def capAllows (cap selectivity : Nat) : Bool :=
  decide (selectivity ≤ cap)

def reached (allowed keep : Nat → Bool) : List Nat → List Nat
  | [] => []
  | level :: rest =>
    if allowed level then level :: (if keep level then reached allowed keep rest else [])
    else []

private theorem takeWhile_last_cutoff (allowed : Nat → Bool) (levels : List Nat)
    (ordered : levels.Pairwise (· < ·)) :
    (levels.takeWhile allowed) =
      levels.takeWhile (fun level => (levels.takeWhile allowed).getLast?.any
        (fun cap => capAllows cap level)) := by
  induction levels with
  | nil => rfl
  | cons level rest ih =>
    have sorted := List.pairwise_cons.mp ordered
    by_cases admitLevel : allowed level = true
    · rw [List.takeWhile_cons_of_pos admitLevel]
      cases tail : (rest.takeWhile allowed).getLast? with
      | none =>
        have empty : rest.takeWhile allowed = [] := List.getLast?_eq_none_iff.mp tail
        rw [empty]
        have admitHead : capAllows level level = true := by simp [capAllows]
        simp only [List.getLast?_singleton, Option.any_some, List.takeWhile_cons_of_pos admitHead]
        cases rest with
        | nil => rfl
        | cons next after =>
          have nextGreater : level < next := sorted.1 next (by simp)
          simp [capAllows, Nat.not_le_of_gt nextGreater]
      | some last =>
        have lastInPrefix : last ∈ rest.takeWhile allowed :=
          List.mem_of_mem_getLast? (by simp [tail])
        have lastInRest := List.takeWhile_subset allowed lastInPrefix
        have admitHead : capAllows last level = true := by
          simp [capAllows, Nat.le_of_lt (sorted.1 last lastInRest)]
        have tailEq := ih sorted.2
        simp only [tail, Option.any_some] at tailEq
        rw [List.getLast?_cons, tail]
        simp only [Option.getD_some, Option.any_some, List.takeWhile_cons_of_pos admitHead]
        rw [tailEq]
    · simp [admitLevel]

private theorem reached_takeWhile (allowed keep : Nat → Bool) (levels : List Nat) :
    reached allowed keep levels = reached (fun _ => true) keep (levels.takeWhile allowed) := by
  induction levels with
  | nil => rfl
  | cons level rest ih =>
    by_cases admitted : allowed level = true <;> simp [reached, admitted, ih]

def FixedLadderContract : Prop :=
  ∀ (cap : Nat) (depthFor : Nat → Nat) (empties : Nat) (keep : Nat → Bool),
    reached (fixedStepAllowed cap depthFor empties) keep ladder =
      reached (fun level => (endSelectivity cap depthFor empties).any
        (fun last => capAllows last level)) keep ladder

theorem fixedLadderContract_proven : FixedLadderContract := by
  intro cap depthFor empties keep
  rw [reached_takeWhile (fixedStepAllowed cap depthFor empties),
    reached_takeWhile (fun level =>
      (endSelectivity cap depthFor empties).any (fun last => capAllows last level))]
  exact congrArg (reached (fun _ => true) keep)
    (takeWhile_last_cutoff (fixedStepAllowed cap depthFor empties) ladder (by decide))

theorem endSelectivity_eq_none_iff (cap : Nat) (depthFor : Nat → Nat) (empties : Nat) :
    endSelectivity cap depthFor empties = none ↔ cap < 1 ∨ depthFor 1 < empties := by
  simp [endSelectivity, ladder, List.takeWhile_cons, fixedStepAllowed]
  omega

theorem endSelectivity_mem_ladder {cap : Nat} {depthFor : Nat → Nat} {empties last : Nat}
    (chosen : endSelectivity cap depthFor empties = some last) : last ∈ ladder := by
  apply List.takeWhile_subset (fixedStepAllowed cap depthFor empties)
  exact List.mem_of_mem_getLast? (by simpa [endSelectivity] using chosen)

abbrev Clock := Nat → Nat
abbrev Continue := Nat → Nat → Bool

def StableBudget (keep : Continue) : Prop :=
  ∀ level earlier later, earlier ≤ later → keep level later = true → keep level earlier = true

def ClockOrder (earlier later : Clock) : Prop :=
  ∀ level, earlier level ≤ later level

private theorem reached_in_ladder (allowed keep : Nat → Bool) (levels : List Nat) :
    (reached allowed keep levels).IsPrefix levels := by
  induction levels with
  | nil => exact ⟨[], rfl⟩
  | cons level rest ih =>
    rw [reached]
    by_cases entered : allowed level = true
    · rw [ite_eq_left entered]
      cases keep level
      · exact ⟨rest, rfl⟩
      · obtain ⟨extra, append⟩ := ih
        exact ⟨extra, by simp [append]⟩
    · rw [ite_eq_right entered]
      exact ⟨level :: rest, rfl⟩

private theorem reached_mono (allowed older newer : Nat → Bool) (levels : List Nat)
    (permission : ∀ level, newer level = true → older level = true) :
    (reached allowed newer levels).IsPrefix (reached allowed older levels) := by
  induction levels with
  | nil => exact ⟨[], rfl⟩
  | cons level rest ih =>
    simp only [reached]
    by_cases entered : allowed level = true
    · simp only [entered, ite_true]
      cases more : newer level
      · exact ⟨if older level then reached allowed older rest else [], by simp⟩
      · obtain ⟨extra, append⟩ := ih
        exact ⟨extra, by simp [permission level more, append]⟩
    · rw [ite_eq_right entered, ite_eq_right entered]
      exact ⟨[], rfl⟩

def lineLevels (allowed : Nat → Bool) (keep : Continue) (time : Clock) : List Nat :=
  reached allowed (fun level => keep level (time level)) ladder

def lastLevel (allowed : Nat → Bool) (keep : Continue) (time : Clock) : Option Nat :=
  (lineLevels allowed keep time).getLast?

private theorem line_nonempty (allowed : Nat → Bool) (keep : Continue) (time : Clock)
    (start : allowed 1 = true) : lineLevels allowed keep time ≠ [] := by
  simp [lineLevels, ladder, reached, start]

private theorem last_mono {later earlier : List Nat} (ordered : earlier.Pairwise (· ≤ ·))
    (contained : later.IsPrefix earlier) (latest : later.getLast? = some recent)
    (older : earlier.getLast? = some past) : recent ≤ past := by
  obtain ⟨extra, extend⟩ := contained
  rw [← extend, List.getLast?_append, latest] at older
  cases tail : extra.getLast? with
  | none => exact Nat.le_of_eq (Option.some.inj (by simpa only [tail, Option.or] using older))
  | some last =>
    have same : last = past := Option.some.inj (by simpa only [tail, Option.or] using older)
    have recentMem : recent ∈ later := List.mem_of_mem_getLast? (by simp [latest])
    have extraMem : last ∈ extra := List.mem_of_mem_getLast? (by simp [tail])
    rw [← extend] at ordered
    exact same ▸ ordered.rel_of_mem_append recentMem extraMem

def PrefixContract : Prop :=
  ∀ (allowed : Nat → Bool) (keep : Continue) (earlier later : Clock),
    StableBudget keep → ClockOrder earlier later →
    (lineLevels allowed keep later).IsPrefix (lineLevels allowed keep earlier) ∧
    (∀ old new, lastLevel allowed keep earlier = some old →
      lastLevel allowed keep later = some new → new ≤ old)

theorem prefixContract_proven : PrefixContract := by
  intro allowed keep earlier later budget times
  have prefixOf : (lineLevels allowed keep later).IsPrefix (lineLevels allowed keep earlier) :=
    reached_mono _ _ _ _ (fun level => budget level _ _ (times level))
  have sorted : ladder.Pairwise (· ≤ ·) := by decide
  have ordered : (lineLevels allowed keep earlier).Pairwise (· ≤ ·) :=
    List.Pairwise.sublist (reached_in_ladder _ _ _).sublist sorted
  exact ⟨prefixOf, fun _ _ old new => last_mono ordered prefixOf new old⟩

def lastStamp (minimum depth : Nat) (levels : List Nat) : Option Stamp :=
  match levels.getLast? with
  | none => none
  | some level => if minimum ≤ level then some ⟨depth, level⟩ else none

def lineStamp (minimum depth : Nat) (allowed : Nat → Bool) (keep : Continue) (time : Clock) : Option Stamp :=
  lastStamp minimum depth (lineLevels allowed keep time)

def AtLeast (depth selectivity : Nat) (marks : List (Option Stamp)) : Prop :=
  ∀ mark ∈ marks, ∃ stamp, mark = some stamp ∧ depth ≤ stamp.depth ∧ selectivity ≤ stamp.selectivity

def FinalContract : Prop :=
  ∀ (minimum depth : Nat) (allowed : Nat → Bool) (keep : Continue) (previous : List Clock)
    (last : Clock) (rows : List Row) (state : Completed),
    allowed 1 = true → StableBudget keep →
    (∀ earlier ∈ previous, ClockOrder earlier last) →
    state.marks = (previous ++ [last]).map (lineStamp minimum depth allowed keep) →
    ∃ final, lastLevel allowed keep last = some final ∧
      (minimum ≤ final →
        AtLeast depth final state.marks ∧
        finish minimum state (.normal ⟨depth, final⟩ rows) = some (capture rows ⟨depth, final⟩ true)) ∧
      (final < minimum → finish minimum state (.normal ⟨depth, final⟩ rows) = state.last)

theorem finalContract_proven : FinalContract := by
  intro minimum depth allowed keep previous last rows state first budget later recorded
  have terminal (time : Clock) : ∃ level, lastLevel allowed keep time = some level := by
    cases found : lastLevel allowed keep time with
    | none => exact False.elim (line_nonempty allowed keep time first (List.getLast?_eq_none_iff.mp found))
    | some level => exact ⟨level, rfl⟩
  obtain ⟨final, ended⟩ := terminal last
  refine ⟨final, ended, fun threshold => ⟨?_, by simp [finish, threshold]⟩,
    fun below => by simp [finish, Nat.not_le_of_gt below]⟩
  intro mark found
  rw [recorded] at found
  obtain ⟨time, included, rfl⟩ := List.mem_map.mp found
  obtain ⟨level, completed⟩ := terminal time
  have timeOrder : ClockOrder time last := by
    rcases List.mem_append.mp included with earlier | self
    · exact later time earlier
    · have same := List.mem_singleton.mp self
      subst time
      exact fun _ => Nat.le_refl _
  have smallest := (prefixContract_proven allowed keep time last budget timeOrder).2 level final completed ended
  have eligible := Nat.le_trans threshold smallest
  change (lineLevels allowed keep time).getLast? = some level at completed
  exact ⟨⟨depth, level⟩, by simp [lineStamp, lastStamp, completed, eligible], Nat.le_refl _, smallest⟩

def attemptAt (index depth : Nat) (rows : Nat → List Row) (level : Nat) : Attempt :=
  ⟨index, ⟨depth, level⟩, true, rows level, false⟩

private theorem step_mark (minimum depth level : Nat) (rows : Nat → List Row) (state : Completed)
    (before after : List (Option Stamp)) (old : Option Stamp)
    (mapped : state.marks = before ++ old :: after) :
    (step minimum state (attemptAt before.length depth rows level)).marks =
      before ++ (if minimum ≤ level then some ⟨depth, level⟩ else old) :: after := by
  by_cases eligible : minimum ≤ level
  · simp [step, nextMarks, SearchPublication.reportable, attemptAt, eligible, mapped, List.set_append_right]
  · simp [step, nextMarks, SearchPublication.reportable, attemptAt, eligible, mapped]

private theorem update_last (minimum depth level : Nat) (rest : List Nat) (old : Option Stamp)
    (ordered : (level :: rest).Pairwise (· ≤ ·)) :
    (lastStamp minimum depth rest).or (if minimum ≤ level then some ⟨depth, level⟩ else old) =
      (lastStamp minimum depth (level :: rest)).or old := by
  cases rest with
  | nil =>
    by_cases eligible : minimum ≤ level <;> simp [lastStamp, eligible, Option.or]
  | cons next rest =>
    cases lookup : (next :: rest).getLast? with
    | none =>
      have empty := List.getLast?_eq_none_iff.mp lookup
      cases empty
    | some terminal =>
      by_cases eligible : minimum ≤ terminal
      · simp [lastStamp, lookup, eligible, Option.or]
      · have seen : terminal ∈ next :: rest := List.mem_of_mem_getLast? (by simp [lookup])
        have bound := (List.pairwise_cons.mp ordered).1 terminal seen
        have skipped : ¬ minimum ≤ level := fun enough => eligible (Nat.le_trans enough bound)
        simp [lastStamp, lookup, eligible, skipped, Option.or]

private theorem run_line_marks (minimum depth : Nat) (rows : Nat → List Row)
    (levels : List Nat) (ordered : levels.Pairwise (· ≤ ·)) (state : Completed)
    (before after : List (Option Stamp)) (old : Option Stamp)
    (mapped : state.marks = before ++ old :: after) :
    (run minimum state (levels.map (attemptAt before.length depth rows))).marks =
      before ++ (lastStamp minimum depth levels).or old :: after := by
  induction levels generalizing state old with
  | nil => exact mapped
  | cons level rest ih =>
    simp only [List.map_cons, run, List.foldl_cons]
    change (run minimum (step minimum state (attemptAt before.length depth rows level))
      (rest.map (attemptAt before.length depth rows))).marks = _
    rw [ih ordered.tail _ _ (step_mark minimum depth level rows state before after old mapped),
      update_last minimum depth level rest old ordered]

def runNormal (minimum depth : Nat) (allowed : Nat → Bool) (keep : Continue)
    (rows : Nat → Nat → List Row) : List Clock → Nat → Completed → Completed
  | [], _, state => state
  | time :: rest, index, state =>
    let attempts := (lineLevels allowed keep time).map (attemptAt index depth (rows index))
    runNormal minimum depth allowed keep rows rest (index + 1) (run minimum state attempts)

private theorem runNormal_marks (minimum depth : Nat) (allowed : Nat → Bool) (keep : Continue)
    (rows : Nat → Nat → List Row) (times : List Clock) (before : List (Option Stamp)) (state : Completed)
    (mapped : state.marks = before ++ List.replicate times.length none) :
    (runNormal minimum depth allowed keep rows times before.length state).marks =
      before ++ times.map (lineStamp minimum depth allowed keep) := by
  induction times generalizing before state with
  | nil => simpa [runNormal] using mapped
  | cons time rest ih =>
    have ordered : (lineLevels allowed keep time).Pairwise (· ≤ ·) :=
      List.Pairwise.sublist (reached_in_ladder _ _ _).sublist (by decide)
    let index := before.length
    let next := run minimum state ((lineLevels allowed keep time).map (attemptAt index depth (rows index)))
    have shape : state.marks = before ++ none :: List.replicate rest.length none := by
      simpa only [List.length_cons, List.replicate_succ] using mapped
    have marks : next.marks =
        (before ++ [lineStamp minimum depth allowed keep time]) ++ List.replicate rest.length none := by
      have checked := run_line_marks minimum depth (rows index) _ ordered state before _ none shape
      simpa [lineStamp, List.append_assoc, Option.or_none] using checked
    have result := ih (before ++ [lineStamp minimum depth allowed keep time]) next marks
    simpa [runNormal, next, index, List.append_assoc] using result

def NormalRunContract : Prop :=
  ∀ (minimum depth cap : Nat) (keep : Continue) (previous : List Clock)
    (last : Clock) (rows : Nat → Nat → List Row) (finalRows : List Row) (inherited : Completed),
    1 ≤ cap → StableBudget keep →
    (∀ earlier ∈ previous, ClockOrder earlier last) →
    let allowed := capAllows cap
    let times := previous ++ [last]
    inherited.marks.length = times.length →
    let state := runNormal minimum depth allowed keep rows times 0 (SearchPublication.beginPhase inherited)
    ∃ final, lastLevel allowed keep last = some final ∧
      (minimum ≤ final →
        AtLeast depth final state.marks ∧
        finish minimum state (.normal ⟨depth, final⟩ finalRows) = some (capture finalRows ⟨depth, final⟩ true)) ∧
      (final < minimum → finish minimum state (.normal ⟨depth, final⟩ finalRows) = state.last)

theorem normalRunContract_proven : NormalRunContract := by
  intro minimum depth cap keep previous last rows finalRows inherited entry budget later allowed times count state
  have first : allowed 1 = true := by simp [allowed, capAllows, entry]
  have marks : state.marks = times.map (lineStamp minimum depth allowed keep) := by
    change (runNormal minimum depth allowed keep rows times 0 (SearchPublication.beginPhase inherited)).marks = _
    simpa only [List.nil_append, List.length_nil] using
      runNormal_marks minimum depth allowed keep rows times [] (SearchPublication.beginPhase inherited)
        (by simp [SearchPublication.beginPhase, SearchPublication.start, count])
  exact finalContract_proven minimum depth allowed keep previous last finalRows state first budget later marks

example : lastLevel (fun _ => true) (fun _ time => decide (time < 10)) (fun level => 2 * level) = some 5 ∧
    lastLevel (fun _ => true) (fun _ time => decide (time < 10)) (fun level => 5 + 2 * level) = some 3 := by
  decide

end NeuralReversi.EndgameProgress

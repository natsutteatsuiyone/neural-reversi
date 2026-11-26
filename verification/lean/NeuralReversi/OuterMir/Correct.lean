import NeuralReversi.OuterMir.Program
import NeuralReversi.EndgameTiming

namespace NeuralReversi.OuterMir

def lineLevels (last minimum : Nat) (stop : Nat → Bool) : List Nat → List Nat
  | [] => []
  | level :: rest =>
    match step (some level) last minimum (stop level) with
    | .endLine => []
    | .accepted searched _ keep =>
      searched :: (if keep then lineLevels last minimum stop rest else [])

theorem lineLevels_subset (levels : List Nat) (last minimum : Nat) (stop : Nat → Bool) :
    ∀ n ∈ lineLevels last minimum stop levels, n ∈ levels := by
  induction levels with
  | nil => simp [lineLevels]
  | cons head rest ih =>
    by_cases cap : last < head <;> cases h : stop head <;> simp [lineLevels, step, cap, h] <;> grind

theorem earlierLine_exact (levels : List Nat) (last minimum : Nat) (early late : Nat → Bool)
    (monotone : ∀ step, early step = true → late step = true) :
    ∀ n ∈ lineLevels last minimum late levels, n ∈ lineLevels last minimum early levels := by
  induction levels with
  | nil => simp [lineLevels]
  | cons level rest ih =>
    by_cases cap : last < level
    · simp [lineLevels, step, cap]
    · cases a : early level <;> cases b : late level <;>
        simp [lineLevels, step, cap, a, b] <;> grind

theorem noClock (minimum : Nat) :
    lineLevels 5 minimum (fun _ => false) ladder = [1, 2, 3, 4, 5] := by
  rfl

theorem firstLevel (last minimum : Nat) (enablesFirst : 1 ≤ last) (stop : Nat → Bool) :
    ∃ rest, lineLevels last minimum stop ladder = 1 :: rest := by
  simp [ladder, lineLevels, step, Nat.not_lt_of_ge enablesFirst]

theorem allPublished (lines : List (Option (Nat × Nat))) (idx depth level : Nat)
    (inside : idx < lines.length) :
    (lineStamp lines idx depth level).1[idx]? = some (some (depth, level)) ∧
    ((lineStamp lines idx depth level).2 = true ↔
      ∀ s ∈ (lineStamp lines idx depth level).1, s = some (depth, level)) := by
  simp [lineStamp, List.getElem?_set_self inside]

theorem allLines_exact (stops : Nat → Nat → Bool) (lastLine last minimum : Nat)
    (monotone : ∀ i, i ≤ lastLine → ∀ level, stops i level = true → stops lastLine level = true)
    (final : 5 ∈ lineLevels last minimum (stops lastLine) ladder) :
    ∀ i, i ≤ lastLine → 5 ∈ lineLevels last minimum (stops i) ladder := by
  intro i earlier
  exact earlierLine_exact ladder last minimum (stops i) (stops lastLine) (monotone i earlier) 5 final

theorem allLines_of_returned_level (priorLevel last minimum lastLine : Nat) (enablesFirst : 1 ≤ last)
    (stops : Nat → Nat → Bool)
    (monotone : ∀ i, i ≤ lastLine → ∀ level, stops i level = true → stops lastLine level = true)
    (current : (lineLevels last minimum (stops lastLine) ladder).getLastD priorLevel = 5) :
    ∀ i, i ≤ lastLine → 5 ∈ lineLevels last minimum (stops i) ladder := by
  apply allLines_exact stops lastLine last minimum monotone
  obtain ⟨rest, searched⟩ := firstLevel last minimum enablesFirst (stops lastLine)
  rw [searched] at current ⊢
  rw [← current]
  exact List.getLast_mem (l := 1 :: rest) (by simp)

def sampledStop (prediction : Option EndgameTiming.Prediction) (nanos : Nat → EndgameProgress.Clock)
    (line level : Nat) : Bool :=
  !(EndgameTiming.softAllows prediction (level - 1) (EndgameTiming.sampledMs (nanos line (level - 1))))

theorem allLines_of_clock (priorLevel last minimum lastLine : Nat) (enablesFirst : 1 ≤ last)
    (prediction : Option EndgameTiming.Prediction) (nanos : Nat → EndgameProgress.Clock)
    (ordered : ∀ i, i ≤ lastLine → EndgameProgress.ClockOrder (nanos i) (nanos lastLine))
    (fits : ∀ step, nanos lastLine step / 1000000 < 2 ^ 64)
    (current : (lineLevels last minimum (sampledStop prediction nanos lastLine) ladder).getLastD priorLevel = 5) :
    ∀ i, i ≤ lastLine → 5 ∈ lineLevels last minimum (sampledStop prediction nanos i) ladder := by
  apply allLines_of_returned_level priorLevel last minimum lastLine enablesFirst _ _ current
  intro i earlier level stopped
  simp only [sampledStop, Bool.not_eq_true'] at stopped ⊢
  have orderedMs := EndgameTiming.sampleContract_proven _ _ (ordered i earlier) fits (level - 1)
  have monotone := EndgameTiming.timingContract_proven.1 prediction (level - 1) _ _ orderedMs
  cases answer : EndgameTiming.softAllows prediction (level - 1)
      (EndgameTiming.sampledMs (nanos lastLine (level - 1))) with
  | false => rfl
  | true => exact Bool.noConfusion (stopped.symm.trans (monotone answer))

theorem finish_exact (minimum : Nat) (reportable : minimum ≤ 5) (sort : PublicationMir.SliceSort)
    (rows : List RootMoveStorage.Row) (depth nodes : Nat) (prior : Option PublicationMir.Iteration)
    (fallback : Unit → PublicationMir.SearchResult) {report : SearchFallback.Report}
    (current : SearchFallback.fromSnapshot (SearchPublication.capture (rows.map PublicationMir.project)
      ⟨depth, 5⟩ true) nodes = some report) :
    ∃ result, finish minimum 5 sort rows depth nodes prior fallback = some result ∧
      PublicationMir.score result = some report.best.score := by
  simp only [finish, ite_eq_left reportable]
  exact PublicationMir.completedScore_of_report sort rows depth 5 true nodes fallback current

theorem capture_stamp (minimum level depth nodes : Nat) (eligible : minimum ≤ level)
    (sort : PublicationMir.SliceSort) (rows : List RootMoveStorage.Row) (prior : Option PublicationMir.Iteration)
    (fallback : Unit → PublicationMir.SearchResult) (report : SearchFallback.Report)
    (returned : finish minimum level sort rows depth nodes prior fallback = some (.bestMove report)) :
    report.stamp = ⟨depth, level⟩ ∧ report.endgame = true := by
  simp only [finish, ite_eq_left eligible, PublicationMir.completed, PublicationMir.record,
    PublicationMir.first, PublicationMir.capture, PublicationMir.intoResult,
    PublicationMir.build, PublicationMir.snapshot] at returned
  cases ordered : sort.byReverseKey PublicationMir.key rows with
  | nil => simp [ordered] at returned
  | cons head rest =>
    simp [ordered] at returned
    subst report
    exact ⟨rfl, rfl⟩

theorem direct_has_current (priorLevel last : Nat) (enablesFirst : 1 ≤ last) (stop : Nat → Bool) :
    1 ≤ (lineLevels last 1 stop ladder).getLastD priorLevel := by
  obtain ⟨rest, searched⟩ := firstLevel last 1 enablesFirst stop
  have member : (lineLevels last 1 stop ladder).getLastD priorLevel ∈ lineLevels last 1 stop ladder := by
    rw [searched]
    exact List.getLast_mem (l := 1 :: rest) (by simp)
  have inLadder := lineLevels_subset ladder last 1 stop _ member
  have positive (n : Nat) (member : n ∈ ladder) : 1 ≤ n := by
    simp [ladder] at member
    omega
  exact positive _ inLadder

theorem direct_stamp (priorLevel last depth nodes : Nat) (enablesFirst : 1 ≤ last)
    (stop : Nat → Bool) (sort : PublicationMir.SliceSort) (rows : List RootMoveStorage.Row)
    (prior : Option PublicationMir.Iteration) (fallback : Unit → PublicationMir.SearchResult)
    (report : SearchFallback.Report)
    (returned : finish 1 ((lineLevels last 1 stop ladder).getLastD priorLevel) sort rows depth nodes
      prior fallback = some (.bestMove report)) :
    report.stamp = ⟨depth, (lineLevels last 1 stop ladder).getLastD priorLevel⟩ ∧ report.endgame = true :=
  capture_stamp _ _ _ _ (direct_has_current priorLevel last enablesFirst stop) _ _ _ _ _ returned

theorem final_unreportable (minimum last : Nat) (stop : Nat → Bool)
    (prior : Nat) (below : (lineLevels last minimum stop ladder).getLastD prior < minimum) :
    ∀ level ∈ lineLevels last minimum stop ladder, level < minimum := by
  have auxiliary (remaining : List Nat) (ordered : remaining.Pairwise (· ≤ ·))
      (h : (lineLevels last minimum stop remaining).getLastD prior < minimum) :
      ∀ level ∈ lineLevels last minimum stop remaining, level < minimum := by
    induction remaining with
    | nil => simp [lineLevels]
    | cons head tail ih =>
      by_cases cap : last < head
      · simp [lineLevels, step, cap]
      · cases clock : stop head with
        | true =>
          simpa [lineLevels, step, cap, clock] using h
        | false =>
          let after := lineLevels last minimum stop tail
          have stepEq : lineLevels last minimum stop (head :: tail) = head :: after := by
            simp [lineLevels, step, cap, clock, after]
          rw [stepEq] at h ⊢
          cases shape : after with
          | nil => simpa [shape] using h
          | cons n rest =>
            have neq : after ≠ [] := by simp [shape]
            have tailBound : after.getLastD prior < minimum := by simpa [shape] using h
            intro level found
            rw [List.mem_cons] at found
            rcases found with rfl | later
            · have member : after.getLastD prior ∈ after := by
                rw [shape]
                exact List.getLast_mem (l := n :: rest) (by simp)
              exact Nat.lt_of_le_of_lt ((List.pairwise_cons.mp ordered).1 _
                (lineLevels_subset tail last minimum stop _ member)) tailBound
            · apply ih (List.pairwise_cons.mp ordered).2 tailBound level
              simpa [after, shape] using later
  exact auxiliary ladder (by decide) below

theorem unpublished_last_line (minimum lastLine : Nat) (state : SearchPublication.Completed)
    (attempts : List SearchPublication.Attempt) (pending : state.marks[lastLine]? = some none)
    (unmarked : ∀ a ∈ attempts, SearchPublication.reportable minimum a = true → a.index ≠ lastLine) :
    (SearchPublication.run minimum state attempts).marks[lastLine]? = some none ∧
    (SearchPublication.run minimum state attempts).last = state.last := by
  induction attempts generalizing state with
  | nil => exact ⟨pending, rfl⟩
  | cons a rest ih =>
    let next := SearchPublication.step minimum state a
    have unset : (SearchPublication.nextMarks minimum state.marks a)[lastLine]? = some none := by
      cases eligible : SearchPublication.reportable minimum a with
      | false => simpa [SearchPublication.nextMarks, eligible] using pending
      | true =>
        have different := unmarked a (by simp) eligible
        simpa [SearchPublication.nextMarks, eligible, List.getElem?_set_ne different] using pending
    have noCapture : SearchPublication.ready minimum state.marks a = false := by
      cases ready : SearchPublication.ready minimum state.marks a with
      | false => rfl
      | true =>
        have all := (Bool.and_eq_true_iff.mp ready).2
        have bad := List.all_eq_true.mp all none (List.mem_of_getElem? unset)
        simp at bad
    have saved : next.last = state.last := by simp [next, SearchPublication.step, SearchPublication.published, noCapture]
    obtain ⟨still, same⟩ := ih next unset (fun x hx => unmarked x (by simp [hx]))
    exact ⟨still, same.trans saved⟩

end NeuralReversi.OuterMir

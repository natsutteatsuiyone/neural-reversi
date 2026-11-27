import NeuralReversi.AspirationMir.SerialStack

namespace NeuralReversi.PublicationMir.InterruptedExact

open PvsParallel
open PvsBoardTree (Position minimax)
open AspirationMir.SerialStack (State Certified)
open AspirationMir.SmallRoot (Answer)
open SearchPublication (Snapshot Attempt Completed)

variable {mc : Bitboard.Moves.X86Config} {fc : Cutoffs.FlipConfig} {b : Position}

/-- Extract the numerical row invariant from a completed exact root trial. The
    trial certificate records the physical root writes and their search bounds;
    neither the maximum row nor its score is a premise of this theorem. -/
theorem certified_rows {state : State} {answer : Answer} (cert : Certified mc fc b state answer)
    (covers : Root.Covers mc b (state.rows.map RootMoveStorage.Row.square))
    (within : 0 < state.rows.length)
    (inside : state.regs.alpha.toInt < answer.value ∧ answer.value < state.regs.beta.toInt) :
    ExactScore.RowScores (minimax mc fc b) answer.rows := by
  obtain ⟨latest, _, writes, attempt⟩ := cert
  obtain ⟨names, _, stored⟩ := writes.scores (Root.roots_nodup covers)
  simp only [List.drop_zero] at stored
  obtain ⟨head, tail, ordered, selected⟩ := attempt.2 inside.1 inside.2
  obtain ⟨perm, _, _, descending, _⟩ := RootOrder.sortContract_proven
    (state.rows.map RootMoveStorage.Row.square) 0 latest.scores (Nat.zero_le _)
  simp only [List.drop_zero] at ordered descending
  rw [ordered] at descending perm
  have exactHead := selected.2.1.trans (Root.value_zero covers (by simpa using within))
  obtain ⟨winner, present, eq⟩ := List.mem_map.mp
    (show head ∈ answer.rows.map RootMoveStorage.Row.square from
      names ▸ perm.mem_iff.mp (by simp))
  refine ⟨?_, winner, present, (stored winner present).trans (eq ▸ exactHead)⟩
  intro row member
  rw [stored row member, ← exactHead]
  have sq := perm.mem_iff.mpr (names ▸ List.mem_map_of_mem
    (f := RootMoveStorage.Row.square) member)
  rcases List.mem_cons.mp sq with same | later
  · rw [same]
    exact Int.le_refl _
  · exact (List.pairwise_cons.mp descending).1 row.square later

/-- The same numerical invariant on the snapshot's score-bearing projection. -/
def RowsExact (truth : Int) (rows : List SearchPublication.Row) : Prop :=
  (∀ row ∈ rows, row.score ≤ truth) ∧ ∃ row ∈ rows, row.score = truth

theorem rowsExact_project {truth : Int} {rows : List RootMoveStorage.Row} :
    RowsExact truth (rows.map project) ↔ ExactScore.RowScores truth rows := by
  constructor
  · rintro ⟨upper, winner, member, same⟩
    obtain ⟨row, present, rfl⟩ := List.mem_map.mp member
    exact ⟨fun r hr => upper (project r) (List.mem_map_of_mem hr), row, present, same⟩
  · rintro ⟨upper, winner, member, same⟩
    refine ⟨?_, project winner, List.mem_map_of_mem member, same⟩
    intro r hr
    obtain ⟨row, present, rfl⟩ := List.mem_map.mp hr
    exact upper row present

/-- A completed exact attempt is justified by a root trial and its accepted
    aspiration window. Aborted and non-exact attempts have no score obligation. -/
def ExactAttempt (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (attempt : Attempt) : Prop :=
  attempt.aborted = false → attempt.endgame = true → attempt.stamp.selectivity = 5 →
    ∃ (state : State) (answer : Answer),
      attempt.rows = answer.rows.map project ∧ Certified mc fc b state answer ∧
      Root.Covers mc b (state.rows.map RootMoveStorage.Row.square) ∧
      0 < state.rows.length ∧
      state.regs.alpha.toInt < answer.value ∧ answer.value < state.regs.beta.toInt

variable {attempt : Attempt} {attempts : List Attempt} {state : Completed} {minimum : Nat}

/-- Construct an attempt's semantic origin from the existing executable exact
    root driver. The driver itself derives the write/search certificate. -/
theorem exactAttempt_of_trial (c : SerialScout.Config)
    (inputs : Nat → SerialExecution.Inputs mc fc)
    (roots : Nat → AspirationMir.Word → AspirationMir.Word → SerialRecursion.RootInput)
    (trialState : State) (closing : Nat) (answer : Answer)
    (run : AspirationMir.SerialStack.trial c b inputs roots trialState = some (closing, answer))
    (valid : SerialExecution.ReadFacts (inputs trialState.index))
    (hints : SerialTermination.Hints (inputs trialState.index))
    (covers : Root.Covers mc b (trialState.rows.map RootMoveStorage.Row.square))
    (within : 0 < trialState.rows.length)
    (window : Aspiration.Window trialState.regs.alpha.toInt trialState.regs.beta.toInt)
    (inside : trialState.regs.alpha.toInt < answer.value ∧ answer.value < trialState.regs.beta.toInt)
    (rows : attempt.rows = answer.rows.map project) : ExactAttempt mc fc b attempt := by
  obtain ⟨other, actual, ran, cert, _⟩ :=
    AspirationMir.SerialStack.trial_returns c b inputs roots trialState valid hints covers within window
  rw [run] at ran
  cases Option.some.inj ran
  intro _ _ _
  exact ⟨trialState, answer, rows, cert, covers, within, inside⟩

theorem exactAttempt_rows (source : ExactAttempt mc fc b attempt)
    (complete : attempt.aborted = false) (ending : attempt.endgame = true)
    (level : attempt.stamp.selectivity = 5) : RowsExact (minimax mc fc b) attempt.rows := by
  obtain ⟨state, answer, rows, cert, cover, positive, inside⟩ := source complete ending level
  rw [rows, rowsExact_project]
  exact certified_rows cert cover positive inside

/-- Exact snapshots contain a true maximum. Their order is tracked as well so
    `SearchFallback.fromSnapshot` may use the first row without sorting again. -/
def SnapshotSafe (truth : Int) (saved : Snapshot) : Prop :=
  saved.endgame = true → saved.stamp.selectivity = 5 →
    RowsExact truth saved.rows ∧ saved.rows.Pairwise (fun a b => b.score ≤ a.score)

def SavedSafe (truth : Int) (last : Option Snapshot) : Prop :=
  ∀ saved, last = some saved → SnapshotSafe truth saved

/-- The actual snapshot return reads its first row. Its exact score follows from
    the retained row invariant and order, without re-running any search or sort. -/
theorem fromSnapshot_exact {truth : Int} {saved : Snapshot} {nodes : Nat}
    {report : SearchFallback.Report} (safe : SnapshotSafe truth saved)
    (run : SearchFallback.fromSnapshot saved nodes = some report)
    (ending : report.endgame = true) (level : report.stamp.selectivity = 5) :
    report.best.score = truth := by
  cases rows : saved.rows with
  | nil => simp [SearchFallback.fromSnapshot, rows] at run
  | cons head tail =>
    simp only [SearchFallback.fromSnapshot, rows, List.head?_cons, Option.map_some,
      Option.some.injEq] at run
    subst report
    obtain ⟨⟨upper, winner, present, same⟩, ordered⟩ := safe ending level
    rw [rows] at upper present ordered
    have bound := upper head (by simp)
    have attained : truth ≤ head.score := by
      rw [← same]
      rcases List.mem_cons.mp present with rfl | later
      · exact Int.le_refl _
      · exact (List.pairwise_cons.mp ordered).1 winner later
    exact Int.le_antisymm bound attained

theorem capture_safe (source : ExactAttempt mc fc b attempt)
    (complete : attempt.aborted = false) :
    SnapshotSafe (minimax mc fc b)
      (SearchPublication.capture attempt.rows attempt.stamp attempt.endgame) := by
  intro ending level
  obtain ⟨upper, winner, member, same⟩ := exactAttempt_rows source complete ending level
  have capture := SearchPublication.captureContract_proven attempt.rows attempt.stamp attempt.endgame
  have perm := capture.2.2.1
  exact ⟨⟨fun row hr => upper row (perm.mem_iff.mp hr),
    winner, perm.mem_iff.mpr member, same⟩, capture.2.2.2.1⟩

/-- An inherited fallback either is unmarked or comes from a previously completed
    exact root trial of this board. No numerical assumption is made about an
    unmarked midgame or selective fallback. -/
inductive Seed (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : Position) :
    Option Snapshot → Prop where
  | none : Seed mc fc b none
  | unmarked (saved : Snapshot) (tag : saved.endgame = false ∨ saved.stamp.selectivity ≠ 5) :
      Seed mc fc b (some saved)
  | completed (attempt : Attempt) (source : ExactAttempt mc fc b attempt)
      (complete : attempt.aborted = false) :
      Seed mc fc b (some (SearchPublication.capture attempt.rows attempt.stamp attempt.endgame))

theorem Seed.safe (source : Seed mc fc b last) : SavedSafe (minimax mc fc b) last := by
  intro saved found
  cases source with
  | none => contradiction
  | unmarked old tag =>
    cases found
    intro ending level
    rcases tag with no | weak
    · cases no.symm.trans ending
    · exact (weak level).elim
  | completed attempt source complete =>
    cases found
    exact capture_safe source complete

theorem step_safe (prior : SavedSafe (minimax mc fc b) state.last)
    (source : ExactAttempt mc fc b attempt) :
    SavedSafe (minimax mc fc b) (SearchPublication.step minimum state attempt).last := by
  cases ready : SearchPublication.ready minimum state.marks attempt with
  | false => simpa [SearchPublication.step, SearchPublication.published, ready] using prior
  | true =>
    have reportable : SearchPublication.reportable minimum attempt = true :=
      (Bool.and_eq_true_iff.mp ready).1
    have complete : attempt.aborted = false := by
      simp only [SearchPublication.reportable, Bool.and_eq_true, Bool.not_eq_true', decide_eq_true_eq] at reportable
      exact reportable.1
    intro saved found
    have eq : SearchPublication.capture attempt.rows attempt.stamp attempt.endgame = saved := by
      simpa [SearchPublication.step, SearchPublication.published, ready] using found
    rw [← eq]
    exact capture_safe source complete

/-- Any finite history preserves exactness; aborted attempts may contain arbitrary
    partial scores, PVs, depths and selectivity tags. -/
theorem run_safe (prior : SavedSafe (minimax mc fc b) state.last)
    (sources : ∀ attempt ∈ attempts, ExactAttempt mc fc b attempt) :
    SavedSafe (minimax mc fc b) (SearchPublication.run minimum state attempts).last := by
  induction attempts generalizing state with
  | nil => exact prior
  | cons attempt rest ih =>
    exact ih (step_safe prior (sources attempt (by simp)))
      (fun later member => sources later (List.mem_cons_of_mem _ member))

theorem phase_safe {truth : Int} (prior : SavedSafe truth state.last) :
    SavedSafe truth (SearchPublication.beginPhase state).last := prior

theorem aborted_no_score_obligation (aborted : attempt.aborted = true) :
    ExactAttempt mc fc b attempt := by
  intro complete
  cases aborted.symm.trans complete

/-- Arbitrarily many observed aborts leave the saved result exactly unchanged,
    including every score and PV field. -/
theorem aborted_preserves (aborted : ∀ attempt ∈ attempts, attempt.aborted = true) :
    SearchPublication.run minimum state attempts = state := by
  induction attempts generalizing state with
  | nil => rfl
  | cons attempt rest ih =>
    have stopped := aborted attempt (by simp)
    have same : SearchPublication.step minimum state attempt = state := by
      cases state
      simp [SearchPublication.step, SearchPublication.published, SearchPublication.ready,
        SearchPublication.nextMarks, SearchPublication.reportable, stopped]
    simp only [SearchPublication.run, List.foldl_cons, same]
    exact ih (fun later member => aborted later (List.mem_cons_of_mem _ member))

/-- A completed, reportable single-PV attempt really installs a snapshot, and
    any subsequent sequence of aborted attempts retains that exact snapshot. -/
theorem completed_then_aborted (source : ExactAttempt mc fc b attempt)
    (complete : attempt.aborted = false) (index : attempt.index = 0)
    (reportable : minimum ≤ attempt.stamp.selectivity)
    (aborted : ∀ later ∈ attempts, later.aborted = true) :
    (SearchPublication.run minimum (SearchPublication.start 1 inherited)
      (attempt :: attempts)).last =
        some (SearchPublication.capture attempt.rows attempt.stamp attempt.endgame) ∧
    SnapshotSafe (minimax mc fc b)
      (SearchPublication.capture attempt.rows attempt.stamp attempt.endgame) := by
  have rest := aborted_preserves (minimum := minimum)
    (state := SearchPublication.step minimum (SearchPublication.start 1 inherited) attempt) aborted
  constructor
  · rw [SearchPublication.run, List.foldl_cons, ← SearchPublication.run, rest]
    simp [SearchPublication.step, SearchPublication.start, SearchPublication.published,
      SearchPublication.ready, SearchPublication.nextMarks, SearchPublication.reportable,
      complete, index, reportable]
  · exact capture_safe source complete

/-- Discharge `ExactScore.Contract`'s saved-row obligation for an interrupted
    single-PV completion history. `last` is the actual saved iteration projection,
    not the live partially searched root vector. -/
theorem interrupted_rows {inherited : Option Snapshot} {last : Option Iteration} {iteration : Iteration}
    (seed : Seed mc fc b inherited)
    (sources : ∀ attempt ∈ attempts, ExactAttempt mc fc b attempt)
    (saved : last.map asSnapshot = SearchPublication.finish minimum
      (SearchPublication.run minimum (SearchPublication.start 1 inherited) attempts) .interrupted)
    (found : last = some iteration) (ending : iteration.endgame = true)
    (level : iteration.selectivity = 5) :
    ExactScore.RowScores (minimax mc fc b) iteration.rows := by
  have safe := run_safe (minimum := minimum) (state := SearchPublication.start 1 inherited)
    seed.safe sources
  have retained : (SearchPublication.run minimum (SearchPublication.start 1 inherited) attempts).last =
      some (asSnapshot iteration) := by
    simpa [SearchPublication.finish, found] using saved.symm
  exact rowsExact_project.mp ((safe _ retained ending level).1)

/-- Directly connect the retained snapshot to the extracted `into_result` body.
    The fallback obligation is relevant only when no snapshot was ever saved. -/
theorem interrupted_intoResult {inherited : Option Snapshot} {last : Option Iteration}
    {nodes : Nat} {fallback : Unit → SearchResult} {result : SearchResult}
    (seed : Seed mc fc b inherited)
    (sources : ∀ attempt ∈ attempts, ExactAttempt mc fc b attempt)
    (saved : last.map asSnapshot = SearchPublication.finish minimum
      (SearchPublication.run minimum (SearchPublication.start 1 inherited) attempts) .interrupted)
    (run : intoResult last nodes fallback = some result)
    (noSnapshot : last = none → ExactScore.Ensures mc fc b (fallback ())) :
    ExactScore.Ensures mc fc b result := by
  cases found : last with
  | none =>
    simp only [found, intoResult] at run
    change some (fallback ()) = some result at run
    rw [← Option.some.inj run]
    exact noSnapshot found
  | some iteration =>
    have retained : (SearchPublication.run minimum (SearchPublication.start 1 inherited) attempts).last =
        some (asSnapshot iteration) := by
      simpa [SearchPublication.finish, found] using saved.symm
    have safe := run_safe (minimum := minimum) (state := SearchPublication.start 1 inherited)
      seed.safe sources
    rw [found, intoResult_fromSnapshot] at run
    obtain ⟨report, built, rfl⟩ := Option.map_eq_some_iff.mp run
    intro v marked
    simp [ExactScore.scoreIfExact] at marked
    obtain ⟨⟨ending, level⟩, rfl⟩ := marked
    have score := fromSnapshot_exact (safe _ retained) built ending level
    exact score.trans (EndgameTreeCache.scaled_discValue mc fc b).symm

/-- All exact-tagged public returns after an interrupted history have the board's
    minimax score. A missing snapshot uses the existing independently validated TT
    fallback condition. There is no final `RowScores` premise. -/
def Contract : Prop :=
  ∀ mc fc b evaluate sort inherited minimum attempts last lookup nodes result,
    Seed mc fc b inherited →
    (∀ attempt ∈ attempts, ExactAttempt mc fc b attempt) →
    last.map asSnapshot = SearchPublication.finish minimum
      (SearchPublication.run minimum (SearchPublication.start 1 inherited) attempts) .interrupted →
    ExactScore.returned mc fc b evaluate sort last lookup nodes .search = some result →
    (last = none → ∀ entry v, lookup = some entry → ExactScore.CacheSource mc b entry v →
      ∃ history, ExactTableHistory.Publications mc fc history ∧
        Nonempty (Cutoffs.Lookup history b entry)) →
    ExactScore.Ensures mc fc b result

theorem contract_proven : Contract := by
  intro mc fc b evaluate sort inherited minimum attempts last lookup nodes result seed sources saved returned cached
  apply ExactScore.contract_proven mc fc b evaluate sort last lookup nodes .search result returned
  · intro _ iteration found ending level
    exact interrupted_rows seed sources saved found ending level
  · intro _ empty entry v hit source
    exact cached empty entry v hit source

/-- A poisoned live row and a newer exact tag do not replace the completed
    snapshot when the attempt was aborted; only the live node count is new. -/
theorem poisoned_live_rows_ignored :
    let saved := SearchPublication.capture [⟨10, 1024, [10, 18]⟩] ⟨12, 5⟩ true
    let stopped : Attempt := ⟨0, ⟨13, 5⟩, true, [⟨19, 16384, [19]⟩], true⟩
    let retained := SearchPublication.finish 2
      (SearchPublication.run 2 (SearchPublication.start 1 (some saved)) [stopped]) .interrupted
    retained = some saved ∧
      (retained.bind (fun s => SearchFallback.fromSnapshot s 77)).map
        (fun r => (r.best.score, r.best.pv, r.stamp.depth, r.nodes)) =
          some (1024, [10, 18], 12, 77) := by
  simp [SearchPublication.finish, SearchPublication.run, SearchPublication.start,
    SearchPublication.step, SearchPublication.published, SearchPublication.ready,
    SearchPublication.nextMarks, SearchPublication.reportable, SearchPublication.capture,
    SearchFallback.fromSnapshot]

end NeuralReversi.PublicationMir.InterruptedExact

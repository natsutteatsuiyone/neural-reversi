import NeuralReversi.PvsParallel.Interleaving.ParallelOperational
import NeuralReversi.PublicationMir.ExactScore

namespace NeuralReversi.PvsParallel.Interleaving.PublicEndgame

open PvsBoardTree (Position)
open AspirationMir
open AspirationMir.SerialTable
open PublicationMir (SearchResult SliceSort)
open PublicationMir.ExactScore (Ensures scoreIfExact)
open ParallelOperational (OperationalRun BranchComplete)

/-- Unified public `Search::run` route covering all four engine return paths:
    1. `directEndgame`: direct endgame entry (single-PV or Multi-PV),
    2. `midgameHandoff`: timed midgame iterative deepening transitioning to exact endgame (single-PV or Multi-PV),
    3. `cacheFallback`: no saved iteration, falling back to a validated TT lookup (including prior searches / other workers),
    4. `nonExactExit`: early exits (`disconnected`, `noLegalMove`, `randomOpening`). -/
inductive PublicRoute (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (r : OperationalRun mc fc) (hBranch : BranchComplete r) (b : Position) :
    SearchResult → Bool → Prop where
  | directEndgame
      (receiver : Nat) (eval : Position → Int)
      (feed : JournalRoot.Feed (r.execution hBranch) b)
      (multi : Bool) (clock : MultiPublic.Clock) (cap : Nat) (paired : Bool)
      (scratch : MoveList.Initialization.Memory Move.Move)
      (sort : SliceSort) (nodes : Nat) (noSaved : Unit → SearchResult)
      (report : SearchFallback.Report) (next : Nat)
      (capOk : 1 ≤ cap)
      (clockOk : clock.ordered ((if multi then (NativeRootBirth.initial mc b).length else 1) - 1))
      (ran : PhysicalCalls.run (r.execution hBranch) r.toRaw.nowrap b receiver eval feed multi clock cap
        paired scratch sort nodes noSaved = some (SearchResult.bestMove report, next)) :
      PublicRoute mc fc r hBranch b (SearchResult.bestMove report) multi
  | midgameHandoff {M : Type} {safe : M → Prop}
      (trial : MidgameIterate.Trial M)
      (works : MidgameIterate.Works mc b safe trial)
      (initial : MidgameIterate.State M)
      (valid : MidgameIterate.Valid mc b safe initial)
      (midSort : SliceSort) (level : TranspositionTable.Selectivity) (depth : Nat)
      (midClock : MidgameIterate.Clock M) (phase : MidgameIterate.Completed M)
      (crossed : MidgameIterate.run trial midSort level depth midClock initial = some (.endgame phase))
      (entry : Nat) (feed : JournalRoot.Feed (r.execution hBranch) b)
      (multi : Bool) (clock : MultiPublic.Clock) (sort : SliceSort)
      (nodes : Nat) (noSaved : Unit → SearchResult)
      (report : SearchFallback.Report) (next : Nat)
      (clockOk : clock.ordered ((if multi then phase.state.rows.length else 1) - 1))
      (ran : JournalExit.handoff (r.execution hBranch) b phase entry feed multi clock sort nodes noSaved =
        some (SearchResult.bestMove report, next)) :
      PublicRoute mc fc r hBranch b (SearchResult.bestMove report) multi
  | cacheFallback
      (consumer : Nat) (eval : Board.Board → Int) (sort : SliceSort) (nodes : Nat)
      (result : SearchResult)
      (ran : PhysicalCalls.unsaved (r.execution hBranch) r.toRaw.nowrap b consumer eval sort nodes = some result) :
      PublicRoute mc fc r hBranch b result false
  | nonExactExit
      (eval : Board.Board → Int) (sort : SliceSort) (last : Option PublicationMir.Iteration)
      (lookup : Option TranspositionTable.EntryData) (nodes : Nat)
      (exit : PublicationMir.ExactScore.Exit) (nonSearch : exit ≠ .search)
      (result : SearchResult)
      (ran : PublicationMir.ExactScore.returned mc fc b eval sort last lookup nodes exit = some result) :
      PublicRoute mc fc r hBranch b result false

/-- Top-level public exact-endgame contract (Task 4):
    From ANY operational parallel search run `r` (combining Task 1 ActivePointer dispatch,
    Task 2 unified split-tape replay, and Task 3 RF+HB cut closure & multi-session warm
    TT / local cache carry) whose emitted frames follow the 10 Rust `pvs::search` /
    `solve_root` branches — with NO assumption that `covered = true` and NO assumption
    on child scores — every public `Search::run` route satisfies:
    1. `PublicationMir.ExactScore.Ensures mc fc b result`:
       whenever `result` is `BestMove { is_endgame: true, selectivity: Selectivity::None, .. }`,
       `score.value() = 256 * V(B)`, AND
    2. For Multi-PV (`multi = true`) on direct endgame entry or timed midgame handoff,
       every legal move appears once in descending score order with `-256 * V(B_m)`
       (`MultiRows.ReportExact mc fc b report`). -/
def Contract : Prop :=
  ∀ (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (r : OperationalRun mc fc) (hBranch : BranchComplete r)
    (b : Position) (result : SearchResult) (multi : Bool),
    PublicRoute mc fc r hBranch b result multi →
    Ensures mc fc b result ∧
    (∀ report, result = .bestMove report → multi = true →
      report.endgame = true → report.stamp.selectivity = 5 →
      MultiRows.ReportExact mc fc b report)

theorem exact_endgame_sound : Contract := by
  intro mc fc r hBranch b result multi route
  cases route with
  | directEndgame receiver eval feed multi clock cap paired scratch sort nodes noSaved report next capOk clockOk ran =>
    have exact := fun (tag : report.endgame = true) (full : report.stamp.selectivity = 5) =>
      PhysicalCalls.direct mc fc (r.execution hBranch) r.toRaw.nowrap b receiver eval feed multi clock cap
        paired scratch sort nodes noSaved report next capOk clockOk ran tag full
    refine ⟨?_,?_⟩
    · intro v marked
      simp only [scoreIfExact,Bool.and_eq_true,beq_iff_eq] at marked
      split at marked <;> try contradiction
      rename_i flags
      cases marked
      exact (exact flags.1 flags.2).1
    · intro rep eq isMulti tag full
      cases eq
      exact (exact tag full).2 isMulti
  | midgameHandoff trial works initial valid midSort level depth midClock phase crossed entry feed multi clock sort nodes noSaved report next clockOk ran =>
    have exact := fun (tag : report.endgame = true) (full : report.stamp.selectivity = 5) =>
      JournalExit.after_midgame (r.execution hBranch) b trial works initial valid crossed feed multi clock clockOk ran tag full
    refine ⟨?_,?_⟩
    · intro v marked
      simp only [scoreIfExact,Bool.and_eq_true,beq_iff_eq] at marked
      split at marked <;> try contradiction
      rename_i flags
      cases marked
      exact (exact flags.1 flags.2).1
    · intro rep eq isMulti tag full
      cases eq
      exact (exact tag full).2 isMulti
  | cacheFallback consumer eval sort nodes result ran =>
    exact ⟨PhysicalCalls.cached (r.execution hBranch) r.toRaw.nowrap b ran,
      fun _ _ bad => by cases bad⟩
  | nonExactExit eval sort last lookup nodes exit nonSearch result ran =>
    refine ⟨?_,fun _ _ bad => by cases bad⟩
    intro v marked
    have ⟨isSearch,_⟩ := PublicationMir.ExactScore.origin ran marked
    exact False.elim (nonSearch isSearch)

end NeuralReversi.PvsParallel.Interleaving.PublicEndgame

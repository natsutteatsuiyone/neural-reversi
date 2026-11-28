module
public import NeuralReversi.Threading.SplitWindow.Stores
public import NeuralReversi.Threading.SplitReturn.Observation

@[expose] public section

namespace NeuralReversi.Threading.SplitReturn

open Aspiration (Window)
open AtomicWord (Event Read Write signed32At)
open CutoffVisibility (History)
open MoveList.Concurrent (Claims advance)
open PvsMoveLoop (Bound scoreMax)
open SplitCompletion (Action Activation Retirement Worker)
open SplitResult (Merge Report nodeValue slots)

inductive Departure where
  | exhausted
  | cutoff
  | cancelled

inductive Loop (len : Nat) (last : BitVec w) : List Nat → List Report → Departure → Prop
  | exhausted : advance len seen = none → Claims len seen after last →
      Loop len last [] [] .exhausted
  | cutoff (pending : Option Nat) : Loop len last pending.toList [] .cutoff
  | cancelled (pending : Option Nat) : Loop len last pending.toList [] .cancelled
  | publish (report : Report) : Loop len last tickets reports reason →
      Loop len last (report.index :: tickets) (report :: reports) reason

private theorem Loop.completed {last : BitVec w} (loop : Loop len last tickets reports reason)
    (drained : reason = .exhausted) :
    advance len last = none ∧
      (∀ index ∈ tickets, ∃ report ∈ reports, report.index = index) := by
  induction loop with
  | exhausted empty tail =>
    cases tail with
    | done => exact ⟨empty, by simp⟩
    | take next _ => rw [empty] at next; cases next
  | cutoff => cases drained
  | cancelled => cases drained
  | publish report _ ih =>
    obtain ⟨empty, covered⟩ := ih drained
    refine ⟨empty, ?_⟩
    intro index member
    rcases List.mem_cons.mp member with head | tail
    · exact ⟨report, by simp, head.symm⟩
    · obtain ⟨other, published, same⟩ := covered index tail
      exact ⟨other, List.mem_cons_of_mem _ published, same⟩

private theorem Loop.report_tickets {last : BitVec w} (loop : Loop len last tickets reports reason) :
    (reports.map Report.index).Sublist tickets := by
  induction loop with
  | exhausted => exact .slnil
  | cutoff | cancelled => exact List.nil_sublist _
  | publish report _ ih => exact .cons_cons report.index ih

structure Work where
  tickets : List Nat
  reports : List Report
  departure : Departure

def Released (actions : List Action) (worker : Worker) : Prop :=
  ∃ event, .finish worker event ∈ actions

structure Dispatches (history : History) (wait : Event) (actions : List Action)
    (owner : Worker) (len : Nat) (last : BitVec w) (tickets : List Nat) (reports : List Report)
    (check : Observation.OwnerCheck history) (localCutoff : Prop) where
  work : Worker → Work
  workers : List Worker
  participant : ∀ worker ∈ workers, worker = owner ∨ ∃ event, .book worker event ∈ actions
  ticketPartition : tickets.Perm (workers.flatMap fun worker => (work worker).tickets)
  reportPartition : reports.Perm (workers.flatMap fun worker => (work worker).reports)
  finished : ∀ worker, Released actions worker →
    Loop len last (work worker).tickets (work worker).reports (work worker).departure
  cut : ∀ worker, Released actions worker → (work worker).departure = .cutoff → localCutoff
  cancelled : ∀ worker, Released actions worker → (work worker).departure = .cancelled →
    localCutoff ∨ Observation.ExternalStop history check wait

private theorem participant_finished
    (retirement : Retirement history mask activation)
    (participated : worker = activation.owner ∨ ∃ event, .book worker event ∈ activation.actions) :
    Released activation.actions worker := by
  have done := (SplitCompletion.emptyReadContract_proven history mask activation retirement).2
  rcases participated with owner | ⟨event, booked⟩
  · exact owner.symm ▸ done.1
  · exact done.2 worker event booked

def UniqueReportsContract : Prop :=
  ∀ (history : History) (mask : List Write) (activation : Activation mask)
    (retirement : Retirement history mask activation) (w len : Nat)
    (start last : BitVec w) (issued : List (BitVec w)) (reports : List Report)
    (check : Observation.OwnerCheck history) (localCutoff : Prop),
    len < 2 ^ w → Claims len start issued last →
    Dispatches history retirement.read.event activation.actions activation.owner len last
      (issued.map BitVec.toNat) reports check localCutoff →
    (reports.map Report.index).Nodup

theorem uniqueReportsContract_proven : UniqueReportsContract := by
  intro history mask activation retirement w len start last issued reports check localCutoff capacity claims runs
  have individually (worker : Worker) (member : worker ∈ runs.workers) :=
    (runs.finished worker (participant_finished retirement (runs.participant worker member))).report_tickets
  have group (workers : List Worker) (member : ∀ worker ∈ workers, worker ∈ runs.workers) :
      (workers.flatMap fun worker => (runs.work worker).reports.map Report.index).Sublist
        (workers.flatMap fun worker => (runs.work worker).tickets) := by
    induction workers with
    | nil => exact .slnil
    | cons worker others ih =>
      exact (individually worker (member worker (by simp))).append
        (ih fun other inTail => member other (List.mem_cons_of_mem _ inTail))
  have distinct := runs.ticketPartition.nodup_iff.mp (claims.nodup capacity)
  apply (runs.reportPartition.map Report.index).nodup_iff.mpr
  rw [List.map_flatMap]
  exact (group runs.workers fun _ member => member).nodup distinct

def CoverageContract : Prop :=
  ∀ (history : History) (mask : List Write) (activation : Activation mask)
    (retirement : Retirement history mask activation) (w len : Nat) (last : BitVec w)
    (tickets : List Nat) (reports : List Report) (check : Observation.OwnerCheck history) (localCutoff : Prop),
    Dispatches history retirement.read.event activation.actions activation.owner len last tickets reports
      check localCutoff →
    check.stopped = false →
    localCutoff ∨ (advance len last = none ∧
      ∀ index ∈ tickets, ∃ report ∈ reports, report.index = index)

theorem coverageContract_proven : CoverageContract := by
  intro history mask activation retirement w len last tickets reports check localCutoff runs accepts
  by_cases marked : localCutoff
  · exact .inl marked
  · have done := (SplitCompletion.emptyReadContract_proven history mask activation retirement).2
    have exhausted (worker : Worker) (released : Released activation.actions worker) :
        (runs.work worker).departure = .exhausted := by
      cases reason : (runs.work worker).departure with
      | exhausted => rfl
      | cutoff => exact False.elim (marked (runs.cut worker released reason))
      | cancelled =>
        have stopped := Observation.cancellationGateContract_proven history check retirement.read.event
          ((runs.cancelled worker released reason).resolve_left marked)
        cases accepts.symm.trans stopped
    have completed (worker : Worker) (released : Released activation.actions worker) :=
      (runs.finished worker released).completed (exhausted worker released)
    refine .inr ⟨(completed activation.owner done.1).1, ?_⟩
    intro index member
    obtain ⟨worker, participant, ticket⟩ := List.mem_flatMap.mp (runs.ticketPartition.mem_iff.mp member)
    have released := participant_finished retirement (runs.participant worker participant)
    obtain ⟨report, published, same⟩ := (completed worker released).2 index ticket
    exact ⟨report, runs.reportPartition.mem_iff.mpr (List.mem_flatMap_of_mem participant published), same⟩

def stateAfter (pv : Bool) (beta : Int) (initial : SplitWindow.State) (reports : List Report) :
    SplitWindow.State :=
  SplitWindow.scan pv beta initial (reports.map Report.score)

def LocalCutoff (pv : Bool) (beta : Int) (initial : SplitWindow.State) (reports : List Report) : Prop :=
  ∃ (earlier : List Report) (report : Report) (later : List Report),
    reports = earlier ++ report :: later ∧
      SplitWindow.marksCutoff pv beta (stateAfter pv beta initial earlier) report.score = true

def ReportSteps (pv : Bool) (beta : Int) (initial : SplitWindow.State)
    (value : Nat → Int) (reports : List Report) : Prop :=
  ∀ earlier report later, reports = earlier ++ report :: later →
    let locked := stateAfter pv beta initial earlier
    report.searchedAlpha ≤ locked.alpha ∧ Bound (value report.index) report.searchedAlpha beta report.score

theorem stateAfter_append (pv : Bool) (beta : Int) (initial : SplitWindow.State)
    (earlier later : List Report) :
    stateAfter pv beta initial (earlier ++ later) =
      stateAfter pv beta (stateAfter pv beta initial earlier) later := by
  simp [stateAfter, SplitWindow.scan, List.foldl_append]

theorem mergeRemaining (pv : Bool) (original beta : Int) (initial : SplitWindow.State)
    (value : Nat → Int) (reports earlier remaining : List Report)
    (window : Window initial.alpha beta) (capped : initial.alpha ≤ max original initial.best)
    (steps : ReportSteps pv beta initial value reports)
    (partition : reports = earlier ++ remaining) :
    Merge value original beta (stateAfter pv beta initial earlier).best remaining
      (stateAfter pv beta initial reports).best := by
  induction remaining generalizing earlier with
  | nil => simpa [partition] using
      (Merge.done : Merge value original beta (stateAfter pv beta initial earlier).best []
        (stateAfter pv beta initial earlier).best)
  | cons report rest ih =>
    have one := steps earlier report rest partition
    have usable := (SplitWindow.stateContract_proven pv original beta initial
      (earlier.map Report.score) window capped).2.2.1
    have later := ih (earlier ++ [report]) (by simpa [List.append_assoc] using partition)
    rw [stateAfter_append] at later
    change Merge value original beta
      (SplitWindow.publish pv beta (stateAfter pv beta initial earlier) report.score).best rest
      (stateAfter pv beta initial reports).best at later
    exact .accept (Int.le_trans one.1 usable) one.2 later

def checked (stop : Bool) (score : Int) : Option Int :=
  if stop then none else some score

def OwnerBestContract : Prop :=
  ∀ (history : History) (writes : List Write) (load : Read writes) (last : Fin writes.length)
    {mask : List Write} {activation : Activation mask} (retirement : Retirement history mask activation)
    pv beta initial (reports : List Report),
    Observation.ScoreRead history retirement load last →
    SplitWindow.BestStoreWindow writes last initial.best (reports.map Report.score) →
    signed32At writes load.source = (stateAfter pv beta initial reports).best

theorem ownerBestContract_proven : OwnerBestContract := by
  intro history writes load last mask activation retirement pv beta initial reports read stores
  exact (Observation.scoreReadContract_proven history writes load last retirement read).2.trans
    (SplitWindow.bestStoreContract_proven writes last pv beta initial (reports.map Report.score) stores)

def ReturnedScoreContract : Prop :=
  ∀ (history : History) (mask : List Write) (activation : Activation mask)
    (retirement : Retirement history mask activation)
    (w len : Nat) (start last : BitVec w) (issued : List (BitVec w))
    (pv : Bool) (check : Observation.OwnerCheck history) (original beta prefixValue : Int)
    (initial : SplitWindow.State)
    (value : Nat → Int) (reports : List Report)
    (writes : List Write) (load : Read writes) (lastScore : Fin writes.length) (returned : Int),
    len < 2 ^ w → Claims len start issued last →
    Window initial.alpha beta → initial.alpha ≤ max original initial.best →
    (pv = false → beta = initial.alpha + 1) →
    Bound prefixValue original beta initial.best → prefixValue ≤ scoreMax →
    (∀ index ∈ slots start.toNat len, value index ≤ scoreMax) →
    ReportSteps pv beta initial value reports →
    (∀ report ∈ reports, report.index ∈ slots start.toNat len) →
    Dispatches history retirement.read.event activation.actions activation.owner len last
      (issued.map BitVec.toNat) reports check (LocalCutoff pv beta initial reports) →
    Observation.ScoreRead history retirement load lastScore →
    SplitWindow.BestStoreWindow writes lastScore initial.best (reports.map Report.score) →
    checked check.stopped (signed32At writes load.source) = some returned →
    Bound (nodeValue value start.toNat len prefixValue) original beta returned

theorem returnedScoreContract_proven : ReturnedScoreContract := by
  intro history mask activation retirement w len start last issued pv check original beta
    prefixValue initial value reports writes load lastScore returned capacity claims window capped narrow
    seed cap caps steps valid runs read stores result
  have observed := ownerBestContract_proven history writes load lastScore retirement
    pv beta initial reports read stores
  rw [observed] at result
  have accepted : check.stopped = false := by
    cases stop : check.stopped <;> simp [checked, stop] at result ⊢
  have final : (stateAfter pv beta initial reports).best = returned := by simpa [checked, accepted] using result
  rw [← final]
  have merge : Merge value original beta initial.best reports (stateAfter pv beta initial reports).best :=
    mergeRemaining pv original beta initial value reports [] reports window capped steps rfl
  apply SplitResult.splitResultContract_proven w len start last issued value prefixValue initial.best
    original beta _ reports capacity claims seed caps cap merge valid
  rcases coverageContract_proven history mask activation retirement w len last (issued.map BitVec.toNat)
    reports check _ runs accepted with cutoff | drained
  · obtain ⟨earlier, report, later, partition, marked⟩ := cutoff
    apply Or.inr
    have witness := SplitWindow.cutoffWitnessContract_proven pv original beta initial
      (earlier.map Report.score) (later.map Report.score) report.score window capped narrow marked
    simpa only [partition, stateAfter, List.map_append, List.map_cons] using witness
  · exact .inl drained

example : Loop 2 (1#32) [0] [] .cancelled ∧ ¬ Loop 2 (1#32) [0] [] .exhausted := by
  refine ⟨.cancelled (some 0), ?_⟩
  intro trace
  have impossible := (trace.completed rfl).2 0 (by simp)
  simp at impossible

end NeuralReversi.Threading.SplitReturn

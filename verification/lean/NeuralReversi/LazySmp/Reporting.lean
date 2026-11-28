module
public import NeuralReversi.LazySmp.Schedule
public import NeuralReversi.LazySmp.Voting
public import NeuralReversi.Threading.CutoffVisibility

@[expose] public section

namespace NeuralReversi.LazySmp

structure Trial (Payload : Type) where
  snapshot : Payload
  work : Nat
  aborted : Bool

structure WorkerReport (Payload : Type) where
  last : Option Payload
  nodes : Nat

def runTrials : List (Trial Payload) → WorkerReport Payload → WorkerReport Payload
  | [], state => state
  | trial :: rest, state =>
    let nodes := state.nodes + trial.work
    if trial.aborted then ⟨state.last, nodes⟩
    else runTrials rest ⟨some trial.snapshot, nodes⟩

def completed : List (Trial Payload) → List Payload
  | [] => []
  | trial :: rest => if trial.aborted then [] else trial.snapshot :: completed rest

def consumed : List (Trial Payload) → List (Trial Payload)
  | [] => []
  | trial :: rest => trial :: if trial.aborted then [] else consumed rest

def ReportContract : Prop :=
  ∀ (Payload : Type) (trials : List (Trial Payload)) (state : WorkerReport Payload),
    (runTrials trials state).last = (completed trials).getLast?.or state.last ∧
    (runTrials trials state).nodes = state.nodes + ((consumed trials).map Trial.work).sum

theorem reportContract_proven : ReportContract := by
  intro Payload trials state
  induction trials generalizing state with
  | nil => simp [runTrials, completed, consumed]
  | cons trial rest ih =>
    cases aborted : trial.aborted
    · have report := ih ⟨some trial.snapshot, state.nodes + trial.work⟩
      have last : (trial.snapshot :: completed rest).getLast?.or state.last =
          (completed rest).getLast?.or (some trial.snapshot) := by
        change ([trial.snapshot] ++ completed rest).getLast?.or state.last = _
        rw [List.getLast?_append]
        cases (completed rest).getLast? <;> rfl
      simpa [runTrials, completed, consumed, aborted, last, Nat.add_assoc] using report
    · simp [runTrials, completed, consumed, aborted]

def cumulative : Nat → List Nat → Prop
  | _, [] => True
  | previous, nodes :: rest => previous ≤ nodes ∧ cumulative nodes rest

def spent : Nat → List Nat → List Nat
  | _, [] => []
  | previous, nodes :: rest => (nodes - previous) :: spent nodes rest

def CounterContract : Prop :=
  ∀ (initial : Nat) (readings : List Nat), cumulative initial readings →
    initial + (spent initial readings).sum = readings.getLast?.getD initial

theorem counterContract_proven : CounterContract := by
  intro initial readings
  induction readings generalizing initial with
  | nil => simp [spent]
  | cons nodes readings ih =>
    rintro ⟨monotone, later⟩
    have last : (nodes :: readings).getLast?.getD initial = readings.getLast?.getD nodes := by
      change ([nodes] ++ readings).getLast?.getD initial = _
      rw [List.getLast?_append]
      cases readings.getLast? <;> rfl
    rw [last, ← ih _ later]
    simp [spent]
    omega

def runScheduled (helper ply maxDepth : Nat) (observe : Nat → Trial Payload) :
    WorkerReport (Nat × Payload) :=
  let observations := (eligibleDepths helper ply maxDepth).map fun depth =>
    let t := observe depth
    (⟨(depth, t.snapshot), t.work, t.aborted⟩ : Trial (Nat × Payload))
  runTrials observations ⟨none, 0⟩

private theorem completed_member {trials : List (Trial Payload)} (member : value ∈ completed trials) :
    ∃ t ∈ trials, t.snapshot = value ∧ t.aborted = false := by
  induction trials with
  | nil => simp [completed] at member
  | cons t rest ih =>
    cases aborted : t.aborted
    · simp only [completed, aborted, Bool.false_eq_true, ite_false, List.mem_cons] at member
      rcases member with rfl | later
      · exact ⟨t, by simp, rfl, aborted⟩
      · obtain ⟨prior, present, h, ok⟩ := ih later
        exact ⟨prior, by simp [present], h, ok⟩
    · simp [completed, aborted] at member

def ScheduledReportContract : Prop :=
  ∀ (Payload : Type) (helper ply maxDepth : Nat) (observe : Nat → Trial Payload)
    (depth : Nat) (result : Payload),
    (runScheduled helper ply maxDepth observe).last = some (depth, result) →
    1 ≤ depth ∧ depth ≤ maxDepth ∧ skipsDepth helper (depth + ply) = false ∧
      (observe depth).snapshot = result ∧ (observe depth).aborted = false

theorem scheduledReportContract_proven : ScheduledReportContract := by
  intro Payload helper ply maxDepth observe depth result found
  let observations := (eligibleDepths helper ply maxDepth).map fun d =>
    let t := observe d
    (⟨(d, t.snapshot), t.work, t.aborted⟩ : Trial (Nat × Payload))
  have last := (reportContract_proven (Nat × Payload) observations ⟨none, 0⟩).1
  change (runTrials observations ⟨none, 0⟩).last = _ at found
  rw [found] at last
  have member : (depth, result) ∈ completed observations :=
    List.mem_of_getLast? (by simpa using last.symm)
  obtain ⟨trial, present, matched, ok⟩ := completed_member member
  obtain ⟨d, eligible, rfl⟩ := List.mem_map.mp present
  have eq : (d, (observe d).snapshot) = (depth, result) := matched
  cases eq
  obtain ⟨lo, hi, allowed⟩ := (eligibleDepths_spec ..).1 eligible
  exact ⟨lo, hi, allowed, rfl, ok⟩

def aggregate (reports : List (WorkerReport Payload)) : Nat × List Payload :=
  reports.foldl (fun (nodes, rows) report => (nodes + report.nodes, rows ++ report.last.toList)) (0, [])

def AggregateContract : Prop :=
  ∀ (Payload : Type) (reports : List (WorkerReport Payload)),
    aggregate reports = ((reports.map WorkerReport.nodes).sum, reports.filterMap WorkerReport.last)

theorem aggregateContract_proven : AggregateContract := by
  have fold_spec {Payload : Type} (reports : List (WorkerReport Payload)) (nodes : Nat) (rows : List Payload) :
      reports.foldl (fun (n, rs) (report : WorkerReport Payload) =>
        (n + report.nodes, rs ++ report.last.toList)) (nodes, rows) =
      (nodes + (reports.map WorkerReport.nodes).sum, rows ++ reports.filterMap WorkerReport.last) := by
    induction reports generalizing nodes rows with
    | nil => simp
    | cons report rest ih =>
      rw [List.foldl_cons, ih]
      cases h : report.last <;> simp [h, Nat.add_assoc, List.append_assoc]
  intro Payload reports
  simp [aggregate, fold_spec]

structure Snapshot (Payload : Type) where
  ballot : Ballot
  payload : Payload

def publish (done timed multiPV : Bool) (main : Option (Snapshot Payload))
    (helpers : List (Snapshot Payload)) : Option (Snapshot Payload) :=
  if done && timed && !multiPV then
    let candidates := main.toList ++ helpers
    (winner (candidates.map Snapshot.ballot)).bind fun best => candidates[best.1]?
  else main

def PublishContract : Prop :=
  ∀ (Payload : Type) (done timed multiPV : Bool) (main : Option (Snapshot Payload))
    (helpers : List (Snapshot Payload)),
    let candidates := main.toList ++ helpers
    ((done && timed && !multiPV) = false → publish done timed multiPV main helpers = main) ∧
    ((done && timed && !multiPV) = true → candidates ≠ [] →
      ∃ i snapshot, publish done timed multiPV main helpers = some snapshot ∧
        candidates[i]? = some snapshot ∧ winner (candidates.map Snapshot.ballot) = some (i, snapshot.ballot))

theorem publishContract_proven : PublishContract := by
  intro Payload done timed multiPV main helpers
  dsimp
  constructor
  · intro gate
    simp [publish, gate]
  · intro gate present
    let candidates := main.toList ++ helpers
    have nonempty : candidates.map Snapshot.ballot ≠ [] := by simpa [candidates] using present
    obtain ⟨best, wins⟩ : ∃ best, winner (candidates.map Snapshot.ballot) = some best := by
      cases ballots : candidates.map Snapshot.ballot with
      | nil => exact (nonempty ballots).elim
      | cons head tail => exact ⟨_, rfl⟩
    have source := (votingContract_proven _ _ wins).1
    rw [List.getElem?_map] at source
    cases saved : candidates[best.1]? with
    | none => simp [saved] at source
    | some snap =>
      rw [saved] at source
      have same : snap.ballot = best.2 := Option.some.inj source
      refine ⟨best.1, snap, ?_, saved, ?_⟩
      · simp only [publish, gate, ite_true]
        change ((winner (candidates.map Snapshot.ballot)).bind fun best => candidates[best.1]?) = _
        rw [wins]
        exact saved
      · rw [wins, same]

def ReportVoteContract : Prop :=
  ∀ (Payload : Type) (done timed multiPV : Bool) (main : Option (Snapshot Payload))
    (reports : List (WorkerReport (Snapshot Payload))) (selected : Snapshot Payload),
    publish done timed multiPV main (aggregate reports).2 = some selected →
    main = some selected ∨ ∃ report ∈ reports, report.last = some selected

theorem reportVoteContract_proven : ReportVoteContract := by
  intro Payload done timed multiPV main reports selected found
  cases gate : done && timed && !multiPV
  · exact .inl (by simpa [publish, gate] using found)
  · have member : selected ∈ main.toList ++ (aggregate reports).2 := by
      simp only [publish, gate, ite_true] at found
      cases chosen : winner ((main.toList ++ (aggregate reports).2).map Snapshot.ballot) with
      | none => rw [chosen] at found; cases found
      | some record =>
        rw [chosen] at found
        exact List.mem_of_getElem? found
    rw [aggregateContract_proven] at member
    rcases List.mem_append.mp member with own | helper
    · exact .inl (by simpa using own)
    · exact .inr (List.mem_filterMap.mp helper)

open Threading
open AtomicWord (Event ReadWindow SequencedBefore)
open CutoffVisibility (History HappensBefore)

structure JobReturn (history : History) (waited : AtomicWord.Read history.epochs)
    (booked finished : Fin history.epochs.length) (published drained : Event) : Prop where
  matching : finished.val = booked.val + 1
  busy : (history.epochs.get booked).value = 1
  idle : (history.epochs.get waited.source).value = 0
  window : ReadWindow waited (HappensBefore history) booked finished.val
  read : waited ∈ history.epochReads
  acquire : waited.acquire = true
  release : (history.epochs.get finished).release = true
  publishedBeforeFinish : SequencedBefore published (history.epochs.get finished).event
  drainAfterWait : SequencedBefore waited.event drained

def JobHandoffContract : Prop :=
  ∀ (history : History) (waited : AtomicWord.Read history.epochs)
    (booked finished : Fin history.epochs.length) (published drained : Event),
    JobReturn history waited booked finished published drained →
    waited.source = finished ∧ HappensBefore history published drained

theorem jobHandoffContract_proven : JobHandoffContract := by
  intro history waited booked finished published drained matched
  obtain ⟨lo, hi⟩ := matched.window.bounds
  have notBooked : waited.source ≠ booked := by
    intro same
    have idle := matched.idle
    rw [same, matched.busy] at idle
    contradiction
  have afterBooking := Nat.lt_of_le_of_ne lo fun eq => notBooked (Fin.ext eq.symm)
  have last : waited.source = finished := by
    apply Fin.ext
    have := matched.matching
    omega
  have sync : HappensBefore history (history.epochs.get finished).event waited.event := by
    apply HappensBefore.synchronized finished waited matched.read matched.acquire
    unfold CutoffVisibility.InReleaseSequence AtomicWord.InReleaseSequence
    rw [last]
    exact ⟨matched.release, Nat.le_refl _, fun _ lo hi => False.elim (by omega)⟩
  exact ⟨last, .trans (.sequenced matched.publishedBeforeFinish)
    (.trans sync (.sequenced matched.drainAfterWait))⟩

example : (runTrials [⟨1, 4, false⟩, ⟨2, 6, true⟩, ⟨3, 20, false⟩] ⟨none, 0⟩).last =
    some (1 : Nat) := by decide

example : (runTrials [⟨1, 4, false⟩, ⟨2, 6, true⟩, ⟨3, 20, false⟩] ⟨none, 0⟩).nodes =
    10 := by decide

end NeuralReversi.LazySmp

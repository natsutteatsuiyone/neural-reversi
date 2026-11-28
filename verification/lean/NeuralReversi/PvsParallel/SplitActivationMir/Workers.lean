module
public import NeuralReversi.PvsParallel.SplitActivationMir.Activation
public import NeuralReversi.PvsParallel.SplitSchedulerMir.Worker

@[expose] public section

namespace NeuralReversi.PvsParallel.SplitActivationMir

open Threading
open AtomicWord (Event Write Read)
open CutoffVisibility (History HappensBefore)
open SplitCompletion (Worker Action)

theorem wake_from_idle_branch (history : History) (writes : List Write) (idle booked : Fin writes.length)
    (load : Read writes) (hasHelperJob : Bool) (consecutive : booked.val = idle.val + 1)
    (wasIdle : (writes.get idle).value = 0) (posted : (writes.get booked).value = 1)
    (released : (writes.get booked).release = true) (acquired : load.acquire = true)
    (dispatched : entersWorker (decide ((writes.get load.source).value = 1)) hasHelperJob = true)
    (window : AtomicWord.ReadWindow load (HappensBefore history) idle booked.val)
    (sync : ∀ i : Fin writes.length, load.acquire = true →
      AtomicWord.InReleaseSequence writes i load.source → history.otherSync (writes.get i).event load.event) :
    load.source = booked ∧ HappensBefore history (writes.get booked).event load.event ∧ hasHelperJob = false := by
  obtain ⟨on, noJob⟩ := Bool.and_eq_true_iff.mp dispatched
  have ready : (writes.get load.source).value = 1 := of_decide_eq_true on
  let wake : SplitPublication.SearchWake history writes :=
    ⟨idle,booked,consecutive,wasIdle,posted,released,load,acquired,ready,window,sync⟩
  obtain ⟨source, causal⟩ := SplitPublication.wakeContract_proven history writes wake
  exact ⟨source, causal, by cases hasHelperJob <;> simp_all⟩

theorem raw_late_booking (history : History) (flags : List Write) (flag : Read flags)
    (latest : Fin flags.length) (current : Fields) (w : Worker) (e : Event) (idle : Bool)
    (value : (flags.get latest).value = if current.accepting then 1 else 0)
    (protectedRead : SplitPublication.LockedFlagRead history flag latest latest.val)
    (entered : gate true (decide ((flags.get flag.source).value = 1)) idle = true)
    (tailRun : Run (onEnroll current w e) rest) : Run current (.book w e :: rest) := by
  have bounds := protectedRead.window.bounds
  have source : flag.source = latest := Fin.ext (by omega)
  rw [source, value] at entered
  cases h : current.accepting <;> simp [h, gate] at entered
  exact .enroll w e (idle := idle) (by simp [gate, h, entered]) tailRun

private theorem none_after_close (later : Run f xs) (closed : f.accepting = false)
    (unassigned : f.assignment w = none) : ∀ event, Action.finish w event ∉ xs := by
  induction later with
  | done => simp
  | enroll worker event ok _ _ => simp [gate, closed] at ok
  | @finish f rest worker receipt release captured _ ih =>
    have other : w ≠ worker := by intro h; subst worker; rw [unassigned] at captured; contradiction
    intro e member
    rcases List.mem_cons.mp member with eq | following
    · injection eq with eqWorker _; exact other eqWorker
    · apply ih (by rfl) _ e following
      simp [onLeave_eq,leave,setAssignment,other,unassigned]

def finishing : Action → Option Worker
  | .book .. => none | .finish w _ => some w

private theorem run_unique (later : Run f xs) : (xs.filterMap finishing).Nodup := by
  induction later with
  | done => simp
  | enroll w e _ _ ih => exact ih
  | @finish rest f w receipt e _ next ih =>
    rw [List.filterMap_cons]
    change (w :: rest.filterMap finishing).Nodup
    refine List.nodup_cons.mpr ⟨?_, ih⟩
    intro mem
    obtain ⟨act, found, eq⟩ := List.mem_filterMap.mp mem
    cases act with
    | book _ _ => cases eq
    | finish who atRmw =>
      have equal : who = w := Option.some.inj eq
      subst who
      exact none_after_close next rfl (by simp [onLeave_eq,leave,setAssignment]) atRmw found

private theorem startup_has_no_releases (early : Startup f xs after) : xs.filterMap finishing = [] := by
  induction early with
  | done => rfl
  | enroll _ _ _ _ ih => simpa [finishing] using ih

theorem unique_releases (early : Startup (onInit owner mark) first f) (later : Run f rest) :
    ((first ++ rest).filterMap finishing).Nodup := by
  rw [List.filterMap_append, startup_has_no_releases early]
  exact run_unique later

theorem retired_workers (history : History) (mask scores : List Write)
    (owner : Worker) (clear mark : Event) (early : Startup (onInit owner mark) first f) (later : Run f rest)
    (mo : ModificationOrder mask (code owner clear mark (first ++ rest)))
    (clock : SplitSchedulerMir.Clock) (log : List (SplitSchedulerMir.Guard mask)) (source : Fin scores.length)
    (run : SplitSchedulerMir.Returned SplitSchedulerMir.idle mask 0 log)
    (openPool : SplitSchedulerMir.Guard.shutdown true ∉ log)
    (memory : SplitSchedulerMir.OwnerMemory history mask scores (activation owner clear mark early later mo)
      clock log source) :
    ((first ++ rest).filterMap finishing).Nodup ∧
    (∃ event, Action.finish owner event ∈ first ++ rest) ∧
    (∀ w event, Action.book w event ∈ first ++ rest → ∃ release, Action.finish w release ∈ first ++ rest) := by
  let act := activation owner clear mark early later mo
  obtain ⟨retired, _, _, _⟩ := SplitSchedulerMir.retired_before_score history mask scores act clock log source
    run openPool memory
  exact ⟨unique_releases early later, (SplitCompletion.emptyReadContract_proven history mask act retired).2⟩

end NeuralReversi.PvsParallel.SplitActivationMir

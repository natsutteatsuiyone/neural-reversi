import NeuralReversi.PvsParallel.Interleaving.Schedule

namespace NeuralReversi.PvsParallel.Interleaving

open Aspiration (Window SoundResult)
open PvsMoveLoop (Bound ChildRelation scoreMax)
open Threading
open Schedule (Task Result)

/-- One owner wait and its subsequent stop check. Mask changes and scored reports
    are projections of the same sequence of worker steps. -/
structure Join (t : Task) (done_ : ChildRelation Nat) (len : Nat) (start : BitVec w) where
  history : CutoffVisibility.History
  mask : List AtomicWord.Write
  activation : SplitCompletion.Activation mask
  wait : SplitCompletion.Retirement history mask activation
  check : SplitReturn.Observation.OwnerCheck history
  state : Tickets.State w
  path : Schedule.RunWith (SplitReturn.Observation.ExternalStop history check wait.read.event)
    t done_ len start activation.owner activation.actions state

def Join.answer (j : Join t done_ len start) : Option Int :=
  SplitReturn.checked j.check.stopped (t.locked j.state.reports).best

private theorem mapPath {p q : ChildRelation Nat} (path : Schedule.RunWith stop t p len start w mask state)
    (lift : ∀ i pv a z v, p i pv a z v → q i pv a z v) : Schedule.RunWith stop t q len start w mask state := by
  induction path with
  | initial => exact .initial
  | book w e _ step ih => exact .book w e ih step
  | finish w e _ step ih => exact .finish w e ih step
  | work _ step effect cause calls ih =>
    apply Retirement.ExecutionWith.work ih step effect cause
    intro post present
    obtain ⟨job,same,scout,wide⟩ := calls post present
    exact ⟨job,same,lift _ _ _ _ _ scout,fun used => lift _ _ _ _ _ (wide used)⟩

/-- Retain the same tickets, stop observation and score when a returned child
    is also present in a larger history. -/
def Join.mapCalls {p q : ChildRelation Nat} (j : Join t p len start)
    (lift : ∀ i pv a z v, p i pv a z v → q i pv a z v) : Join t q len start :=
  {j with path := mapPath j.path lift}

theorem Join.finished (j : Join t done_ len start) (returns : j.answer = some score) :
    j.state.retired ∧ j.state.outside = false ∧ (t.locked j.state.reports).best = score := by
  have active : j.check.stopped = false := by
    cases h : j.check.stopped <;> simp_all [Join.answer,SplitReturn.checked]
  have running : j.state.outside = false := by
    cases h : j.state.outside with
    | false => rfl
    | true =>
      have stopped := SplitReturn.Observation.cancellationGateContract_proven _ _ _ (j.path.external_cause h)
      simp [active] at stopped
  exact ⟨j.path.retired j.wait,running,by simpa [Join.answer,SplitReturn.checked,active] using returns⟩

theorem Join.accepted {value : Nat → Int} {start : BitVec w} (j : Join t done_ len start)
    (fits : len < 2^w) (returns : j.answer = some score)
    (window : Window t.entry.alpha t.beta) (cap : t.entry.alpha ≤ max original t.entry.best)
    (narrow : t.pv = false → t.beta = t.entry.alpha+1)
    (seed : Bound prefixValue original t.beta t.entry.best) (prefixCap : prefixValue ≤ scoreMax)
    (limits : ∀ i ∈ SplitResult.slots start.toNat len, value i ≤ scoreMax)
    (children : ∀ i, start.toNat ≤ i → i < len → ∀ cp a b s, Window a b →
      (cp = false → b = a+1) → done_ i cp a b s → SoundResult (-(value i)) a b s) :
    Result t original value start.toNat len prefixValue j.state ∧
      (t.locked j.state.reports).best = score := by
  obtain ⟨retired,normal,same⟩ := j.finished returns
  exact ⟨Schedule.Run.accepted j.path fits retired normal window cap narrow seed prefixCap limits children,same⟩

/-- Connect the modeled return to the owner's relaxed score word read after the mask
    Acquire. The existing best-store rule keeps the stores made by improving reports. -/
theorem Join.physical_answer (j : Join t done_ len start) (load : AtomicWord.Read writes) (last : Fin writes.length)
    (read : SplitReturn.Observation.ScoreRead j.history j.wait load last)
    (stores : SplitWindow.BestStoreWindow writes last t.entry.best ((Schedule.reports j.state).map SplitResult.Report.score)) :
    SplitReturn.checked j.check.stopped (AtomicWord.signed32At writes load.source) = j.answer := by
  rw [SplitReturn.ownerBestContract_proven j.history writes load last j.wait t.pv t.beta t.entry
    (Schedule.reports j.state) read stores]
  rfl

end NeuralReversi.PvsParallel.Interleaving

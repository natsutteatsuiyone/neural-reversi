import NeuralReversi.Threading.SearchAbort

namespace NeuralReversi.Threading.SearchAbort.Persistent

open AtomicWord (Write Read SequencedBefore)

/-- Updates permitted while one search generation remains active. Starting a
    new generation is deliberately absent: `begin_search` resets the marker
    and is performed before dispatching the next request. Stale targeted
    requests and failed/spurious compare-exchanges may leave the word alone. -/
inductive Update : BitVec 64 → BitVec 64 → Prop where
  | unchanged (word) : Update word word
  | request (word) : Update word (SearchAbort.request word)
  | targeted (word target) : Update word ((requestFor word target).getD word)

theorem Update.stopped (step : Update before after) (prior : Stopped before) : Stopped after := by
  cases step with
  | unchanged => exact prior
  | request => exact request_stops _
  | targeted target =>
    have guarded := (generationWordContract_proven before).2.2.2.2.1 target
    by_cases current : generation before = target
    · simp only [current,ite_true] at guarded
      rw [guarded]
      exact request_stops before
    · simp only [current,ite_false] at guarded
      simpa only [guarded,Option.getD_none] using prior

/-- A stopped word remains stopped through any number of requests in the
    same active interval, including requests for other generations. -/
theorem stopped_between (writes : List Write) (first last : Fin writes.length)
    (ordered : first.val ≤ last.val) (stopped : Stopped (writes.get first).value)
    (active : ∀ (a b : Fin writes.length), first.val ≤ a.val → a.val + 1 = b.val → b.val ≤ last.val →
      Update (writes.get a).value (writes.get b).value) :
    Stopped (writes.get last).value := by
  have holds : ∀ k (inside : k < writes.length), first.val ≤ k → k ≤ last.val →
      Stopped (writes.get ⟨k,inside⟩).value := by
    intro k
    induction k using Nat.strongRecOn with
    | ind k ih =>
      intro inside lower upper
      by_cases same : k = first.val
      · subst k
        exact stopped
      · have positive : 0 < k := by omega
        let earlier : Fin writes.length := ⟨k-1,by omega⟩
        let current : Fin writes.length := ⟨k,inside⟩
        have previous := ih (k-1) (by omega) earlier.isLt (by omega) (by omega)
        exact (active earlier current (by dsimp [earlier]; omega)
          (by dsimp [earlier,current]; omega) upper).stopped previous
  exact holds last.val last.isLt ordered (Nat.le_refl _)

/-- Two reads in one thread cannot move backwards in this atomic object's
    modification order. Keeping this standard coherence condition explicit
    prevents a proof from treating unrelated Boolean polls as a native trace. -/
def ReadPair (writes : List Write) (early late : Read writes) : Prop :=
  SequencedBefore early.event late.event ∧
    (SequencedBefore early.event late.event → early.source.val ≤ late.source.val)

theorem observed_stays_stopped (writes : List Write) (early late : Read writes)
    (reads : ReadPair writes early late)
    (active : ∀ (a b : Fin writes.length), early.source.val ≤ a.val → a.val + 1 = b.val → b.val ≤ late.source.val →
      Update (writes.get a).value (writes.get b).value)
    (stopped : Stopped (writes.get early.source).value) :
    Stopped (writes.get late.source).value :=
  stopped_between writes early.source late.source (reads.2 reads.1) stopped active

/-- The Boolean returned by `AbortState::is_aborted`: equality of the two
    generation halves. The existing word theorem connects this to `Stopped`. -/
def observed (word : BitVec 64) : Bool := generation word == requested word

theorem observed_iff (word : BitVec 64) : observed word = true ↔ Stopped word := by
  rw [stopped_iff]
  simp [observed]

def Contract : Prop :=
  ∀ (writes : List Write) (early late : Read writes),
    ReadPair writes early late →
    (∀ (a b : Fin writes.length), early.source.val ≤ a.val → a.val + 1 = b.val → b.val ≤ late.source.val →
      Update (writes.get a).value (writes.get b).value) →
    observed (writes.get early.source).value = true →
    observed (writes.get late.source).value = true

theorem contract_proven : Contract := by
  intro writes early late reads active stopped
  exact (observed_iff _).2
    (observed_stays_stopped writes early late reads active ((observed_iff _).1 stopped))

/-- A reset between the two polls invalidates persistence. This is a real
    boundary of the result, even when the 32-bit generation wraps around. -/
theorem begin_resets (word : BitVec 64) : observed (begin word) = false := by
  have fresh := (generationWordContract_proven word).2.2.2.2.2
  cases result : observed (begin word) with
  | false => rfl
  | true => exact (fresh ((observed_iff _).1 result)).elim

end NeuralReversi.Threading.SearchAbort.Persistent

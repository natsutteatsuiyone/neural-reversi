import NeuralReversi.PvsParallel.WorkerFrameMir.Booking

namespace NeuralReversi.PvsParallel.WorkerFrameMir

open Threading
open AtomicWord (Read ReadWindow)
open CutoffVisibility (History HappensBefore)
open SplitCompletion (Worker Action Activation Retirement)

private theorem other_action_preserves (a : Action) (u : Worker) (old : BitVec 64)
    (on : AtomicBitSet.test old u = true) (other : ∀ e, a ≠ .finish u e) :
    AtomicBitSet.test (a.updateBits old) u = true := by
  have update := (AtomicBitSet.maskContract_proven.2 old).2.2.2.2
  cases a with
  | book w e => exact ((update w u).1).trans (by simp [on])
  | finish w e =>
    have different : u ≠ w := fun eq => other e (eq ▸ rfl)
    exact ((update w u).2).trans (by simp [on,different])

private theorem stayed_set (act : Activation ws) (u : Worker) (first : Fin ws.length)
    (inside : act.start.val ≤ first.val)
    (set : AtomicBitSet.test (ws.get first).value u = true)
    (other : ∀ i : Fin act.actions.length, first.val < (act.updateIndex i).val →
      ∀ e, act.actions.get i ≠ .finish u e)
    (n : Nat) (bound : first.val + n ≤ act.start.val + act.actions.length) :
    AtomicBitSet.test (ws.get ⟨first.val + n,by have := act.endInBounds; omega⟩).value u = true := by
  induction n with
  | zero => exact set
  | succ n ih =>
    let i : Fin act.actions.length := ⟨first.val + n - act.start.val,by omega⟩
    have eq : act.start.val + i.val = first.val + n := by dsimp [i]; omega
    have idx : (act.updateIndex i).val = first.val + n + 1 := by simp [Activation.updateIndex,eq]
    have inc : first.val < (act.updateIndex i).val := by omega
    have was := ih (by omega)
    have prev : AtomicBitSet.test (ws.get ⟨act.start.val + i.val,by have := act.endInBounds; omega⟩).value u = true := by
      simpa [eq] using was
    have bit := other_action_preserves (act.actions.get i) u _ prev (other i inc)
    have next := (act.updates i).2.2.2
    rw [← next] at bit
    simpa [eq,Nat.add_assoc] using bit

/-- For a successful helpful-owner mask test followed by the owning idle-loop's empty
    test on that activation, identify a physical reset of exactly the helpful owner.
    Its position is strictly after the set-bit source and no later than the empty source.
    It synchronizes with retirement even when the helpful-owner read was relaxed. -/
theorem helpful_retirement_witness (history : History) (ws : List AtomicWord.Write)
    (act : Activation ws) (empty : Retirement history ws act) (u : Worker) (helpful : Read ws)
    (range : ReadWindow helpful (HappensBefore history) act.start (act.start.val + act.actions.length))
    (accepted : canJoin searching failed true deepest (AtomicBitSet.test (ws.get helpful.source).value u) = true) :
    ∃ i : Fin act.actions.length, ∃ e, act.actions.get i = .finish u e ∧
      helpful.source.val < (act.updateIndex i).val ∧ (act.updateIndex i).val ≤ empty.read.source.val ∧
      HappensBefore history e empty.read.event := by
  have bounds := range.bounds
  have last := (SplitCompletion.emptyReadContract_proven history ws act empty).1
  have granted : AtomicBitSet.test (ws.get helpful.source).value u = true :=
    (Bool.and_eq_true_iff.mp (Bool.and_eq_true_iff.mp accepted).2).2
  by_cases found : ∃ i : Fin act.actions.length, helpful.source.val < (act.updateIndex i).val ∧
      ∃ e, act.actions.get i = .finish u e
  · obtain ⟨i,later,e,eq⟩ := found
    have done := SplitCompletion.handoffContract_proven history ws act empty i (by rw [eq]; rfl)
    rw [eq] at done
    refine ⟨i,e,eq,later,?_,done⟩
    change act.start.val + i.val + 1 ≤ empty.read.source.val
    have := i.isLt; omega
  · have remains := stayed_set act u helpful.source bounds.1 granted
      (fun i later e eq => found ⟨i,later,e,eq⟩) (empty.read.source.val-helpful.source.val) (by omega)
    have idx : helpful.source.val + (empty.read.source.val - helpful.source.val) = empty.read.source.val := by omega
    have still : AtomicBitSet.test (ws.get empty.read.source).value u = true := by simpa [idx] using remains
    have absent := ((AtomicBitSet.maskContract_proven.2 (ws.get empty.read.source).value).2.2.1.mp
      (by rw [empty.zero]; rfl)) u
    rw [still] at absent; contradiction

/-- A prior completion ordered before the booking guard cannot explain retirement.
    A B frame whose owner membership is retained until after this idle-loop test cannot
    explain it either. The remaining input is where each actual B-owner reset occurs
    relative to its A frame on the other worker's call stack. -/
theorem pending_owner_cannot_retire (history : History) (ws : List AtomicWord.Write)
    (act : Activation ws) (empty : Retirement history ws act) (u : Worker) (helpful : Read ws)
    (range : ReadWindow helpful (HappensBefore history) act.start (act.start.val + act.actions.length))
    (accepted : canJoin searching failed true deepest (AtomicBitSet.test (ws.get helpful.source).value u) = true)
    (frame : ∀ i : Fin act.actions.length, ∀ e, act.actions.get i = .finish u e →
      HappensBefore history e helpful.event ∨ HappensBefore history empty.read.event e) : False := by
  obtain ⟨i,e,eq,lt,le,_⟩ := helpful_retirement_witness history ws act empty u helpful range accepted
  have physical : (ws.get (act.updateIndex i)).event = e := (act.updates i).1.trans (by rw [eq]; rfl)
  rcases frame i e eq with before | later
  · have impossible := range.writeRead (act.updateIndex i) (physical ▸ before)
    omega
  · have impossible := empty.readWrite (act.updateIndex i) (physical ▸ later)
    omega

/-- Both normal returns are observed. If the B owner skips A until it sees B empty,
    a prior B participation on the A owner's stack must already have finished before
    the booking guard. A B participation suspended over A cannot retire in time.
    The proof obtains the required A reset and release/acquire order from A's return. -/
theorem completed_crossed_wait (history : History) (aMask bMask : List AtomicWord.Write)
    (a : Activation aMask) (b : Activation bMask) (aEmpty : Retirement history aMask a)
    (bEmpty : Retirement history bMask b) (helpful : Read bMask) (enroll : AtomicWord.Event)
    (posted : .book b.owner enroll ∈ a.actions)
    (range : ReadWindow helpful (HappensBefore history) b.start (b.start.val + b.actions.length))
    (accepted : canJoin searching failed true deepest (AtomicBitSet.test (bMask.get helpful.source).value a.owner) = true)
    (waiting : ∀ e, Action.finish b.owner e ∈ a.actions → AtomicWord.SequencedBefore bEmpty.read.event e)
    (ownerStack : ∀ i : Fin b.actions.length, ∀ e, b.actions.get i = .finish a.owner e →
      HappensBefore history e helpful.event ∨ AtomicWord.SequencedBefore aEmpty.read.event e) : False := by
  obtain ⟨reset,wrote⟩ := (SplitCompletion.emptyReadContract_proven history aMask a aEmpty).2.2 b.owner enroll posted
  obtain ⟨i,found⟩ := List.get_of_mem wrote
  have published := SplitCompletion.handoffContract_proven history aMask a aEmpty i (by rw [found]; rfl)
  rw [found] at published
  have beforeA : HappensBefore history bEmpty.read.event aEmpty.read.event :=
    .trans (.sequenced (waiting reset wrote)) published
  apply pending_owner_cannot_retire history bMask b bEmpty a.owner helpful range accepted
  intro j e eq
  exact (ownerStack j e eq).imp_right (fun after => beforeA.trans (.sequenced after))

end NeuralReversi.PvsParallel.WorkerFrameMir

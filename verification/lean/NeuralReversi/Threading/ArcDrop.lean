module
public import NeuralReversi.Threading.CutoffVisibility

@[expose] public section

namespace NeuralReversi.Threading.ArcDrop

open AtomicWord (Event SequencedBefore Write Read InReleaseSequence)
open CutoffVisibility (History HappensBefore)

inductive StrongUpdate : BitVec 64 → Write → Prop
  | retain : count ≠ 0 → count ≠ -1 →
      StrongUpdate count ⟨event, count + 1, false, true⟩
  | release : count ≠ 0 →
      StrongUpdate count ⟨event, count - 1, true, true⟩

def predecessor (i : Fin n) : Fin n :=
  ⟨i.val - 1, Nat.lt_of_le_of_lt (Nat.sub_le ..) i.isLt⟩

structure StrongHistory (writes : List Write) where
  publication : Fin writes.length
  initial : (writes.get publication).value = 1
  updates : ∀ i : Fin writes.length, publication.val < i.val →
    StrongUpdate (writes.get (predecessor i)).value (writes.get i)

private theorem StrongUpdate.nonzero (change : StrongUpdate count write) : count ≠ 0 := by
  cases change <;> assumption

private theorem StrongUpdate.isRmw (change : StrongUpdate count write) : write.rmw = true := by
  cases change <;> rfl

private theorem StrongUpdate.release_to_zero (change : StrongUpdate 1 write) (release : write.release = true) :
    write.value = 0 := by
  cases change with
  | retain => cases release
  | release => rfl

structure LastRelease {writes : List Write} (strong : StrongHistory writes) where
  index : Fin writes.length
  read : Read writes
  published : strong.publication.val < index.val
  readsPrevious : read.source.val + 1 = index.val
  readsOne : (writes.get read.source).value = 1
  eventEq : read.event = (writes.get index).event
  release : (writes.get index).release = true

theorem LastRelease.zero {strong : StrongHistory writes} (last : LastRelease strong) :
    (writes.get last.index).value = 0 := by
  have same : predecessor last.index = last.read.source := Fin.ext (by
    change last.index.val - 1 = last.read.source.val
    have ordered := last.readsPrevious
    omega)
  have step := strong.updates last.index last.published
  rw [same, last.readsOne] at step
  exact step.release_to_zero last.release

theorem LastRelease.isLast {strong : StrongHistory writes} (last : LastRelease strong) :
    last.index.val + 1 = writes.length := by
  by_cases nextExists : last.index.val + 1 < writes.length
  · let following : Fin writes.length := ⟨last.index.val + 1, nextExists⟩
    have afterPublication : strong.publication.val < following.val :=
      Nat.lt_trans last.published (Nat.lt_succ_self _)
    have pred : predecessor following = last.index := Fin.ext (Nat.add_sub_cancel _ _)
    have nonzero := (strong.updates following afterPublication).nonzero
    rw [pred, last.zero] at nonzero
    exact (nonzero rfl).elim
  · have bound := last.index.isLt
    omega

/-- The standard build uses an acquire fence. ThreadSanitizer builds use
    an acquire load of strong in its place. -/
inductive DestructorAcquire (history : History) {writes : List Write}
    (last : Read writes) : Event → Prop
  | fence (acquire : Event) :
      SequencedBefore last.event acquire → SequencedBefore acquire destroy →
      (∀ head : Fin writes.length, InReleaseSequence writes head last.source →
        history.otherSync (writes.get head).event acquire) →
      DestructorAcquire history last destroy
  | load (acquire : Read writes) :
      acquire.acquire = true →
      SequencedBefore last.event acquire.event → SequencedBefore acquire.event destroy →
      (∀ i : Fin writes.length, HappensBefore history (writes.get i).event acquire.event →
        i.val ≤ acquire.source.val) →
      (∀ head : Fin writes.length, InReleaseSequence writes head acquire.source →
        history.otherSync (writes.get head).event acquire.event) →
      DestructorAcquire history last destroy

private theorem release_sequence (strong : StrongHistory writes)
    (published : strong.publication.val < head.val) (released : (writes.get head).release = true)
    (ordered : head.val ≤ source.val) : InReleaseSequence writes head source :=
  ⟨released, ordered, fun i afterHead _ => (strong.updates i (Nat.lt_trans published afterHead)).isRmw⟩

def ReleaseToDestructorContract : Prop :=
  ∀ (history : History) (writes : List Write) (strong : StrongHistory writes)
    (last : LastRelease strong) (destroy : Event) (head : Fin writes.length),
    strong.publication.val < head.val → (writes.get head).release = true →
    DestructorAcquire history last.read destroy →
    HappensBefore history (writes.get head).event destroy

theorem releaseToDestructorContract_proven : ReleaseToDestructorContract := by
  intro history writes strong last destroy head published released acquire
  cases acquire with
  | fence fence afterLast beforeDestroy synchronize =>
    by_cases same : head = last.index
    · subst head
      have order := HappensBefore.sequenced (history := history) afterLast
      rw [last.eventEq] at order
      exact .trans order (.sequenced beforeDestroy)
    · have earlier := Fin.val_ne_of_ne same
      have bound := head.isLt
      have final := last.isLast
      have previous := last.readsPrevious
      have seq := release_sequence strong published released (source := last.read.source) (by omega)
      exact .trans (.other (synchronize head seq)) (.sequenced beforeDestroy)
  | load read isAcquire afterLast beforeDestroy coherent synchronize =>
    have prior : HappensBefore history (writes.get last.index).event read.event :=
      .sequenced (last.eventEq ▸ afterLast)
    have minimum := coherent last.index prior
    have final := last.isLast
    have headBound := head.isLt
    have seq := release_sequence strong published released (source := read.source) (by omega)
    exact .trans (.other (synchronize head seq)) (.sequenced beforeDestroy)

end NeuralReversi.Threading.ArcDrop

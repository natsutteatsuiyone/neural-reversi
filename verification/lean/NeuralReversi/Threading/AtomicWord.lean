import Lean

namespace NeuralReversi.Threading.AtomicWord

structure Event where
  worker : Nat
  step : Nat
  deriving DecidableEq

def SequencedBefore (a b : Event) : Prop :=
  a.worker = b.worker ∧ a.step < b.step

structure Write where
  event : Event
  value : BitVec 64
  release : Bool
  rmw : Bool

structure Read (writes : List Write) where
  event : Event
  source : Fin writes.length
  acquire : Bool

def signed32At (writes : List Write) (i : Fin writes.length) : Int :=
  ((writes.get i).value.setWidth 32).toInt

/-- The C++20 release sequence consists of the head and contiguous
    read-modify-writes that follow it in this object's modification order. -/
def InReleaseSequence (writes : List Write) (head source : Fin writes.length) : Prop :=
  (writes.get head).release = true ∧ head.val ≤ source.val ∧
    ∀ i : Fin writes.length, head.val < i.val → i.val ≤ source.val →
      (writes.get i).rmw = true

structure ReadWindow {writes : List Write} (load : Read writes)
    (before : Event → Event → Prop) (first : Fin writes.length) (last : Nat) : Prop where
  firstBefore : before (writes.get first).event load.event
  beforeNext : ∀ i : Fin writes.length, i.val = last + 1 → before load.event (writes.get i).event
  writeRead : ∀ i : Fin writes.length, before (writes.get i).event load.event → i.val ≤ load.source.val
  readWrite : ∀ i : Fin writes.length, before load.event (writes.get i).event → load.source.val < i.val

theorem ReadWindow.bounds {writes : List Write} {load : Read writes}
    {before : Event → Event → Prop} {first : Fin writes.length} {last : Nat}
    (window : ReadWindow load before first last) :
    first.val ≤ load.source.val ∧ load.source.val ≤ last := by
  refine ⟨window.writeRead first window.firstBefore, ?_⟩
  by_cases upper : load.source.val ≤ last
  · exact upper
  · let next : Fin writes.length := ⟨last + 1, Nat.lt_of_le_of_lt (by omega) load.source.isLt⟩
    have lower := window.readWrite next (window.beforeNext next rfl)
    change load.source.val < last + 1 at lower
    omega

end NeuralReversi.Threading.AtomicWord

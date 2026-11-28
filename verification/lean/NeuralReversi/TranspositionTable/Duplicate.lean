module
public import NeuralReversi.TranspositionTable.Size

@[expose] public section

namespace NeuralReversi.TranspositionTable.Duplicate

abbrev Bytes := Nat → BitVec 8

def copy (before : Bytes) (source dest bytes : Nat) : Bytes := fun addr =>
  if dest ≤ addr ∧ addr < dest + bytes then before (source + (addr - dest)) else before addr

def slotAddress (base slot offset : Nat) : Nat := base + slot * 32 + offset

def CopyContract : Prop :=
  ∀ bits mib count (before : Bytes) source dest,
    (bits = 32 ∨ bits = 64) → Size.clusterCount bits mib = some count →
    let bytes := count * 2 * 32
    source % 64 = 0 → dest % 64 = 0 →
    source + bytes ≤ 2 ^ bits → dest + bytes ≤ 2 ^ bits →
    (source + bytes ≤ dest ∨ dest + bytes ≤ source) →
    let result := copy before source dest bytes
    bytes = Size.bytesRequested mib ∧ source % 32 = 0 ∧ dest % 32 = 0 ∧
    (∀ i offset, i < count * 2 → offset < 32 →
      slotAddress source i offset < 2 ^ bits ∧ slotAddress dest i offset < 2 ^ bits ∧
      result (slotAddress dest i offset) = before (slotAddress source i offset) ∧
      result (slotAddress source i offset) = before (slotAddress source i offset)) ∧
    (∀ addr, addr < dest ∨ dest + bytes ≤ addr → result addr = before addr)

theorem copyContract_proven : CopyContract := by
  intro bits mib count before source dest width admitted bytes sourceAligned destAligned sourceEnd destEnd apart result
  have bytesExact := ((Size.sizeContract_proven bits mib width).2 count admitted).2.1
  refine ⟨by dsimp [bytes]; omega, by omega, by omega, ?_, ?_⟩
  · intro i offset slotValid offsetValid
    have sourceInside : source ≤ slotAddress source i offset ∧ slotAddress source i offset < source + bytes := by
      dsimp [slotAddress, bytes]; omega
    have destInside : dest ≤ slotAddress dest i offset ∧ slotAddress dest i offset < dest + bytes := by
      dsimp [slotAddress, bytes]; omega
    have sourceOutside : ¬(dest ≤ slotAddress source i offset ∧ slotAddress source i offset < dest + bytes) := by
      rcases apart with prior | later <;> omega
    refine ⟨by omega, by omega, ?_, ?_⟩
    · change copy before source dest bytes (slotAddress dest i offset) = _
      rw [copy, ite_eq_left destInside]
      congr 1
      unfold slotAddress
      omega
    · exact ite_eq_right sourceOutside
  · intro addr outside
    have untouched : ¬(dest ≤ addr ∧ addr < dest + bytes) := by
      rcases outside with low | high <;> omega
    exact ite_eq_right untouched

def OrderingContract : Prop :=
  ∀ {Event : Type} (hb : Event → Event → Prop) (enter exit sample : Event)
    (sourceWrites : List Event),
    (∀ {a b c}, hb a b → hb b c → hb a c) → hb enter sample → hb sample exit →
    (∀ w ∈ sourceWrites, hb w enter ∨ hb exit w) →
    ∀ w ∈ sourceWrites, hb w sample ∨ hb sample w

theorem orderingContract_proven : OrderingContract := by
  intro Event hb enter exit sample sourceWrites trans afterEnter beforeExit order w belongs
  rcases order w belongs with prior | later
  · exact Or.inl (trans prior afterEnter)
  · exact Or.inr (trans beforeExit later)

end NeuralReversi.TranspositionTable.Duplicate

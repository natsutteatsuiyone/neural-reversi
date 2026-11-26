import NeuralReversi.TranspositionTable.Clear
import NeuralReversi.AlignedBuffer.Allocation

namespace NeuralReversi.TranspositionTable.Size

open NeuralReversi.AlignedBuffer (checkedLayout)

def bytesRequested (mib : Nat) : Nat := if mib = 0 then 1024 else mib * 1048576

def clusterCount (bits mib : Nat) : Option Nat :=
  let bytes := bytesRequested mib
  if bytes < 2 ^ bits then
    let count := bytes / 64
    match checkedLayout bits (count * 2) 32 with
    | .allocated _ => some count
    | _ => none
  else none

theorem bytes_divisible (mib : Nat) : bytesRequested mib / 64 * 64 = bytesRequested mib := by
  unfold bytesRequested
  split <;> omega

def SizeContract : Prop :=
  ∀ bits mib, (bits = 32 ∨ bits = 64) →
    (clusterCount bits mib = none ↔ 2 ^ (bits - 1) - 64 < bytesRequested mib) ∧
    (∀ count, clusterCount bits mib = some count →
      0 < count ∧ count * 64 = bytesRequested mib ∧ count * 2 < 2 ^ bits ∧ count * 64 / 1048576 = mib)

theorem sizeContract_proven : SizeContract := by
  intro bits mib width
  let bytes := bytesRequested mib
  let clusters := bytes / 64
  have byteCount : clusters * 2 * 32 = bytes := by
    have whole := bytes_divisible mib
    change clusters * 64 = bytes at whole
    omega
  have positive : 0 < bytes := by
    change 0 < bytesRequested mib
    unfold bytesRequested
    split <;> omega
  have address : 2 ^ (bits - 1) - 64 < 2 ^ bits := by
    rcases width with rfl | rfl <;> decide
  have reported : bytes / 1048576 = mib := by
    change bytesRequested mib / 1048576 = mib
    unfold bytesRequested
    split <;> omega
  have computation : clusterCount bits mib =
      if bytes ≤ 2 ^ (bits - 1) - 64 then some clusters else none := by
    change (if bytes < 2 ^ bits then
        match checkedLayout bits (clusters * 2) 32 with
        | .allocated _ => some clusters
        | _ => none
      else none) = _
    unfold checkedLayout
    rw [byteCount]
    unfold AlignedBuffer.checkedBytes AlignedBuffer.cacheLine
    have nonzero : bytes ≠ 0 := by omega
    by_cases fits : bytes < 2 ^ bits
    · have noOverflow : ¬2 ^ bits ≤ bytes := by omega
      simp only [fits, noOverflow, nonzero, ite_true, ite_false]
      by_cases layout : bytes ≤ 2 ^ (bits - 1) - 64 <;> simp [layout]
    · have invalid : ¬bytes ≤ 2 ^ (bits - 1) - 64 := by omega
      simp [fits, invalid]
  change (clusterCount bits mib = none ↔ _) ∧ _
  rw [computation]
  by_cases layout : bytes ≤ 2 ^ (bits - 1) - 64
  · have fits := Nat.lt_of_le_of_lt layout address
    refine ⟨by simp [layout]; exact layout, ?_⟩
    intro count same
    have eq : clusters = count := by simpa [layout] using same
    subst count
    change 0 < clusters ∧ clusters * 64 = bytes ∧ clusters * 2 < 2 ^ bits ∧
      clusters * 64 / 1048576 = mib
    have exactSize : clusters * 64 = bytes := by omega
    rw [← and_assoc, exactSize, reported]
    omega
  · exact ⟨by simpa [layout] using (Nat.lt_of_not_ge layout),
      fun count same => by simp [layout] at same⟩

def ConstructedSlotContract : Prop :=
  ∀ bits mib count key offset, (bits = 32 ∨ bits = 64) →
    clusterCount bits mib = some count → offset < Clear.clusterSize →
    Clear.getClusterIdx key (BitVec.ofNat 64 count) + offset < count * Clear.clusterSize

theorem constructedSlotContract_proven : ConstructedSlotContract := by
  intro bits mib count key offset width allocated offsetFits
  obtain ⟨positive, _, entriesFit, _⟩ := (sizeContract_proven bits mib width).2 count allocated
  have countFits : count < 2 ^ 64 := by
    rcases width with rfl | rfl <;> omega
  have word : (BitVec.ofNat 64 count).toNat = count := Nat.mod_eq_of_lt countFits
  simpa only [word] using Clear.probe_slot_in_bounds key (BitVec.ofNat 64 count) offset
    (by rwa [word]) offsetFits

end NeuralReversi.TranspositionTable.Size

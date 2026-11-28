module
public import NeuralReversi.TranspositionTable.Size

@[expose] public section

namespace NeuralReversi.TranspositionTable.Sampling

def sampleCount (clusters : Nat) : Nat := min 1024 clusters

def stride (clusters : Nat) : Nat := clusters / sampleCount clusters

def slotIndex (clusters sample offset : Nat) : Nat := sample * stride clusters * 2 + offset

def sampleSlots (clusters : Nat) : List Nat :=
  (List.range (sampleCount clusters)).flatMap fun i => [slotIndex clusters i 0, slotIndex clusters i 1]

def countOccupied (clusters : Nat) (occupied : Nat → Bool) : Nat :=
  (sampleSlots clusters).countP occupied

private theorem stride_positive (clusters : Nat) (positive : 0 < clusters) : 0 < stride clusters := by
  exact Nat.div_pos (Nat.min_le_right _ _) (by unfold sampleCount; omega)

private theorem slot_lt (clusters i slot : Nat) (hi : i < sampleCount clusters) (hs : slot < 2) :
    slotIndex clusters i slot < clusters * 2 := by
  have positive : 0 < clusters := Nat.lt_of_lt_of_le (Nat.zero_lt_of_lt hi) (Nat.min_le_right _ _)
  have step := Nat.mul_lt_mul_of_pos_right hi (stride_positive clusters positive)
  have covered : sampleCount clusters * stride clusters ≤ clusters :=
    Nat.mul_comm _ _ ▸ Nat.div_mul_le_self clusters (sampleCount clusters)
  have within := Nat.lt_of_lt_of_le step covered
  unfold slotIndex
  omega

private theorem two_slots_length (clusters : Nat) (samples : List Nat) :
    (samples.flatMap fun i => [slotIndex clusters i 0, slotIndex clusters i 1]).length = samples.length * 2 := by
  induction samples with
  | nil => rfl
  | cons i rest ih => simp [ih]; omega

def Contract : Prop :=
  (∀ clusters, 0 < clusters → 0 < sampleCount clusters ∧ sampleCount clusters ≤ 1024 ∧ 0 < stride clusters) ∧
  (∀ clusters, (sampleSlots clusters).length = 2 * sampleCount clusters) ∧
  (∀ clusters occupied, countOccupied clusters occupied ≤ 2 * sampleCount clusters) ∧
  (∀ bits mib clusters i slot, (bits = 32 ∨ bits = 64) → Size.clusterCount bits mib = some clusters →
    i < sampleCount clusters → slot < 2 →
    slotIndex clusters i slot < clusters * 2 ∧ slotIndex clusters i slot < 2 ^ bits) ∧
  (∀ clusters, clusters ≤ 1024 → 0 < clusters →
    sampleCount clusters = clusters ∧ ∀ i slot, slotIndex clusters i slot = i * 2 + slot)

theorem contract_proven : Contract := by
  have len (clusters : Nat) : (sampleSlots clusters).length = 2 * sampleCount clusters := by
    simp only [sampleSlots, two_slots_length, List.length_range, Nat.mul_comm]
  refine ⟨?_, len, ?_, ?_, ?_⟩
  · intro clusters positive
    exact ⟨by unfold sampleCount; omega, Nat.min_le_left _ _, stride_positive clusters positive⟩
  · intro clusters occupied
    rw [← len]
    exact List.countP_le_length
  · intro bits mib clusters i offset width admitted hi hs
    have within := slot_lt clusters i offset hi hs
    have allocated := ((Size.sizeContract_proven bits mib width).2 clusters admitted).2.2.1
    exact ⟨within, Nat.lt_trans within allocated⟩
  · intro clusters small positive
    have count : sampleCount clusters = clusters := Nat.min_eq_right small
    exact ⟨count, fun i slot => by simp [slotIndex, stride, count, Nat.div_self positive]⟩

end NeuralReversi.TranspositionTable.Sampling

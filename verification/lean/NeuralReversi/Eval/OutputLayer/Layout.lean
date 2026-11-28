module

@[expose] public section

namespace NeuralReversi.Eval.OutputLayer

def packedLength (capacity : Nat) : Nat :=
  ((capacity + 15) / 16) * 32

def packedAddress (i : Nat) (high : Bool) : Nat :=
  (i / 16) * 32 + (if high then 16 else 0) + i % 16

def PackedLayoutContract : Prop :=
  (∀ capacity i high, i < capacity → packedAddress i high < packedLength capacity) ∧
  (∀ i high, let addr := packedAddress i high
    (addr / 32) * 16 + addr % 16 = i ∧ (16 ≤ addr % 32 ↔ high = true)) ∧
  (∀ i, 16 ∣ i → packedAddress i false = 2 * i ∧ packedAddress i true = 2 * i + 16)

theorem packedLayoutContract_proven : PackedLayoutContract := by
  refine ⟨?_, ?_, ?_⟩
  · intro capacity i high h
    cases high <;> simp [packedAddress, packedLength] <;> omega
  · intro i high
    dsimp
    cases high <;> simp [packedAddress] <;> omega
  · intro i h
    obtain ⟨k, hk⟩ := h
    rw [hk]
    simp [packedAddress]
    omega

def StreamLayoutContract : Prop :=
  ∀ capacity,
    4 ≤ 4 + 2 * capacity ∧
    (∀ i, i < capacity → 4 + 2 * i + 2 ≤ 4 + 2 * capacity) ∧
    (∀ i j, i < j → 4 + 2 * i + 2 ≤ 4 + 2 * j)

theorem streamLayoutContract_proven : StreamLayoutContract := by
  intro capacity
  exact ⟨by omega, fun _ _ => by omega, fun _ _ _ => by omega⟩

def SegmentOffsetsContract : Prop :=
  ∀ first second inputs capacity : Nat,
    first + second = inputs → inputs ≤ capacity →
    (∀ i, i < first → i < capacity) ∧
    (∀ j, j < second → first + j < capacity) ∧
    first ≤ capacity ∧ first + second ≤ capacity

theorem segmentOffsetsContract_proven : SegmentOffsetsContract := by
  intro first second inputs capacity h₁ h₂
  exact ⟨fun _ _ => by omega, fun _ _ => by omega, by omega, by omega⟩

def SegmentWindowsContract : Prop :=
  ∀ width segmentLen offset capacity chunk count,
    0 < width → offset + segmentLen ≤ capacity → chunk + count ≤ segmentLen / width →
    width * (chunk + count) ≤ segmentLen ∧
    offset + width * (chunk + count) ≤ capacity ∧
    2 * (offset + width * (chunk + count)) ≤ packedLength capacity

theorem segmentWindowsContract_proven : SegmentWindowsContract := by
  intro width segmentLen offset capacity chunk count positive fits h
  have endIn : width * (chunk + count) ≤ segmentLen := by
    rw [Nat.mul_comm]
    exact (Nat.le_div_iff_mul_le positive).mp h
  unfold packedLength
  exact ⟨endIn, by omega, by omega⟩

def ChunkCoverageContract : Prop :=
  ∀ width len, 0 < width → width ∣ len →
    width * (len / width) = len ∧
    (∀ i, i < len → i / width < len / width ∧ i % width < width ∧
      width * (i / width) + i % width = i)

theorem chunkCoverageContract_proven : ChunkCoverageContract := by
  intro width len positive multiple
  have cover : width * (len / width) = len := by
    obtain ⟨n, h⟩ := multiple
    rw [h, Nat.mul_div_cancel_left n positive]
  refine ⟨cover, fun i hi => ⟨?_, Nat.mod_lt i positive, Nat.div_add_mod i width⟩⟩
  exact (Nat.div_lt_iff_lt_mul positive).mpr (by rwa [Nat.mul_comm, cover])

def AlignedWindowsContract : Prop :=
  ∀ width byteBase weightBase offset chunk,
    width ∣ byteBase → (2 * width) ∣ weightBase → width ∣ offset →
    width ∣ byteBase + width * chunk ∧
    (2 * width) ∣ weightBase + 2 * (offset + width * chunk)

theorem alignedWindowsContract_proven : AlignedWindowsContract := by
  intro width byteBase weightBase offset chunk inputAligned weightAligned offsetAligned
  obtain ⟨skip, h⟩ := offsetAligned
  refine ⟨Nat.dvd_add inputAligned ⟨chunk, rfl⟩, Nat.dvd_add weightAligned ?_⟩
  refine ⟨skip + chunk, ?_⟩
  rw [h, ← Nat.mul_add, Nat.mul_assoc]

def wholeChunks (width first second : Nat) : Bool :=
  decide (first % width = 0) && decide (second % width = 0)

def DispatchContract : Prop :=
  ∀ width first second,
    wholeChunks width first second = true ↔ width ∣ first ∧ width ∣ second

theorem dispatchContract_proven : DispatchContract := by
  intro width first second
  simp [wholeChunks, Nat.dvd_iff_mod_eq_zero]

def NetworkSegmentsContract : Prop :=
  (64 + 256 = 320) ∧
  (∀ width, width = 16 ∨ width = 32 →
    width ∣ 64 ∧ width ∣ 256 ∧ 2 * width ∣ 64) ∧
  (∀ base : Nat, 64 ∣ base → 16 ∣ base ∧ 32 ∣ base)

theorem networkSegmentsContract_proven : NetworkSegmentsContract := by
  refine ⟨rfl, ?_, ?_⟩
  · rintro width (rfl | rfl) <;> decide
  · intro base aligned
    exact ⟨Nat.dvd_trans (by decide : 16 ∣ 64) aligned,
      Nat.dvd_trans (by decide : 32 ∣ 64) aligned⟩

end NeuralReversi.Eval.OutputLayer

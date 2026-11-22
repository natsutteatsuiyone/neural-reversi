import NeuralReversi.Eval.InputLayer.Layout
import NeuralReversi.AlignedBuffer.Allocation

namespace NeuralReversi.Eval.MainScratch

def alignOffset (base : Nat) : Nat := (64 - base % 64) % 64

def ScratchWindowContract : Prop :=
  ∀ base size field fieldBytes, field + fieldBytes ≤ size → 64 ∣ field →
    let offset := alignOffset base
    offset < 64 ∧ 64 ∣ base + offset ∧ offset + size ≤ size + 64 ∧
      64 ∣ base + offset + field ∧ offset + field + fieldBytes ≤ size + 64 ∧
      (∀ skip, 64 ∣ base + skip → offset ≤ skip)

theorem scratchWindowContract_proven : ScratchWindowContract := by
  intro base size field fieldBytes inBounds aligned
  have off : alignOffset base < 64 := Nat.mod_lt _ (by decide)
  have atStart : 64 ∣ base + alignOffset base := by
    rw [Nat.dvd_iff_mod_eq_zero]
    unfold alignOffset
    omega
  refine ⟨off, atStart, by omega, Nat.dvd_add atStart aligned, by omega, ?_⟩
  intro skip skipAligned
  rw [Nat.dvd_iff_mod_eq_zero] at skipAligned
  unfold alignOffset
  omega

def BufferDimensionsContract : Prop :=
  let paddedInputs := 32 * ((128 + 128 + 1 + 31) / 32)
  let allocations := [AlignedBuffer.roundToLine paddedInputs, AlignedBuffer.roundToLine (32 * 4),
    AlignedBuffer.roundToLine 32, AlignedBuffer.roundToLine (64 * 4),
    AlignedBuffer.roundToLine 64]
  paddedInputs = 288 ∧ allocations = [320, 128, 64, 256, 64] ∧ allocations.sum = 832 ∧
    64 ∣ (64 : Nat) ∧ 64 ∣ (256 : Nat)

theorem bufferDimensionsContract_proven : BufferDimensionsContract := by
  unfold BufferDimensionsContract
  decide

abbrev Byte := BitVec 8
abbrev Inputs := Fin 288 → Byte
abbrev Transform := Fin 128 → Byte

def prepare (old : Inputs) (base phase : Transform) (mobility : Nat) : Inputs :=
  fun i =>
    if b : i.val < 128 then base ⟨i.val, b⟩
    else if p : i.val < 256 then phase ⟨i.val - 128, by omega⟩
    else if i.val = 256 then BitVec.ofNat 8 (7 * mobility)
    else old i

def PaddingZero (input : Inputs) : Prop :=
  ∀ i, 257 ≤ i.val → input i = 0

def PrepareContract : Prop :=
  ∀ old base phase mobility,
    let prepared := prepare old base phase mobility
    (∀ i : Fin 128, prepared ⟨i.val, by omega⟩ = base i) ∧
    (∀ i : Fin 128, prepared ⟨128 + i.val, by omega⟩ = phase i) ∧
    (mobility ≤ 35 → (prepared ⟨256, by decide⟩).toNat = 7 * mobility) ∧
    (∀ i, 257 ≤ i.val → prepared i = old i)

theorem prepareContract_proven : PrepareContract := by
  intro old base phase mobility
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro i
    simp [prepare, i.isLt]
  · intro i
    have h : 128 + i.val < 256 := by omega
    have sub : 128 + i.val - 128 = i.val := by omega
    simp [prepare, h, sub, show ¬ 128 + i.val < 128 by omega]
  · intro bounded
    simp [prepare, Nat.mod_eq_of_lt (show 7 * mobility < 256 by omega)]
  · intro i pad
    simp [prepare, show ¬ i.val < 128 by omega, show ¬ i.val < 256 by omega,
      show i.val ≠ 256 by omega]

def Request := Transform × Transform × Nat

def prepareMany : List Request → Inputs → Inputs
  | [], old => old
  | (base, phase, mobility) :: rest, old => prepareMany rest (prepare old base phase mobility)

def ReuseContract : Prop :=
  ∀ requests old, PaddingZero old →
    PaddingZero (prepareMany requests old) ∧
    ∀ base phase mobility,
      prepare (prepareMany requests old) base phase mobility =
        prepare (fun _ => 0) base phase mobility

theorem reuseContract_proven : ReuseContract := by
  intro requests old zero
  have keep : PaddingZero (prepareMany requests old) := by
    induction requests generalizing old with
    | nil => exact zero
    | cons request rest ih =>
      obtain ⟨base, phase, mobility⟩ := request
      apply ih
      intro i pad
      rw [(prepareContract_proven old base phase mobility).2.2.2 i pad]
      exact zero i pad
  refine ⟨keep, fun base phase mobility => ?_⟩
  funext i
  by_cases h : 257 ≤ i.val
  · rw [(prepareContract_proven _ base phase mobility).2.2.2 i h, keep i h,
      (prepareContract_proven _ base phase mobility).2.2.2 i h]
  · unfold prepare
    have cases : i.val < 256 ∨ i.val = 256 := by omega
    rcases cases with below | atMobility
    · simp [below]
    · simp [atMobility]

def bypass (input : Inputs) (i : Fin 256) : Byte := input ⟨i.val, by omega⟩

def BypassContract : Prop :=
  ∀ old base phase mobility (i : Fin 128),
    bypass (prepare old base phase mobility) ⟨i.val, by omega⟩ = base i ∧
    bypass (prepare old base phase mobility) ⟨128 + i.val, by omega⟩ = phase i

theorem bypassContract_proven : BypassContract := by
  intro old base phase mobility i
  exact ⟨(prepareContract_proven old base phase mobility).1 i,
    (prepareContract_proven old base phase mobility).2.1 i⟩

end NeuralReversi.Eval.MainScratch

import Lean

namespace NeuralReversi.Eval.LinearLayer

def rustWeightIndex (i inputs outputs : Nat) : Nat :=
  (i / 4) % (inputs / 4) * (outputs * 4) + i / inputs * 4 + i % 4

def packedSlot (outputs chunk row lane : Nat) : Nat :=
  (chunk * outputs + row) * 4 + lane

def diskSlot (chunks row chunk lane : Nat) : Nat :=
  (row * chunks + chunk) * 4 + lane

def WeightLayoutContract : Prop :=
  ∀ chunks outputs row chunk lane,
    chunk < chunks → lane < 4 →
    rustWeightIndex (diskSlot chunks row chunk lane) (4 * chunks) outputs =
      packedSlot outputs chunk row lane

theorem weightLayoutContract_proven : WeightLayoutContract := by
  intro chunks outputs row chunk lane chunkOk laneOk
  have word : diskSlot chunks row chunk lane / 4 = row * chunks + chunk := by
    unfold diskSlot
    omega
  have byte : diskSlot chunks row chunk lane % 4 = lane := by
    unfold diskSlot
    omega
  have h : 0 < chunks := by omega
  have chunkIndex : (row * chunks + chunk) % chunks = chunk := by
    rw [Nat.mul_comm row chunks, Nat.mul_add_mod, Nat.mod_eq_of_lt chunkOk]
  have rowIndex : (row * chunks + chunk) / chunks = row := by
    rw [Nat.mul_comm row chunks, Nat.mul_add_div h, Nat.div_eq_of_lt chunkOk, Nat.add_zero]
  unfold rustWeightIndex packedSlot
  rw [← Nat.div_div_eq_div_mul, word, byte, rowIndex,
    Nat.mul_div_cancel_left _ (by decide : 0 < 4), chunkIndex, Nat.add_mul, Nat.mul_assoc]

private theorem slotBlock (outputs chunk row : Nat) (h : row < outputs) :
    chunk * outputs + row < (chunk + 1) * outputs := by
  rw [Nat.add_mul, Nat.one_mul]
  omega

def PackedSlotsContract : Prop :=
  (∀ chunks outputs chunk row lane,
    chunk < chunks → row < outputs → lane < 4 →
    packedSlot outputs chunk row lane < 4 * chunks * outputs) ∧
  (∀ outputs chunk row lane,
    row < outputs → lane < 4 →
    let slot := packedSlot outputs chunk row lane
    slot / 4 / outputs = chunk ∧ slot / 4 % outputs = row ∧ slot % 4 = lane)

theorem packedSlotsContract_proven : PackedSlotsContract := by
  constructor
  · intro chunks outputs chunk row lane chunkOk rowOk laneOk
    have bound := Nat.lt_of_lt_of_le (slotBlock outputs chunk row rowOk)
      (Nat.mul_le_mul_right outputs (show chunk + 1 ≤ chunks by omega))
    unfold packedSlot
    rw [Nat.mul_comm 4 chunks, Nat.mul_assoc chunks 4 outputs, Nat.mul_comm 4 outputs,
      ← Nat.mul_assoc]
    omega
  · intro outputs chunk row lane rowOk laneOk
    have h : 0 < outputs := by omega
    have word : packedSlot outputs chunk row lane / 4 = chunk * outputs + row := by
      unfold packedSlot
      omega
    change packedSlot outputs chunk row lane / 4 / outputs = chunk ∧
      packedSlot outputs chunk row lane / 4 % outputs = row ∧
      packedSlot outputs chunk row lane % 4 = lane
    rw [word, Nat.mul_comm chunk outputs, Nat.mul_add_mod, Nat.mod_eq_of_lt rowOk,
      Nat.mul_add_div h, Nat.div_eq_of_lt rowOk, Nat.add_zero]
    unfold packedSlot
    omega

def paddedChunks (inputs : Nat) : Nat := ((inputs + 7) / 8) * 2

def streamChunks (inputs : Nat) : Nat := (inputs + 3) / 4

def ChunkCountsContract : Prop :=
  ∀ inputs paddedInputs,
    8 * ((inputs + 7) / 8) ≤ paddedInputs →
    inputs ≤ 4 * streamChunks inputs ∧
    streamChunks inputs ≤ paddedChunks inputs ∧
    4 * paddedChunks inputs ≤ paddedInputs ∧
    (∀ chunk, chunk < paddedChunks inputs → 4 * chunk + 4 ≤ paddedInputs) ∧
    (∀ chunk, chunk + 4 ≤ paddedChunks inputs → 4 * chunk + 16 ≤ paddedInputs)

theorem chunkCountsContract_proven : ChunkCountsContract := by
  intro inputs paddedInputs fits
  unfold streamChunks paddedChunks
  exact ⟨by omega, by omega, by omega, fun _ _ => by omega, fun _ _ => by omega⟩

def VectorWindowsContract : Prop :=
  ∀ chunks width regs chunk reg base,
    0 < width → chunk < chunks → reg < regs → base % (width * 4) = 0 →
    let outputs := width * regs
    let offset := chunk * (outputs * 4) + reg * (width * 4)
    offset + width * 4 ≤ (4 * chunks) * outputs ∧
    (base + offset) % (width * 4) = 0 ∧
    reg * width + width ≤ outputs

theorem vectorWindowsContract_proven : VectorWindowsContract := by
  intro chunks width regs chunk reg base _ chunkOk regOk aligned
  have regEnd : reg * width + width ≤ width * regs := by
    rw [← Nat.add_one_mul, Nat.mul_comm width regs]
    exact Nat.mul_le_mul_right width (by omega)
  have stride : (width * regs) * 4 = regs * (width * 4) := by
    rw [Nat.mul_comm width regs, Nat.mul_assoc]
  have endOfChunk : chunk * (width * regs) + reg * width + width ≤ (chunk + 1) * (width * regs) := by
    rw [Nat.add_one_mul, Nat.add_assoc]
    exact Nat.add_le_add_left regEnd _
  have finalEnd := Nat.le_trans endOfChunk
    (Nat.mul_le_mul_right (width * regs) (show chunk + 1 ≤ chunks by omega))
  change chunk * ((width * regs) * 4) + reg * (width * 4) + width * 4 ≤
      (4 * chunks) * (width * regs) ∧
    (base + (chunk * ((width * regs) * 4) + reg * (width * 4))) % (width * 4) = 0 ∧
    reg * width + width ≤ width * regs
  refine ⟨?_, ?_, regEnd⟩
  · have scaled := Nat.mul_le_mul_right 4 finalEnd
    simp only [Nat.add_mul, Nat.mul_assoc] at scaled ⊢
    rw [← Nat.mul_assoc 4 chunks, Nat.mul_comm 4 chunks, Nat.mul_assoc chunks,
      Nat.mul_comm 4 (width * regs), Nat.mul_assoc width regs 4]
    exact scaled
  · rw [stride, ← Nat.mul_assoc, ← Nat.add_mul, Nat.add_mul_mod_self_right, aligned]

def i8mmMaxRegs : Nat := 16

def useI8mm (neonShape cpuI8mm : Bool) (outputs : Nat) : Bool :=
  neonShape && decide (outputs / 4 ≤ i8mmMaxRegs) && cpuI8mm

def I8mmBoundsContract : Prop :=
  ∀ neonShape cpuI8mm outputs,
    useI8mm neonShape cpuI8mm outputs = true →
    outputs / 4 ≤ i8mmMaxRegs ∧
    (∀ reg, reg < outputs / 4 → reg < i8mmMaxRegs) ∧
    (outputs / 4 ≤ 4 → ∀ reg lane, reg < outputs / 4 → lane < 4 →
      reg * 4 + lane < i8mmMaxRegs)

theorem i8mmBoundsContract_proven : I8mmBoundsContract := by
  intro neonShape cpuI8mm outputs enabled
  simp [useI8mm] at enabled
  have cap := enabled.1.2
  unfold i8mmMaxRegs at *
  exact ⟨cap, fun _ _ => by omega, fun _ _ _ _ _ => by omega⟩

end NeuralReversi.Eval.LinearLayer

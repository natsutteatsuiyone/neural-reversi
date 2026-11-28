module
public import NeuralReversi.Solve1.LineTable

@[expose] public section

namespace NeuralReversi.Solve1.LineSafety
open LineTable

def above (pos : Fin 8) (line : Byte) : Usize := line.setWidth 64 >>> (pos.val + 1)
def below (pos : Fin 8) (line : Byte) : Usize :=
  line.setWidth 64 &&& (((1 : Usize) <<< pos.val) - 1)
def logIndex (pos : Fin 8) (line : Byte) : Usize := 63 - (below pos line).clz

def upperCount (pos : Fin 8) (line : Byte) : Nat :=
  if above pos line = 0 then 0 else 2 * (above pos line).ctz.toNat

def lowerCount (pos : Fin 8) (line : Byte) : Nat :=
  if below pos line = 0 then 0 else 2 * (pos.val - 1 - (logIndex pos line).toNat)

def Safe (pos : Fin 8) (line : Byte) : Prop :=
  (above pos line ≠ 0 →
    (above pos line).ctz.toNat ≤ 6 ∧
    ((above pos line).ctz.setWidth 8).toNat = (above pos line).ctz.toNat) ∧
  (below pos line ≠ 0 →
    1 ≤ pos.val ∧ (below pos line).clz.toNat ≤ 63 ∧
    (logIndex pos line).toNat < pos.val ∧
    BitVec.usubOverflow (BitVec.ofNat 64 pos.val) 1 = false ∧
    BitVec.usubOverflow 63 (below pos line).clz = false ∧
    BitVec.usubOverflow (BitVec.ofNat 64 pos.val - 1) (logIndex pos line) = false ∧
    ((BitVec.ofNat 64 pos.val - 1 - logIndex pos line).setWidth 8).toNat =
      pos.val - 1 - (logIndex pos line).toNat) ∧
  upperCount pos line + lowerCount pos line ≤ 12 ∧
  (countFlip pos line).toNat = upperCount pos line + lowerCount pos line

private def aboveByte (pos : Fin 8) (line : Byte) : Byte := line >>> (pos.val + 1)
private def belowByte (pos : Fin 8) (line : Byte) : Byte := line &&& (((1 : Byte) <<< pos.val) - 1)

private theorem fast_safe_all : ∀ pos : Fin 8, ∀ n : Fin 256,
    let line := BitVec.ofNat 8 n.val
    let a := aboveByte pos line
    let b := belowByte pos line
    (a ≠ 0 → fastCtz8 a ≤ 6) ∧
    (b ≠ 0 → 1 ≤ pos.val ∧ fastClz8 b ≤ 7 ∧ 7 - fastClz8 b < pos.val) ∧
    (if a = 0 then 0 else 2 * fastCtz8 a) +
      (if b = 0 then 0 else 2 * (pos.val - 1 - (7 - fastClz8 b))) =
      2 * reference pos line := by decide +kernel

theorem all_safe (pos : Fin 8) (line : Byte) : Safe pos line := by
  have hf := fast_safe_all pos ⟨line.toNat, line.isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hf
  obtain ⟨ha_bnd, hb_bnd, hsum_fast⟩ := hf
  let a := aboveByte pos line
  let b := belowByte pos line
  have ha_eq : above pos line = a.setWidth 64 := above_eq_setWidth pos line
  have hb_eq : below pos line = b.setWidth 64 := below_eq_setWidth pos line
  have ha_zero : above pos line = 0 ↔ a = 0 := by
    rw [ha_eq]; exact setWidth64_eq_zero_iff a
  have hb_zero : below pos line = 0 ↔ b = 0 := by
    rw [hb_eq]; exact setWidth64_eq_zero_iff b
  have ha_ctz : a ≠ 0 → (above pos line).ctz.toNat = fastCtz8 a := by
    intro h; rw [ha_eq, setWidth64_ctz_toNat a h]
  have hb_clz_nat : b ≠ 0 → (below pos line).clz.toNat = 56 + fastClz8 b := by
    intro h; rw [hb_eq, setWidth64_clz_toNat b h]
  have hlog_nat : b ≠ 0 → (logIndex pos line).toNat = 7 - fastClz8 b := by
    intro h
    have hbb : 1 ≤ pos.val ∧ fastClz8 b ≤ 7 ∧ 7 - fastClz8 b < pos.val := hb_bnd h
    have ⟨_, hclz, _⟩ := hbb
    have hclz64 := hb_clz_nat h
    unfold logIndex
    rw [BitVec.toNat_sub, hclz64]
    have h63 : (63 : Usize).toNat = 63 := rfl
    rw [h63]
    omega
  have hupper : upperCount pos line = if a = 0 then 0 else 2 * fastCtz8 a := by
    unfold upperCount
    by_cases h : a = 0
    · rw [ite_eq_left (ha_zero.mpr h), ite_eq_left h]
    · rw [ite_eq_right (fun h0 => h (ha_zero.mp h0)), ite_eq_right h, ha_ctz h]
  have hlower : lowerCount pos line = if b = 0 then 0 else 2 * (pos.val - 1 - (7 - fastClz8 b)) := by
    unfold lowerCount
    by_cases h : b = 0
    · rw [ite_eq_left (hb_zero.mpr h), ite_eq_left h]
    · rw [ite_eq_right (fun h0 => h (hb_zero.mp h0)), ite_eq_right h, hlog_nat h]
  have hsum : upperCount pos line + lowerCount pos line = 2 * reference pos line := by
    rw [hupper, hlower]; exact hsum_fast
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro hne
    have ha : a ≠ 0 := fun h => hne (ha_zero.mpr h)
    have hctz := ha_ctz ha
    have hle : fastCtz8 a ≤ 6 := ha_bnd ha
    refine ⟨by omega, ?_⟩
    rw [BitVec.toNat_setWidth, hctz]
    omega
  · intro hne
    have hb : b ≠ 0 := fun h => hne (hb_zero.mpr h)
    have hbb : 1 ≤ pos.val ∧ fastClz8 b ≤ 7 ∧ 7 - fastClz8 b < pos.val := hb_bnd hb
    have ⟨hpos, hclz, hlt⟩ := hbb
    have hclz64 := hb_clz_nat hb
    have hlog := hlog_nat hb
    have hp8 := pos.isLt
    have h1u : (1 : Usize).toNat = 1 := rfl
    have h63u : (63 : Usize).toNat = 63 := rfl
    have hsub1 : (BitVec.ofNat 64 pos.val - 1).toNat = pos.val - 1 := by
      rw [BitVec.toNat_sub, BitVec.toNat_ofNat, h1u]
      omega
    refine ⟨hpos, by omega, by omega, ?_, ?_, ?_, ?_⟩
    · simp only [BitVec.usubOverflow, BitVec.toNat_ofNat, h1u, decide_eq_false_iff_not]
      omega
    · simp only [BitVec.usubOverflow, h63u, hclz64, decide_eq_false_iff_not]
      omega
    · simp only [BitVec.usubOverflow, hsub1, hlog, decide_eq_false_iff_not]
      omega
    · rw [BitVec.toNat_setWidth, BitVec.toNat_sub, hsub1, hlog]
      omega
  · rw [hsum]
    have := reference_bound pos line
    omega
  · rw [countFlip_correct, hsum]

end NeuralReversi.Solve1.LineSafety

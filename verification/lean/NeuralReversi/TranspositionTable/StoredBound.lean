module
public import NeuralReversi.Aspiration
public import NeuralReversi.TranspositionTable.Packing

@[expose] public section

namespace NeuralReversi.TranspositionTable.StoredBound

open Packing

theorem narrow_score (s : Score) (lo : -32767 < s.raw.toInt) (hi : s.raw.toInt < 32767) :
    (s.raw.setWidth 16).toInt = s.raw.toInt := by
  calc
    _ = s.raw.toInt.bmod (2^16) := by
      rw [BitVec.toInt_setWidth, BitVec.toInt_eq_toNat_bmod,
        Int.bmod_bmod_of_dvd (by decide : 2^16 ∣ 2^32)]
    _ = _ := Int.bmod_eq_of_le (by omega) (by omega)

def Contract : Prop :=
  ∀ pv best alpha beta value, Aspiration.Window alpha.raw.toInt beta.raw.toInt →
    Aspiration.SoundResult value alpha.raw.toInt beta.raw.toInt best.raw.toInt →
    ∀ move depth sel gen endgame ttPv bigEndian,
      let entry := newData best move (classify pv best alpha beta) depth sel gen endgame ttPv
      let decoded := unpack bigEndian (pack bigEndian entry)
      decoded.score.toInt = best.raw.toInt ∧ BoundSound decoded value ∧
        ∀ checked threshold requestedDepth requestedSel requireEndgame,
          canCut checked decoded threshold requestedDepth requestedSel requireEndgame = .ok true →
          (threshold.raw.toInt ≤ best.raw.toInt ↔ threshold.raw.toInt ≤ value) ∧
          (threshold.raw.toInt ≤ best.raw.toInt → best.raw.toInt ≤ value) ∧
          (best.raw.toInt < threshold.raw.toInt → value ≤ best.raw.toInt)

theorem contract_proven : Contract := by
  intro pv best alpha beta value _window sound move depth sel gen endgame ttPv bigEndian
  simp only [unpack_pack]
  let entry := newData best move (classify pv best alpha beta) depth sel gen endgame ttPv
  have score : entry.score.toInt = best.raw.toInt := narrow_score best sound.lower sound.upper
  have valid : BoundSound entry value := by
    simp only [BoundSound, score]
    change ((boundByte (classify pv best alpha beta)).getLsbD 0 = true → _) ∧
      ((boundByte (classify pv best alpha beta)).getLsbD 1 = true → _)
    obtain ⟨lower, precise, upper, nonempty⟩ := classify_contract pv best alpha beta
    cases tag : classify pv best alpha beta with
    | none => exact (nonempty tag).elim
    | lower => simpa [boundByte] using sound.failHigh (lower.mp tag)
    | exact =>
      obtain ⟨_, left, right⟩ := precise.mp tag
      simp [boundByte, sound.exact left right]
    | upper =>
      obtain ⟨belowBeta, _⟩ := upper.mp tag
      by_cases low : best.raw.toInt ≤ alpha.raw.toInt
      · simpa [boundByte] using sound.failLow low
      · simp [boundByte, sound.exact (by omega) belowBeta]
  refine ⟨score, valid, ?_⟩
  intro checked threshold qDepth qSel qEnd cut
  have side := (canCut_sound checked entry threshold qDepth qSel qEnd value valid cut).1
  obtain lower | upper := ((canCut_true_iff checked entry threshold qDepth qSel qEnd).mp cut).2.2
  · exact ⟨by rwa [score] at side, fun _ => score ▸ valid.1 lower.2, by rw [← score]; omega⟩
  · exact ⟨by rwa [score] at side, by rw [← score]; omega, fun _ => score ▸ valid.2 upper.2⟩

end NeuralReversi.TranspositionTable.StoredBound

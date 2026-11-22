import NeuralReversi.Flip.LrmaskModel
import NeuralReversi.Flip.Spec

namespace NeuralReversi.Flip
open NeuralReversi.Bitboard.Moves (OnRay)

set_option maxRecDepth 8192
set_option maxHeartbeats 12000000

def FastCell (sq : Square) (d : Fin 8) (i : Nat) : Prop :=
  sq.val < 64 ∧ i < 64 ∧
  match d.val with
  | 0 => i / 8 = sq.val / 8 ∧ sq.val % 8 < i % 8
  | 1 => i % 8 = sq.val % 8 ∧ sq.val / 8 < i / 8
  | 2 => sq.val / 8 < i / 8 ∧ sq.val % 8 < i % 8 ∧
      i / 8 - sq.val / 8 = i % 8 - sq.val % 8
  | 3 => sq.val / 8 < i / 8 ∧ i % 8 < sq.val % 8 ∧
      i / 8 - sq.val / 8 = sq.val % 8 - i % 8
  | 4 => i / 8 = sq.val / 8 ∧ i % 8 < sq.val % 8
  | 5 => i % 8 = sq.val % 8 ∧ i / 8 < sq.val / 8
  | 6 => i / 8 < sq.val / 8 ∧ i % 8 < sq.val % 8 ∧
      sq.val / 8 - i / 8 = sq.val % 8 - i % 8
  | _ => i / 8 < sq.val / 8 ∧ sq.val % 8 < i % 8 ∧
      sq.val / 8 - i / 8 = i % 8 - sq.val % 8

instance (sq : Square) (d : Fin 8) (i : Nat) : Decidable (FastCell sq d i) := by
  unfold FastCell
  split <;> infer_instance

theorem cell_fast (sq : Square) (d : Fin 8) (i : Nat) :
    Cell sq d i ↔ FastCell sq d i := by
  by_cases hs : sq.val < 64
  · by_cases hi : i < 64
    · simp only [Cell, FastCell, hs, hi, true_and]
      have hd : d.val = 0 ∨ d.val = 1 ∨ d.val = 2 ∨ d.val = 3 ∨
          d.val = 4 ∨ d.val = 5 ∨ d.val = 6 ∨ d.val = 7 := by omega
      rcases hd with hd | hd | hd | hd | hd | hd | hd | hd
      all_goals simp only [delta, hd, OnRay, Int.mul_zero, Int.mul_one,
        Int.mul_neg, Int.add_zero, Int.reduceNeg]
      · constructor
        · rintro ⟨⟨k, hk⟩, hp, hr, hc⟩; omega
        · rintro ⟨hr, hc⟩
          refine ⟨⟨i % 8 - sq.val % 8, by omega⟩, ?_, ?_, ?_⟩ <;> (try dsimp) <;> omega
      · constructor
        · rintro ⟨⟨k, hk⟩, hp, hr, hc⟩; omega
        · rintro ⟨hc, hr⟩
          refine ⟨⟨i / 8 - sq.val / 8, by omega⟩, ?_, ?_, ?_⟩ <;> (try dsimp) <;> omega
      · constructor
        · rintro ⟨⟨k, hk⟩, hp, hr, hc⟩; omega
        · rintro ⟨hr, hc, he⟩
          refine ⟨⟨i / 8 - sq.val / 8, by omega⟩, ?_, ?_, ?_⟩ <;> (try dsimp) <;> omega
      · constructor
        · rintro ⟨⟨k, hk⟩, hp, hr, hc⟩; omega
        · rintro ⟨hr, hc, he⟩
          refine ⟨⟨i / 8 - sq.val / 8, by omega⟩, ?_, ?_, ?_⟩ <;> (try dsimp) <;> omega
      · constructor
        · rintro ⟨⟨k, hk⟩, hp, hr, hc⟩; omega
        · rintro ⟨hr, hc⟩
          refine ⟨⟨sq.val % 8 - i % 8, by omega⟩, ?_, ?_, ?_⟩ <;> (try dsimp) <;> omega
      · constructor
        · rintro ⟨⟨k, hk⟩, hp, hr, hc⟩; omega
        · rintro ⟨hc, hr⟩
          refine ⟨⟨sq.val / 8 - i / 8, by omega⟩, ?_, ?_, ?_⟩ <;> (try dsimp) <;> omega
      · constructor
        · rintro ⟨⟨k, hk⟩, hp, hr, hc⟩; omega
        · rintro ⟨hr, hc, he⟩
          refine ⟨⟨sq.val / 8 - i / 8, by omega⟩, ?_, ?_, ?_⟩ <;> (try dsimp) <;> omega
      · constructor
        · rintro ⟨⟨k, hk⟩, hp, hr, hc⟩; omega
        · rintro ⟨hr, hc, he⟩
          refine ⟨⟨sq.val / 8 - i / 8, by omega⟩, ?_, ?_, ?_⟩ <;> (try dsimp) <;> omega
    · simp [Cell, FastCell, hi]
  · simp [Cell, FastCell, hs]

theorem fast_coefficients : ∀ sq : Square, ∀ d : Fin 8, ∀ i : Fin 64,
    (mask sq d).getLsbD i.val = decide (FastCell sq d i.val) := by decide

theorem mask_coefficients : ∀ sq : Square, ∀ d : Fin 8, ∀ i : Fin 64,
    (mask sq d).getLsbD i.val = decide (Cell sq d i.val) := by
  intro sq d i
  rw [fast_coefficients]
  apply Bool.eq_iff_iff.mpr
  simp only [decide_eq_true_eq]
  exact (cell_fast sq d i.val).symm

theorem mask_bit (sq : Square) (d : Fin 8) (i : Nat) :
    (mask sq d).getLsbD i = true ↔ Cell sq d i := by
  by_cases hi : i < 64
  · rw [mask_coefficients sq d ⟨i, hi⟩]
    simp only [decide_eq_true_eq]
  · rw [BitVec.getLsbD_of_ge _ _ (by omega)]
    simp [Cell, hi]


end NeuralReversi.Flip

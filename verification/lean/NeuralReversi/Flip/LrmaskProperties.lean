module
public import NeuralReversi.Flip.LrmaskModel
public import NeuralReversi.Flip.Spec

@[expose] public section

namespace NeuralReversi.Flip
open NeuralReversi.Bitboard.Moves (OnRay)

set_option maxRecDepth 8192

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

private def rayList (sq : Square) (d : Fin 8) : List Nat :=
  let (dr, dc) := delta d
  (List.range 7).filterMap fun k =>
    let r : Int := (sq.val / 8 : Nat) + dr * (k + 1 : Nat)
    let c : Int := (sq.val % 8 : Nat) + dc * (k + 1 : Nat)
    if 0 ≤ r ∧ r < 8 ∧ 0 ≤ c ∧ c < 8 then some (r.toNat * 8 + c.toNat) else none

private def rayMask (xs : List Nat) : Word :=
  xs.foldr (fun k acc => (1#64 <<< k) ||| acc) 0#64

private theorem rayMask_bit (xs : List Nat) (i : Nat) (hi : i < 64) :
    (rayMask xs).getLsbD i = decide (i ∈ xs) := by
  induction xs with
  | nil => simp [rayMask]
  | cons k rest ih =>
    have h1 : ((1#64 <<< k).getLsbD i) = decide (i = k) := by
      simp only [BitVec.getLsbD_shiftLeft, hi, decide_true, Bool.true_and]
      by_cases heq : i = k
      · subst heq; simp
      · by_cases hlt : i < k
        · simp [hlt, heq]
        · have hne : i - k ≠ 0 := by omega
          simp [hlt, heq, BitVec.getLsbD_one, hne]
    change ((1#64 <<< k) ||| rayMask rest).getLsbD i = decide (i ∈ k :: rest)
    rw [BitVec.getLsbD_or, h1, ih]
    by_cases heq : i = k <;> simp [heq]

private theorem mem_rayList_iff (sq : Square) (d : Fin 8) (i : Nat) :
    i ∈ rayList sq d ↔ Cell sq d i := by
  simp only [rayList, List.mem_filterMap, List.mem_range, Cell, OnRay]
  constructor
  · rintro ⟨k, hk, hsome⟩
    split at hsome
    · rename_i hb
      have hi_eq := Option.some.inj hsome
      have hsq : sq.val < 64 := sq.isLt
      refine ⟨hsq, by omega, ⟨k + 1, by omega⟩, by dsimp; omega, ?_, ?_⟩
      · dsimp only
        rw [Int.mul_comm ((k + 1 : Nat) : Int)]
        omega
      · dsimp only
        rw [Int.mul_comm ((k + 1 : Nat) : Int)]
        omega
    · cases hsome
  · rintro ⟨_, hi, ⟨k, hk⟩, hpos, hr, hc⟩
    dsimp only at hpos hr hc
    refine ⟨k - 1, by omega, ?_⟩
    have hsq : sq.val < 64 := sq.isLt
    have hk1 : ((k - 1 + 1 : Nat) : Int) = (k : Int) := by omega
    have hreq : ((sq.val / 8 : Nat) : Int) + (delta d).1 * ((k - 1 + 1 : Nat) : Int) = (i / 8 : Nat) := by
      rw [hk1, Int.mul_comm]; omega
    have hceq : ((sq.val % 8 : Nat) : Int) + (delta d).2 * ((k - 1 + 1 : Nat) : Int) = (i % 8 : Nat) := by
      rw [hk1, Int.mul_comm]; omega
    simp only [hreq, hceq]
    rw [ite_eq_left (by omega)]
    congr 1; omega

private theorem mask_eq_rayMask : ∀ sq : Square, ∀ d : Fin 8,
    mask sq d = rayMask (rayList sq d) := by decide +kernel

theorem mask_coefficients : ∀ sq : Square, ∀ d : Fin 8, ∀ i : Fin 64,
    (mask sq d).getLsbD i.val = decide (Cell sq d i.val) := by
  intro sq d i
  rw [mask_eq_rayMask, rayMask_bit _ _ i.isLt]
  exact Bool.eq_iff_iff.mpr (by simp only [decide_eq_true_eq]; exact mem_rayList_iff sq d i.val)

theorem fast_coefficients : ∀ sq : Square, ∀ d : Fin 8, ∀ i : Fin 64,
    (mask sq d).getLsbD i.val = decide (FastCell sq d i.val) := by
  intro sq d i
  rw [mask_coefficients]
  exact Bool.eq_iff_iff.mpr (by simp only [decide_eq_true_eq]; exact cell_fast sq d i.val)

theorem mask_bit (sq : Square) (d : Fin 8) (i : Nat) :
    (mask sq d).getLsbD i = true ↔ Cell sq d i := by
  by_cases hi : i < 64
  · rw [mask_coefficients sq d ⟨i, hi⟩]
    simp only [decide_eq_true_eq]
  · rw [BitVec.getLsbD_of_ge _ _ (by omega)]
    simp [Cell, hi]


end NeuralReversi.Flip

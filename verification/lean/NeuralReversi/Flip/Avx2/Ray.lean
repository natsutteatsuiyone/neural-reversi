import NeuralReversi.Flip.Avx2.Spec
import NeuralReversi.Flip.Avx2.Word
import NeuralReversi.Flip.LrmaskProperties
import NeuralReversi.Flip.Geometry
import NeuralReversi.Flip.Directional

namespace NeuralReversi.Flip.Avx2
open NeuralReversi.Flip
open NeuralReversi.Flip.Scalar (Greatest)
open NeuralReversi.Bitboard.Moves (OnRay)

set_option maxRecDepth 8192
set_option maxHeartbeats 2000000

def rightStep (d : Fin 8) : Nat :=
  match d.val with
  | 4 => 1 | 5 => 8 | 6 => 9 | _ => 7

theorem right_ray_gap (sq i j : Nat) (a b : Fin 8) (d : Fin 8) (hd : 4 ≤ d.val)
    (hi : OnRay sq i a.val (delta d).1 (delta d).2)
    (hj : OnRay sq j b.val (delta d).1 (delta d).2) (hij : i ≤ j) :
    i + (a.val - b.val) * rightStep d = j := by
  have hcases : d.val = 4 ∨ d.val = 5 ∨ d.val = 6 ∨ d.val = 7 := by omega
  rcases hcases with h | h | h | h
  all_goals simp only [OnRay, delta, rightStep, h, Int.mul_zero, Int.mul_one,
    Int.mul_neg, Int.add_zero, Int.reduceNeg] at hi hj ⊢
  all_goals omega

theorem shadow_on_ray (sq : Square) (d : Fin 8) (hd : 4 ≤ d.val)
    (player : Word) (i : Nat) (hi : (mask sq d).getLsbD i = true) :
    (shadowFill (player &&& mask sq d) (rightStep d) 3).getLsbD i = true ↔
      ∃ j, (mask sq d).getLsbD j = true ∧ player.getLsbD j = true ∧ i ≤ j := by
  rw [shadow8_bit]
  constructor
  · rintro ⟨k, _, hb⟩
    have hbits : player.getLsbD (i + k * rightStep d) = true ∧
        (mask sq d).getLsbD (i + k * rightStep d) = true := by
      simpa only [BitVec.getLsbD_and, Bool.and_eq_true] using hb
    exact ⟨i + k * rightStep d, hbits.2, hbits.1, by omega⟩
  · rintro ⟨j, hj, hp, hij⟩
    obtain ⟨_, _, a, _, hra⟩ := (mask_bit sq d i).mp hi
    obtain ⟨_, _, b, _, hrb⟩ := (mask_bit sq d j).mp hj
    refine ⟨a.val - b.val, by have := a.isLt; omega, ?_⟩
    rw [right_ray_gap sq.val i j a b d hd hra hrb hij]
    simp only [BitVec.getLsbD_and, hp, hj, Bool.and_self]

theorem right_mask_msb (sq : Square) (d : Fin 8) (hd : 4 ≤ d.val) :
    (mask sq d).msb = false := by
  apply Bool.eq_false_iff.mpr
  intro hm
  have hb : (mask sq d).getLsbD 63 = true := by
    simpa only [BitVec.msb_eq_getLsbD_last] using hm
  obtain ⟨hs, _, k, hk, hr⟩ := (mask_bit sq d 63).mp hb
  have self : OnRay sq.val sq.val 0 (delta d).1 (delta d).2 := by simp [OnRay]
  have order := (onRay_order sq.val sq.val 63 0 k.val d self hr).mpr hk
  simp only [Before, Nat.not_lt_of_ge hd, ↓reduceIte] at order
  omega

theorem prefix_coordinate (sq : Square) (d : Fin 8) (opponent : Word) (i : Nat) :
    Prefix (mask sq d) opponent i ↔ CoordinatePrefix sq d opponent i := by
  constructor
  · rintro ⟨hm, hbelow⟩
    refine ⟨(mask_bit sq d i).mp hm, ?_⟩
    intro k hk hki
    exact opponent_true_of_nonOpponent_mask_false (mask sq d) opponent k
      ((mask_bit sq d k).mpr hk) (hbelow k hki)
  · rintro ⟨hi, hbelow⟩
    refine ⟨(mask_bit sq d i).mpr hi, ?_⟩
    intro k hki
    by_cases hk : (mask sq d).getLsbD k = true
    · have ho := hbelow k ((mask_bit sq d k).mp hk) hki
      simp only [BitVec.getLsbD_and, BitVec.getLsbD_not, ho, Bool.not_true,
        Bool.and_false, Bool.false_and]
    · have hz := Bool.eq_false_iff.mpr hk
      simp only [BitVec.getLsbD_and, hz, Bool.and_false]

theorem left_coordinate (sq : Square) (d : Fin 8) (player opponent : Word) (i : Nat) :
    (leftWord (mask sq d) player opponent).getLsbD i = true ↔
      LeftRawBit sq d player opponent i := by
  simp only [left_raw, prefix_coordinate, LeftRawBit]

theorem greatest_anchor_coordinate (sq : Square) (d : Fin 8) (player opponent : Word) :
    (∃ j, Greatest (~~~opponent &&& mask sq d) j ∧ player.getLsbD j = true) ↔
      RightAnchor sq d player opponent := by
  constructor
  · rintro ⟨j, ⟨hj, hhigh⟩, hp⟩
    obtain ⟨hm, ho⟩ := (nonOpponent_mask_true (mask sq d) opponent j).mp hj
    refine ⟨j, (mask_bit sq d j).mp hm, hp, ho, ?_⟩
    intro k hk hjk
    exact opponent_true_of_nonOpponent_mask_false (mask sq d) opponent k
      ((mask_bit sq d k).mpr hk) (hhigh k hjk)
  · rintro ⟨j, hj, hp, ho, hhigh⟩
    refine ⟨j, ⟨(nonOpponent_mask_true (mask sq d) opponent j).mpr
      ⟨(mask_bit sq d j).mpr hj, ho⟩, ?_⟩, hp⟩
    intro k hjk
    by_cases hk : (mask sq d).getLsbD k = true
    · have hok := hhigh k ((mask_bit sq d k).mp hk) hjk
      simp only [BitVec.getLsbD_and, BitVec.getLsbD_not, hok, Bool.not_true,
        Bool.and_false, Bool.false_and]
    · have hz := Bool.eq_false_iff.mpr hk
      simp only [BitVec.getLsbD_and, hz, Bool.and_false]

theorem right_comparison (sq : Square) (d : Fin 8) (hd : 4 ≤ d.val)
    (player opponent : Word) :
    ((~~~opponent &&& mask sq d) ^^^ (player &&& mask sq d)).toInt <
      (player &&& mask sq d).toInt ↔ RightAnchor sq d player opponent := by
  have hm := right_mask_msb sq d hd
  have hn : (~~~opponent &&& mask sq d).msb = false := by simp [hm]
  have hp : (player &&& mask sq d).msb = false := by simp [hm]
  rw [xor_signed_comparison _ _ hn hp]
  rw [← greatest_anchor_coordinate]
  apply exists_congr
  intro j
  constructor
  · rintro ⟨hj, hb⟩
    have hb' : player.getLsbD j = true ∧ (mask sq d).getLsbD j = true := by
      simpa only [BitVec.getLsbD_and, Bool.and_eq_true] using hb
    exact ⟨hj, hb'.1⟩
  · rintro ⟨hj, hpj⟩
    have hmj := ((nonOpponent_mask_true (mask sq d) opponent j).mp hj.1).1
    exact ⟨hj, by simp only [BitVec.getLsbD_and, hpj, hmj, Bool.and_self]⟩

def rightWord (m player opponent : Word) (step : Nat) : Word :=
  let rp := player &&& m
  let rs := shadowFill rp step 3
  let re := (~~~opponent &&& m) ^^^ rp
  if re.toInt < rp.toInt then m &&& ~~~rs else 0

theorem right_coordinate (sq : Square) (d : Fin 8) (hd : 4 ≤ d.val)
    (player opponent : Word) (i : Nat) :
    (rightWord (mask sq d) player opponent (rightStep d)).getLsbD i = true ↔
      RightRawBit sq d player opponent i := by
  by_cases hg : ((~~~opponent &&& mask sq d) ^^^ (player &&& mask sq d)).toInt <
      (player &&& mask sq d).toInt
  · have anchor := (right_comparison sq d hd player opponent).mp hg
    simp only [rightWord, hg, ↓reduceIte, BitVec.getLsbD_and, BitVec.getLsbD_not,
      Bool.and_eq_true, Bool.not_eq_true', decide_eq_true_eq]
    constructor
    · rintro ⟨hm, _, hn⟩
      refine ⟨anchor, (mask_bit sq d i).mp hm, ?_⟩
      rintro ⟨j, hj, hp, hij⟩
      have hfilled := (shadow_on_ray sq d hd player i hm).mpr
        ⟨j, (mask_bit sq d j).mpr hj, hp, hij⟩
      simp only [hn, Bool.false_eq_true] at hfilled
    · rintro ⟨_, hi, hnone⟩
      have hm := (mask_bit sq d i).mpr hi
      refine ⟨hm, BitVec.lt_of_getLsbD hm, ?_⟩
      apply Bool.eq_false_iff.mpr
      intro hfilled
      obtain ⟨j, hj, hp, hij⟩ := (shadow_on_ray sq d hd player i hm).mp hfilled
      exact hnone ⟨j, (mask_bit sq d j).mp hj, hp, hij⟩
  · have hno : ¬ RightAnchor sq d player opponent :=
      fun h => hg ((right_comparison sq d hd player opponent).mpr h)
    simp only [rightWord, hg, ↓reduceIte, RightRawBit, hno, false_and]
    simp


end NeuralReversi.Flip.Avx2

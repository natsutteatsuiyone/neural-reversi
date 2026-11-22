import NeuralReversi.Flip.Geometry
import NeuralReversi.Flip.ScalarProperties

namespace NeuralReversi.Flip
open Scalar
open NeuralReversi.Bitboard.Moves (OnRay)

def Edge (x : Word) (d : Fin 8) (j : Nat) : Prop :=
  x.getLsbD j = true ∧ ∀ k, Before d k j → x.getLsbD k = false

def abstractLane (m : Word) (d : Fin 8) (player opponent : Word) : Word :=
  if d.val < 4 then flipLeft m player (~~~opponent)
  else flipRightValue m player (~~~opponent)

theorem abstract_lane_bits (m : Word) (d : Fin 8) (player opponent : Word) (i : Nat) :
    (abstractLane m d player opponent).getLsbD i = true ↔
      ∃ j, Edge (~~~opponent &&& m) d j ∧ player.getLsbD j = true ∧
        m.getLsbD i = true ∧ Before d i j := by
  by_cases hd : d.val < 4
  · simpa only [abstractLane, hd, ↓reduceIte, Edge, Before, Least] using
      left_contract m player (~~~opponent) i
  · simpa only [abstractLane, hd, ↓reduceIte, Edge, Before, Greatest] using
      right_contract m player (~~~opponent) i

theorem nonOpponent_mask_true (m opponent : Word) (i : Nat) :
    (~~~opponent &&& m).getLsbD i = true ↔
      m.getLsbD i = true ∧ opponent.getLsbD i = false := by
  by_cases hi : i < 64
  · simp only [BitVec.getLsbD_and, BitVec.getLsbD_not, hi, decide_true,
      Bool.true_and, Bool.and_eq_true, Bool.not_eq_true']
    exact and_comm
  · have hm := BitVec.getLsbD_of_ge m i (by omega)
    simp [BitVec.getLsbD_and, hm]

theorem opponent_true_of_nonOpponent_mask_false (m opponent : Word) (i : Nat)
    (hm : m.getLsbD i = true) (hn : (~~~opponent &&& m).getLsbD i = false) :
    opponent.getLsbD i = true := by
  have hi := BitVec.lt_of_getLsbD hm
  simp only [BitVec.getLsbD_and, BitVec.getLsbD_not, hi, decide_true,
    Bool.true_and, hm, Bool.and_true] at hn
  cases ho : opponent.getLsbD i <;> simp_all

theorem abstract_lane_coordinate (sq : Square) (d : Fin 8) (m player opponent : Word)
    (hm : RayMask sq d m) (i : Nat) :
    (abstractLane m d player opponent).getLsbD i = true ↔
      DirectionalBit sq d player opponent i := by
  rw [abstract_lane_bits]
  constructor
  · rintro ⟨anchor, ⟨han, hbefore⟩, hp, him, hib⟩
    obtain ⟨ham, hao⟩ := (nonOpponent_mask_true m opponent anchor).mp han
    obtain ⟨hs, ha, distance, hdpos, hda⟩ := (hm anchor).mp ham
    obtain ⟨_, hi, k, hkpos, hki⟩ := (hm i).mp him
    have hkd := (onRay_order sq.val i anchor k.val distance.val d hki hda).mp hib
    refine ⟨hs, hi, anchor, distance.val, ha, by omega, by have := distance.isLt; omega,
      hp, hao, hda, ?_, k.val, hkpos, hkd, hki⟩
    intro t htpos htd
    obtain ⟨index, hindex, hr⟩ := onRay_prefix_exists sq.val anchor distance.val t
      (delta d).1 (delta d).2 hs ha (delta_range d).1 (delta_range d).2 (by omega) hda
    have htm : m.getLsbD index = true := (hm index).mpr
      ⟨hs, hindex, ⟨t, by have := distance.isLt; omega⟩, htpos, hr⟩
    have htb := (onRay_order sq.val index anchor t distance.val d hr hda).mpr htd
    exact ⟨index, hindex, hr,
      opponent_true_of_nonOpponent_mask_false m opponent index htm (hbefore index htb)⟩
  · rintro ⟨hs, hi, anchor, distance, ha, hdlo, hdhi, hp, hao, hda, hchain,
      k, hkpos, hkd, hki⟩
    have ham : m.getLsbD anchor = true := (hm anchor).mpr
      ⟨hs, ha, ⟨distance, by omega⟩, (show 0 < distance from by omega), hda⟩
    have him : m.getLsbD i = true := (hm i).mpr
      ⟨hs, hi, ⟨k, by omega⟩, hkpos, hki⟩
    refine ⟨anchor, ⟨(nonOpponent_mask_true m opponent anchor).mpr ⟨ham, hao⟩, ?_⟩,
      hp, him, (onRay_order sq.val i anchor k distance d hki hda).mpr hkd⟩
    intro index hib
    by_cases himask : m.getLsbD index = true
    · obtain ⟨_, _, t, htpos, hti⟩ := (hm index).mp himask
      have htd := (onRay_order sq.val index anchor t.val distance d hti hda).mp hib
      obtain ⟨middle, _, hmid, homid⟩ := hchain t.val htpos htd
      have heq := onRay_unique sq.val index middle t.val (delta d).1 (delta d).2 hti hmid
      have ho : opponent.getLsbD index = true := heq ▸ homid
      simp only [BitVec.getLsbD_and, BitVec.getLsbD_not, ho, Bool.not_true,
        Bool.and_false, Bool.false_and]
    · have hz : m.getLsbD index = false := Bool.eq_false_iff.mpr himask
      simp only [BitVec.getLsbD_and, hz, Bool.and_false]

end NeuralReversi.Flip

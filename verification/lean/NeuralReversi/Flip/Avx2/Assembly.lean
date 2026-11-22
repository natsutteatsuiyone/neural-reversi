import NeuralReversi.Flip.Avx2.Spec
import NeuralReversi.Flip.Avx2.Ray
import NeuralReversi.Flip.Properties

namespace NeuralReversi.Flip.Avx2
open NeuralReversi.Flip

theorem disjoint_player_notOpponent (player opponent : Word) (h : player &&& opponent = 0)
    (j : Nat) (hp : player.getLsbD j = true) : opponent.getLsbD j = false := by
  have hb := congrArg (fun w : Word => w.getLsbD j) h
  simpa [BitVec.getLsbD_and, hp] using hb

theorem disjoint_opponent_notPlayer (player opponent : Word) (h : player &&& opponent = 0)
    (j : Nat) (ho : opponent.getLsbD j = true) : player.getLsbD j = false := by
  exact disjoint_player_notOpponent opponent player (by simpa [BitVec.and_comm] using h) j ho

theorem left_disjoint (m player opponent : Word) (hdis : player &&& opponent = 0) (i : Nat) :
    (leftWord m player opponent).getLsbD i = true ↔
      (Scalar.flipLeft m player (~~~opponent)).getLsbD i = true := by
  rw [left_raw, Scalar.left_contract]
  constructor
  · rintro ⟨⟨hmi, hbelowi⟩, hpi, j, ⟨hmj, hbelowj⟩, hpj⟩
    have hoj := disjoint_player_notOpponent player opponent hdis j hpj
    have hxj := (nonOpponent_mask_true m opponent j).mpr ⟨hmj, hoj⟩
    have hij : i < j := by
      by_cases hji : j < i
      · have hz := hbelowi j hji
        simp only [hxj, Bool.true_eq_false] at hz
      · by_cases heq : i = j
        · subst i
          simp only [hpj, Bool.true_eq_false] at hpi
        · omega
    exact ⟨j, ⟨hxj, hbelowj⟩, hpj, hmi, hij⟩
  · rintro ⟨j, ⟨hxj, hbelowj⟩, hpj, hmi, hij⟩
    have hmj := ((nonOpponent_mask_true m opponent j).mp hxj).1
    have hoi := opponent_true_of_nonOpponent_mask_false m opponent i hmi (hbelowj i hij)
    exact ⟨⟨hmi, fun k hk => hbelowj k (by omega)⟩,
      disjoint_opponent_notPlayer player opponent hdis i hoi, j, ⟨hmj, hbelowj⟩, hpj⟩

theorem right_disjoint (sq : Square) (d : Fin 8) (hd : 4 ≤ d.val)
    (player opponent : Word) (hdis : player &&& opponent = 0) (i : Nat) :
    (rightWord (mask sq d) player opponent (rightStep d)).getLsbD i = true ↔
      (Scalar.flipRightValue (mask sq d) player (~~~opponent)).getLsbD i = true := by
  rw [right_coordinate sq d hd, Scalar.right_contract]
  constructor
  · rintro ⟨anchor, hi, hnone⟩
    obtain ⟨j, hj, hp⟩ := (greatest_anchor_coordinate sq d player opponent).mpr anchor
    have hmj := ((nonOpponent_mask_true (mask sq d) opponent j).mp hj.1).1
    have hji : j < i := by
      apply Classical.byContradiction
      intro hn
      exact hnone ⟨j, (mask_bit sq d j).mp hmj, hp, by omega⟩
    exact ⟨j, hj, hp, (mask_bit sq d i).mpr hi, hji⟩
  · rintro ⟨j, hj, hpj, hmi, hji⟩
    refine ⟨(greatest_anchor_coordinate sq d player opponent).mp ⟨j, hj, hpj⟩,
      (mask_bit sq d i).mp hmi, ?_⟩
    rintro ⟨k, hk, hpk, hik⟩
    have hok := disjoint_player_notOpponent player opponent hdis k hpk
    have hxk := (nonOpponent_mask_true (mask sq d) opponent k).mpr
      ⟨(mask_bit sq d k).mpr hk, hok⟩
    have hz := hj.2 k (by omega)
    simp only [hxk, Bool.true_eq_false] at hz

def lane (sq : Square) (d : Fin 8) (player opponent : Word) : Word :=
  if d.val < 4 then leftWord (mask sq d) player opponent
  else rightWord (mask sq d) player opponent (rightStep d)



theorem lane_raw (sq : Square) (d : Fin 8) (player opponent : Word) (i : Nat) :
    (lane sq d player opponent).getLsbD i = true ↔ LaneSpec sq d player opponent i := by
  by_cases hd : d.val < 4
  · simp only [lane, LaneSpec, hd, ↓reduceIte]
    exact left_coordinate sq d player opponent i
  · simp only [lane, LaneSpec, hd, ↓reduceIte]
    exact right_coordinate sq d (by omega) player opponent i

theorem lane_disjoint (sq : Square) (d : Fin 8) (player opponent : Word)
    (hdis : player &&& opponent = 0) (i : Nat) :
    (lane sq d player opponent).getLsbD i = true ↔
      (NeuralReversi.Flip.laneValue sq d player opponent).getLsbD i = true := by
  by_cases hd : d.val < 4
  · simp only [lane, NeuralReversi.Flip.laneValue, hd, ↓reduceIte]
    exact left_disjoint (mask sq d) player opponent hdis i
  · simp only [lane, NeuralReversi.Flip.laneValue, hd, ↓reduceIte]
    exact right_disjoint sq d (by omega) player opponent hdis i

def summary (sq : Square) (player opponent : Word) : Word :=
  lane sq 0 player opponent ||| lane sq 1 player opponent |||
    lane sq 2 player opponent ||| lane sq 3 player opponent |||
    lane sq 4 player opponent ||| lane sq 5 player opponent |||
    lane sq 6 player opponent ||| lane sq 7 player opponent

theorem summary_bit (sq : Square) (player opponent : Word) (i : Nat) :
    (summary sq player opponent).getLsbD i = true ↔
      ∃ d : Fin 8, (lane sq d player opponent).getLsbD i = true := by
  rw [NeuralReversi.Flip.exists_fin8]
  simp only [summary, BitVec.getLsbD_or, Bool.or_eq_true, or_assoc]

theorem summary_raw (sq : Square) (player opponent : Word) (i : Nat) :
    (summary sq player opponent).getLsbD i = true ↔ RawSpec sq player opponent i := by
  rw [summary_bit]
  exact exists_congr fun d => lane_raw sq d player opponent i

theorem disjoint_coordinate (sq : Square) (player opponent : Word)
    (hdis : player &&& opponent = 0) (i : Nat) :
    (summary sq player opponent).getLsbD i = true ↔ CoordinateBit sq player opponent i := by
  rw [summary_bit]
  apply exists_congr
  intro d
  exact (lane_disjoint sq d player opponent hdis i).trans
    (NeuralReversi.Flip.lane_coordinate sq d player opponent i)

theorem disjoint_scalar_equality (sq : Square) (player opponent : Word)
    (hdis : player &&& opponent = 0) :
    summary sq player opponent = NeuralReversi.Flip.value sq player opponent := by
  apply BitVec.eq_of_getLsbD_eq
  intro i _
  apply Bool.eq_iff_iff.mpr
  exact (disjoint_coordinate sq player opponent hdis i).trans
    (NeuralReversi.Flip.value_coordinate sq player opponent i).symm

end NeuralReversi.Flip.Avx2

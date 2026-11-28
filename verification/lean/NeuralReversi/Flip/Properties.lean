module
public import NeuralReversi.Flip.Model
public import NeuralReversi.Flip.Directional
public import NeuralReversi.Flip.LrmaskProperties

@[expose] public section

namespace NeuralReversi.Flip
open Scalar
open NeuralReversi.Bitboard.Moves (OnRay)

theorem operational_eq_value (checks : Bool) (sq : Square) (player opponent : Word) :
    flip checks sq player opponent = .ok (value sq player opponent) := by
  simp only [flip, flipRight_eq_value]
  rfl

def laneValue (sq : Square) (d : Fin 8) (player opponent : Word) : Word :=
  if d.val < 4 then flipLeft (mask sq d) player (~~~opponent)
  else flipRightValue (mask sq d) player (~~~opponent)

theorem exists_fin8 (P : Fin 8 → Prop) :
    (∃ d, P d) ↔ P 0 ∨ P 1 ∨ P 2 ∨ P 3 ∨ P 4 ∨ P 5 ∨ P 6 ∨ P 7 := by
  simp only [Fin.exists_fin_succ, Fin.exists_fin_zero, or_false]
  rfl

theorem value_bit (sq : Square) (player opponent : Word) (i : Nat) :
    (value sq player opponent).getLsbD i = true ↔
      ∃ d : Fin 8, (laneValue sq d player opponent).getLsbD i = true := by
  rw [exists_fin8]
  simp [value, laneValue, BitVec.getLsbD_or, Bool.or_eq_true, or_assoc]
  rfl

theorem table_rayMask (sq : Square) (d : Fin 8) :
    RayMask sq d (mask sq d) := by
  intro i
  change (mask sq d).getLsbD i = true ↔ Cell sq d i
  exact mask_bit sq d i

theorem lane_coordinate (sq : Square) (d : Fin 8) (player opponent : Word) (i : Nat) :
    (laneValue sq d player opponent).getLsbD i = true ↔
      DirectionalBit sq d player opponent i := by
  exact abstract_lane_coordinate sq d (mask sq d) player opponent
    (table_rayMask sq d) i

theorem value_coordinate (sq : Square) (player opponent : Word) (i : Nat) :
    (value sq player opponent).getLsbD i = true ↔ CoordinateBit sq player opponent i := by
  rw [value_bit]
  exact exists_congr fun d => lane_coordinate sq d player opponent i

theorem full_contract (checks : Bool) (sq : Square) (player opponent : Word) :
    ∃ bits, flip checks sq player opponent = .ok bits ∧
      ∀ i, bits.getLsbD i = true ↔ CoordinateBit sq player opponent i := by
  exact ⟨value sq player opponent, operational_eq_value checks sq player opponent,
    value_coordinate sq player opponent⟩

theorem flipped_is_opponent (sq : Square) (player opponent : Word) (i : Nat)
    (h : (value sq player opponent).getLsbD i = true) : opponent.getLsbD i = true := by
  obtain ⟨d, _, _, anchor, distance, _, _, _, _, _, _, chain, k, hk, hkd, ray⟩ :=
    (value_coordinate sq player opponent i).mp h
  obtain ⟨middle, _, hm, ho⟩ := chain k hk hkd
  have heq := onRay_unique sq.val i middle k
    (delta d).1 (delta d).2 ray hm
  exact heq ▸ ho

def ScalarFlipContract : Prop :=
  ∀ checks sq player opponent, ∃ bits,
    flip checks sq player opponent = .ok bits ∧
    ∀ i, bits.getLsbD i = true ↔ CoordinateBit sq player opponent i

theorem scalarFlipContract_proven : ScalarFlipContract := full_contract

end NeuralReversi.Flip

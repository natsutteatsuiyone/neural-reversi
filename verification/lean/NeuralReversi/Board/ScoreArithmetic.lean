module
public import NeuralReversi.Board.ScoreModel

@[expose] public section

namespace NeuralReversi.Board

theorem score_small_toInt (value : Score) (bound : value.toNat ≤ 64) :
    value.toInt = (value.toNat : Int) := by
  apply BitVec.toInt_eq_toNat_of_lt
  omega

theorem score_word_value (player : Score) (bound : player.toNat ≤ 64) :
    (player * 2 - 64).toInt = 2 * (player.toNat : Int) - 64 := by
  rw [BitVec.toInt_sub, BitVec.toInt_mul, score_small_toInt player bound]
  change (((player.toNat : Int) * 2).bmod 4294967296 - 64).bmod 4294967296 = _
  rw [Int.bmod_eq_of_le (n := (player.toNat : Int) * 2) (by omega) (by omega)]
  rw [Int.bmod_eq_of_le (by omega) (by omega)]
  omega

theorem score_bounded_add (score empties : Score) (lower : -64 ≤ score.toInt)
    (upper : score.toInt ≤ 128) (bound : empties.toNat ≤ 64) :
    (score + empties).toInt = score.toInt + (empties.toNat : Int) := by
  rw [BitVec.toInt_add, score_small_toInt empties bound]
  exact Int.bmod_eq_of_le (by omega) (by omega)

theorem score_bounded_add_safe (score empties : Score) (lower : -64 ≤ score.toInt)
    (upper : score.toInt ≤ 128) (bound : empties.toNat ≤ 64) :
    BitVec.saddOverflow score empties = false := by
  simp only [BitVec.saddOverflow, score_small_toInt empties bound]
  have below : ¬ score.toInt + (empties.toNat : Int) < -2147483648 := by omega
  have above : ¬ score.toInt + (empties.toNat : Int) ≥ 2147483648 := by omega
  simpa using And.intro above below

theorem score_ops_safe (player : Score) (bound : player.toNat ≤ 64) :
    BitVec.smulOverflow player 2 = false ∧
    BitVec.ssubOverflow (player * 2) 64 = false := by
  have playerInt := score_small_toInt player bound
  have product : (player * 2).toInt = (player.toNat : Int) * 2 := by
    rw [BitVec.toInt_mul, playerInt]
    change ((player.toNat : Int) * 2).bmod 4294967296 = _
    exact Int.bmod_eq_of_le (by omega) (by omega)
  constructor
  · simp only [BitVec.smulOverflow, playerInt]
    change (decide ((player.toNat : Int) * 2 ≥ 2147483648) ||
      decide ((player.toNat : Int) * 2 < -2147483648)) = false
    simp only [Bool.or_eq_false_iff, decide_eq_false_iff_not]
    omega
  · simp only [BitVec.ssubOverflow, product]
    change (decide ((player.toNat : Int) * 2 - 64 ≥ 2147483648) ||
      decide ((player.toNat : Int) * 2 - 64 < -2147483648)) = false
    simp only [Bool.or_eq_false_iff, decide_eq_false_iff_not]
    omega

theorem finishScore_policy (player opponent empties : Score) (playerBound : player.toNat ≤ 64)
    (emptyBound : empties.toNat ≤ 64)
    (conservation : player.toNat + opponent.toNat + empties.toNat = 64) :
    (finishScore (player * 2 - 64) empties).toInt =
      if player.toNat = opponent.toNat then 0
      else if player.toNat > opponent.toNat then (player.toNat : Int) - opponent.toNat + empties.toNat
      else (player.toNat : Int) - opponent.toNat - empties.toNat := by
  have scoreValue := score_word_value player playerBound
  have diffValue := score_bounded_add (player * 2 - 64) empties (by omega) (by omega) emptyBound
  have awardedValue := score_bounded_add ((player * 2 - 64) + empties) empties
    (by omega) (by omega) emptyBound
  simp only [finishScore]
  split <;> rename_i tie
  · rw [diffValue, scoreValue] at tie ⊢
    have equal : player.toNat = opponent.toNat := by omega
    simp [equal] at *
    omega
  · split <;> rename_i win
    · rw [awardedValue, diffValue, scoreValue]
      rw [diffValue, scoreValue] at tie win
      have different : player.toNat ≠ opponent.toNat := by omega
      have greater : player.toNat > opponent.toNat := by omega
      simp [different, greater]
      omega
    · rw [scoreValue]
      rw [diffValue, scoreValue] at tie win
      have different : player.toNat ≠ opponent.toNat := by omega
      have notGreater : ¬player.toNat > opponent.toNat := by omega
      simp [different, notGreater]
      omega

end NeuralReversi.Board

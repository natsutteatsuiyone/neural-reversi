import NeuralReversi.Board.ScaledModel
import NeuralReversi.Board.ScoreProperties
import NeuralReversi.ScaledScore.Properties

namespace NeuralReversi.Board
open Bitboard

theorem player_empty_budget (board : Board) :
    (playerCount board).toNat + (emptyCount board).toNat ≤ 64 := by
  have disjoint : board.player &&& emptySquares board = 0#64 := by
    ext i hi
    simp only [emptySquares, BitVec.getElem_and, BitVec.getElem_not, BitVec.getElem_or,
      BitVec.getElem_zero]
    cases board.player[i] <;> simp
  have countUnion := count_or_of_disjoint board.player (emptySquares board) disjoint
  have bounded := (populationCountContract_proven (board.player ||| emptySquares board)).2.1
  change (count board.player).toNat + (count (emptySquares board)).toNat ≤ 64
  omega

theorem actual_finish_bounds (board : Board) :
    -64 ≤ (finishScore (discScore board) (emptyCount board)).toInt ∧
    (finishScore (discScore board) (emptyCount board)).toInt ≤ 64 := by
  have scoreFacts := discScoreContract_proven board
  have emptiesBound := (populationCountContract_proven (emptySquares board)).2.1
  have budget := player_empty_budget board
  have diffValue := score_bounded_add (discScore board) (emptyCount board)
    scoreFacts.2.1 (by omega) emptiesBound
  have awardedValue := score_bounded_add (discScore board + emptyCount board) (emptyCount board)
    (by omega) (by omega) emptiesBound
  by_cases tie : (discScore board + emptyCount board).toInt = 0
  · simp [finishScore, tie]
  · by_cases win : (discScore board + emptyCount board).toInt > 0
    · simp only [finishScore, tie, ↓reduceIte, win]
      rw [awardedValue, diffValue]
      have scoreValue := scoreFacts.1
      constructor <;> omega
    · simpa only [finishScore, tie, ↓reduceIte, win] using
        And.intro scoreFacts.2.1 scoreFacts.2.2.1

theorem scale_small (checked : Bool) (score : Score) (lo : -64 ≤ score.toInt)
    (hi : score.toInt ≤ 64) :
    ScaledScore.fromDiscDiff checked score = .ok ⟨score <<< 8⟩ := by
  have valid : ¬ (score.toInt < -127 ∨ score.toInt > 127) := by omega
  simp [ScaledScore.fromDiscDiff, valid]

def FinalScaledContract : Prop :=
  ∀ (checked : Bool) (board : Board),
    (finalScoreScaled checked board).map (fun score => (ScaledScore.value score).toInt) =
      if checked && emptySquares board != 0 then
        .error "final_score requires no empty squares; use solve() for boards with empties"
      else .ok (256 * (2 * ((playerCount board).toNat : Int) - 64))

theorem finalScaledContract_proven : FinalScaledContract := by
  intro checked board
  have facts := discScoreContract_proven board
  have scale := scale_small checked (discScore board) facts.2.1 facts.2.2.1
  have raw := ScaledScore.bounded_scaled_value (discScore board) (by omega) (by omega)
  cases guardValue : (checked && emptySquares board != 0) with
  | true => simp only [finalScoreScaled, finalScore, guardValue, ↓reduceIte, Except.map]
  | false =>
    simp only [finalScoreScaled, finalScore, guardValue, Bool.false_eq_true, ↓reduceIte, scale, Except.map,
      ScaledScore.value, Except.ok.injEq]
    rw [raw, facts.1]
    omega

def SolveScaledContract : Prop :=
  ∀ (board : Board) (empties : BitVec 32),
    (solveScaled false board empties).map (fun score => (ScaledScore.value score).toInt) =
      .ok ((releaseScoreFormula board empties * 256).bmod 4294967296) ∧
    (solveScaled true board empties =
      .ok ⟨finishScore (discScore board) empties <<< 8⟩ ↔ empties = emptyCount board) ∧
    (empties ≠ emptyCount board → solveScaled true board empties =
      .error s!"n_empties ({empties.toNat}) does not match actual empty count ({(emptyCount board).toNat})") ∧
    (empties = emptyCount board → ∃ result : ScaledScore.ScaledScore,
      solveScaled true board empties = .ok result ∧ solveScaled false board empties = .ok result ∧
      ScaledScore.toDiscDiff result = finishScore (discScore board) empties ∧
      (ScaledScore.value result).toInt = (finishScore (discScore board) empties).toInt * 256 ∧
      -16384 ≤ (ScaledScore.value result).toInt ∧ (ScaledScore.value result).toInt ≤ 16384) ∧
    (wellFormed board → empties = emptyCount board →
      (solveScaled true board empties).map (fun score => (ScaledScore.value score).toInt) =
        .ok (256 * AwardedScore board))

theorem solveScaledContract_proven : SolveScaledContract := by
  intro board empties
  have release : solveScaled false board empties =
      .ok ⟨finishScore (discScore board) empties <<< 8⟩ := by
    simp [solveScaled, solve, ScaledScore.fromDiscDiff]
  have releaseFormula : (finishScore (discScore board) empties).toInt =
      releaseScoreFormula board empties := by
    simpa only [solve, Bool.false_and, Bool.false_eq_true, ↓reduceIte, Except.map,
      Except.ok.injEq] using releaseSolveContract_proven board empties
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · simp only [release, Except.map, ScaledScore.value, Except.ok.injEq]
    rw [ScaledScore.scaled_value, releaseFormula]
  · by_cases equal : empties = emptyCount board
    · subst empties
      have bounds := actual_finish_bounds board
      simp [solveScaled, solve, scale_small true _ bounds.1 bounds.2]
    · simp [solveScaled, solve, equal]
  · intro unequal
    simp [solveScaled, solve, unequal]
  · intro equal
    subst empties
    have bounds := actual_finish_bounds board
    let score := finishScore (discScore board) (emptyCount board)
    have scoreLo : -64 ≤ score.toInt := bounds.1
    have scoreHi : score.toInt ≤ 64 := bounds.2
    have raw := ScaledScore.bounded_scaled_value score (by omega) (by omega)
    refine ⟨⟨score <<< 8⟩, ?_, release, ?_, raw, ?_, ?_⟩
    · simp [solveScaled, solve, scale_small true score bounds.1 bounds.2, score]
    · exact ScaledScore.bounded_roundtrip score (by omega) (by omega)
    · change -16384 ≤ (score <<< 8).toInt
      rw [raw]
      omega
    · change (score <<< 8).toInt ≤ 16384
      rw [raw]
      omega
  · intro disjoint equal
    subst empties
    obtain ⟨score, _, checked, policy, lo, hi, _⟩ := solvePolicyContract_proven board disjoint
    simp only [solveScaled, checked]
    rw [scale_small true score lo hi]
    simp only [Except.map, ScaledScore.value, Except.ok.injEq]
    rw [ScaledScore.bounded_scaled_value score (by omega) (by omega), policy]
    omega

end NeuralReversi.Board

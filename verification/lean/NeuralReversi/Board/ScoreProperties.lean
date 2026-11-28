module
public import NeuralReversi.Board.ScoreArithmetic
public import NeuralReversi.Board.Counting

@[expose] public section

namespace NeuralReversi.Board
open Bitboard

def DiscScoreContract : Prop :=
  ∀ b : Board,
    (discScore b).toInt = 2 * ((playerCount b).toNat : Int) - 64 ∧
    -64 ≤ (discScore b).toInt ∧ (discScore b).toInt ≤ 64 ∧
    BitVec.smulOverflow (playerCount b) 2 = false ∧
    BitVec.ssubOverflow (playerCount b * 2) 64 = false

theorem discScoreContract_proven : DiscScoreContract := by
  intro b
  have bound : (playerCount b).toNat ≤ 64 := (populationCountContract_proven b.player).2.1
  have value : (discScore b).toInt = 2 * ((playerCount b).toNat : Int) - 64 :=
    score_word_value (playerCount b) bound
  have safe := score_ops_safe (playerCount b) bound
  refine ⟨value, ?_, ?_, safe.1, safe.2⟩ <;> omega

def FinalScoreContract : Prop :=
  ∀ b : Board,
    finalScore false b = .ok (discScore b) ∧
    (finalScore true b = .ok (discScore b) ↔ emptySquares b = 0#64) ∧
    (emptySquares b ≠ 0#64 → finalScore true b =
      .error "final_disc_diff requires no empty squares; use solve_disc_diff() for boards with empties") ∧
    (wellFormed b → emptySquares b = 0#64 →
      (discScore b).toInt = ((playerCount b).toNat : Int) - (opponentCount b).toNat)

theorem finalScoreContract_proven : FinalScoreContract := by
  intro b
  refine ⟨by rfl, ?_, ?_, ?_⟩
  · by_cases empty : emptySquares b = 0#64 <;> simp_all [finalScore]
  · intro nonempty
    simp_all [finalScore]
  · intro disjoint full
    have conservation := countConservation_proven b disjoint
    have noEmpties : (emptyCount b).toNat = 0 := by
      simp [emptyCount, full, count]
    have value := (discScoreContract_proven b).1
    omega

def SolveGuardContract : Prop :=
  ∀ (b : Board) (empties : BitVec 32),
    solve false b empties = .ok (finishScore (discScore b) empties) ∧
    (solve true b empties = .ok (finishScore (discScore b) empties) ↔ empties = emptyCount b) ∧
    (empties ≠ emptyCount b → solve true b empties =
      .error s!"n_empties ({empties.toNat}) does not match actual empty count ({(emptyCount b).toNat})")

theorem solveGuardContract_proven : SolveGuardContract := by
  intro b empties
  by_cases equal : empties = emptyCount b <;> simp_all [solve]

def releaseScoreFormula (b : Board) (empties : BitVec 32) : Int :=
  let score := 2 * ((playerCount b).toNat : Int) - 64
  let diff := (score + empties.toInt).bmod 4294967296
  if diff = 0 then 0
  else if diff > 0 then (diff + empties.toInt).bmod 4294967296
  else score

def ReleaseSolveContract : Prop :=
  ∀ (b : Board) (empties : BitVec 32),
    (solve false b empties).map BitVec.toInt = .ok (releaseScoreFormula b empties)

theorem releaseSolveContract_proven : ReleaseSolveContract := by
  intro b empties
  have value := (discScoreContract_proven b).1
  simp only [solve, Bool.false_and, Bool.false_eq_true, ↓reduceIte, Except.map,
    Except.ok.injEq]
  simp only [finishScore, releaseScoreFormula]
  split <;> rename_i tie
  · simp only [BitVec.toInt_add, value] at tie ⊢
    simp [tie]
  · split <;> rename_i win
    · simp only [BitVec.toInt_add, value] at tie win ⊢
      simp [tie, win]
    · simp only [BitVec.toInt_add, value] at tie win ⊢
      simp [tie, win]

def AwardedScore (b : Board) : Int :=
  if (playerCount b).toNat = (opponentCount b).toNat then 0
  else if (playerCount b).toNat > (opponentCount b).toNat then
    ((playerCount b).toNat : Int) - (opponentCount b).toNat + (emptyCount b).toNat
  else ((playerCount b).toNat : Int) - (opponentCount b).toNat - (emptyCount b).toNat

def SolvePolicyContract : Prop :=
  ∀ b : Board, wellFormed b →
    ∃ result : Score,
      solve false b (emptyCount b) = .ok result ∧ solve true b (emptyCount b) = .ok result ∧
      result.toInt = AwardedScore b ∧ -64 ≤ result.toInt ∧ result.toInt ≤ 64 ∧
      BitVec.saddOverflow (discScore b) (emptyCount b) = false ∧
      BitVec.saddOverflow (discScore b + emptyCount b) (emptyCount b) = false

theorem solvePolicyContract_proven : SolvePolicyContract := by
  intro b disjoint
  have playerBound := (populationCountContract_proven b.player).2.1
  have emptyBound := (populationCountContract_proven (emptySquares b)).2.1
  have conservation := countConservation_proven b disjoint
  have value : (finishScore (discScore b) (emptyCount b)).toInt = AwardedScore b :=
    finishScore_policy (playerCount b) (opponentCount b) (emptyCount b)
      playerBound emptyBound conservation
  have scoreFacts := discScoreContract_proven b
  have diffValue := score_bounded_add (discScore b) (emptyCount b)
    scoreFacts.2.1 (by omega) emptyBound
  have firstSafe := score_bounded_add_safe (discScore b) (emptyCount b)
    scoreFacts.2.1 (by omega) emptyBound
  have secondSafe := score_bounded_add_safe (discScore b + emptyCount b) (emptyCount b)
    (by omega) (by omega) emptyBound
  refine ⟨finishScore (discScore b) (emptyCount b), by rfl, ?_, value, ?_, ?_, firstSafe, secondSafe⟩
  · simp [solve]
  all_goals
    rw [value]
    unfold AwardedScore
    split <;> omega

end NeuralReversi.Board

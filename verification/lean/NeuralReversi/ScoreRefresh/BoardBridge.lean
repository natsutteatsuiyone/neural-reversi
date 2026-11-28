module
public import NeuralReversi.ScoreRefresh.Model
public import NeuralReversi.Board.ScaledProperties

@[expose] public section

namespace NeuralReversi.ScoreRefresh.BoardBridge

abbrev Board := NeuralReversi.Board.Board

def discScore (b : Board) : DiscScore :=
  DiscScore.new (NeuralReversi.Board.playerCount b * 2 - discMax.raw)

def finalDiscDiff (checked : Bool) (b : Board) : Except String DiscScore :=
  if checked && NeuralReversi.Board.emptySquares b != 0 then
    .error "final_disc_diff requires no empty squares; use solve_disc_diff() for boards with empties"
  else .ok (discScore b)

def finalScore (checked : Bool) (b : Board) : Except String Score :=
  match finalDiscDiff checked b with
  | .error reason => .error reason
  | .ok s => fromDiscDiff checked s

def solveDiscDiff (checked : Bool) (b : Board) (empties : Word) : Except String DiscScore :=
  if checked && empties != NeuralReversi.Board.emptyCount b then
    .error s!"n_empties ({empties.toNat}) does not match actual empty count ({(NeuralReversi.Board.emptyCount b).toNat})"
  else
    let score := (discScore b).raw
    let diff := score + empties
    .ok (DiscScore.new (if diff.toInt = 0 then diff else if diff.toInt > 0 then diff + empties else score))

def solve (checked : Bool) (b : Board) (empties : Word) : Except String Score :=
  match solveDiscDiff checked b empties with
  | .error reason => .error reason
  | .ok s => fromDiscDiff checked s

theorem discScore_raw (b : Board) : (discScore b).raw = NeuralReversi.Board.discScore b := rfl

theorem solveDiscDiff_raw (checked : Bool) (b : Board) (empties : Word) :
    (solveDiscDiff checked b empties).map DiscScore.value = NeuralReversi.Board.solve checked b empties := by
  unfold solveDiscDiff NeuralReversi.Board.solve
  split <;> rfl

theorem solveDiscDiff_wrap (checked : Bool) (b : Board) (empties : Word) :
    solveDiscDiff checked b empties = (NeuralReversi.Board.solve checked b empties).map DiscScore.new := by
  unfold solveDiscDiff NeuralReversi.Board.solve
  split <;> rfl

theorem finalDiscDiff_raw (checked : Bool) (b : Board) :
    (finalDiscDiff checked b).map DiscScore.value =
      if checked && NeuralReversi.Board.emptySquares b != 0 then
        .error "final_disc_diff requires no empty squares; use solve_disc_diff() for boards with empties"
      else .ok (NeuralReversi.Board.discScore b) := by
  unfold finalDiscDiff
  split <;> rfl

theorem solve_scaled_legacy (checked : Bool) (b : Board) (empties : Word) :
    (solve checked b empties).map legacy = NeuralReversi.Board.solveScaled checked b empties := by
  rw [solve, solveDiscDiff_wrap]
  unfold NeuralReversi.Board.solveScaled
  cases h : NeuralReversi.Board.solve checked b empties with
  | error reason => rfl
  | ok s =>
    simp only [Except.map]
    cases checked with
    | false => rfl
    | true =>
      have he : empties = NeuralReversi.Board.emptyCount b := by
        by_cases he : empties = NeuralReversi.Board.emptyCount b
        · exact he
        · simp [NeuralReversi.Board.solve, he] at h
      subst empties
      have hval : s = NeuralReversi.Board.finishScore (NeuralReversi.Board.discScore b)
          (NeuralReversi.Board.emptyCount b) := by
        simpa [NeuralReversi.Board.solve] using h.symm
      have hb := NeuralReversi.Board.actual_finish_bounds b
      rw [← hval] at hb
      have valid : ¬ (s.toInt < -127 ∨ s.toInt > 127) := by omega
      simp [fromDiscDiff, ScaledScore.fromDiscDiff, DiscScore.new, valid, legacy]

theorem solveDiscDiff_awarded (b : Board) (wf : NeuralReversi.Board.wellFormed b) :
    ∃ s : DiscScore,
      solveDiscDiff false b (NeuralReversi.Board.emptyCount b) = .ok s ∧
      solveDiscDiff true b (NeuralReversi.Board.emptyCount b) = .ok s ∧
      s.raw.toInt = NeuralReversi.Board.AwardedScore b ∧ -64 ≤ s.raw.toInt ∧ s.raw.toInt ≤ 64 := by
  obtain ⟨s, release, checked, score, lo, hi, _⟩ := NeuralReversi.Board.solvePolicyContract_proven b wf
  refine ⟨DiscScore.new s, ?_, ?_, score, lo, hi⟩
  · rw [solveDiscDiff_wrap, release]; rfl
  · rw [solveDiscDiff_wrap, checked]; rfl

theorem solve_scaled_awarded (b : Board) (wf : NeuralReversi.Board.wellFormed b) :
    (solve true b (NeuralReversi.Board.emptyCount b)).map (fun s => s.raw.toInt) =
      .ok (256 * NeuralReversi.Board.AwardedScore b) := by
  have old := (NeuralReversi.Board.solveScaledContract_proven b (NeuralReversi.Board.emptyCount b)).2.2.2.2 wf rfl
  rw [← solve_scaled_legacy] at old
  cases hs : solve true b (NeuralReversi.Board.emptyCount b) <;>
    simp_all [Except.map, legacy, ScaledScore.value]

theorem finalScore_scaled (checked : Bool) (b : Board) :
    (finalScore checked b).map (fun s => s.raw.toInt) =
      if checked && NeuralReversi.Board.emptySquares b != 0 then
        .error "final_disc_diff requires no empty squares; use solve_disc_diff() for boards with empties"
      else .ok (256 * (2 * ((NeuralReversi.Board.playerCount b).toNat : Int) - 64)) := by
  have hb := NeuralReversi.Board.discScoreContract_proven b
  have valid : ¬ ((NeuralReversi.Board.discScore b).toInt < -127 ∨
      (NeuralReversi.Board.discScore b).toInt > 127) := by omega
  have scaled := ScaledScore.bounded_scaled_value (NeuralReversi.Board.discScore b)
    (by omega) (by omega)
  cases hg : (checked && NeuralReversi.Board.emptySquares b != 0) with
  | true => simp only [finalScore, finalDiscDiff, hg, ↓reduceIte, Except.map]
  | false =>
    simp only [finalScore, finalDiscDiff, hg, Bool.false_eq_true, ↓reduceIte,
      fromDiscDiff, discScore_raw, valid, decide_false, Bool.and_false, Except.map, Except.ok.injEq]
    rw [scaled, hb.1]
    omega

end NeuralReversi.ScoreRefresh.BoardBridge

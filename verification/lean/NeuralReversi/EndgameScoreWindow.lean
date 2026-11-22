import NeuralReversi.ScoreRefresh.Properties
import NeuralReversi.Solve34.Target
import NeuralReversi.TranspositionTable.Proofs

namespace NeuralReversi.EndgameScoreWindow
open ScoreRefresh (Score DiscScore toDiscDiff fromDiscDiff legacy)
open TranspositionTable (classify Bound)

def WindowContract : Prop :=
  ∀ (checked : Bool) (alpha beta : Score) (result : DiscScore) (truth : Int),
    beta.raw.toInt = alpha.raw.toInt + 1 →
    -127 ≤ result.raw.toInt → result.raw.toInt ≤ 127 →
    ∃ score, fromDiscDiff checked result = .ok score ∧ score.raw.toInt = result.raw.toInt * 256 ∧
      let discAlpha := (toDiscDiff alpha).raw.toInt
      let sound := Solve2.NWS discAlpha truth result.raw.toInt
      let bound := classify false (legacy score) (legacy alpha) (legacy beta)
      (sound ↔ Solve2.NWS alpha.raw.toInt (truth * 256) score.raw.toInt) ∧
      bound = (if discAlpha < result.raw.toInt then Bound.lower else Bound.upper) ∧
      (sound → (bound = .lower → score.raw.toInt ≤ truth * 256) ∧
               (bound = .upper → truth * 256 ≤ score.raw.toInt))

theorem windowContract_proven : WindowContract := by
  intro checked alpha beta result truth window lo hi
  obtain ⟨score, ctor, _, scaled, _⟩ := ScoreRefresh.bounded_roundtrip result lo hi
  have valid : ¬ (result.raw.toInt < -127 ∨ result.raw.toInt > 127) := by omega
  have converted : fromDiscDiff checked result = .ok score := by
    simpa [fromDiscDiff, valid] using ctor
  refine ⟨score, converted, scaled, ?_⟩
  dsimp only
  rw [ScoreRefresh.disc_floor, scaled]
  have side : beta.raw.toInt ≤ score.raw.toInt ↔ alpha.raw.toInt / 256 < result.raw.toInt := by
    omega
  simp only [classify, legacy, ge_iff_le, side, Bool.false_and, Bool.false_eq_true, ↓reduceIte,
    Solve2.NWS]
  split <;> simp_all <;> omega

theorem solve4_window (checked : Bool) (board : Solve34.EmptyBoard 4) (alpha beta : Score)
    (window : beta.raw.toInt = alpha.raw.toInt + 1) :
    let discAlpha := (toDiscDiff alpha).raw
    let result : DiscScore := ⟨(Solve34.source4Run board discAlpha).score⟩
    let truth := Solve34.exactFourValue board * 256
    ∃ score, fromDiscDiff checked result = .ok score ∧
      Solve2.NWS alpha.raw.toInt truth score.raw.toInt ∧
      let bound := classify false (legacy score) (legacy alpha) (legacy beta)
      (bound = .lower → score.raw.toInt ≤ truth) ∧ (bound = .upper → truth ≤ score.raw.toInt) := by
  let discAlpha := (toDiscDiff alpha).raw
  have safe : Solve34.WindowSafe discAlpha := by
    have floor := ScoreRefresh.disc_floor alpha
    have lo := alpha.raw.le_toInt
    have hi := alpha.raw.toInt_lt
    change -2147483648 < (toDiscDiff alpha).raw.toInt ∧ (toDiscDiff alpha).raw.toInt < 2147483647
    omega
  have valid := Solve34.solve4Concrete_bounds board discAlpha safe
  rw [← Solve34.source4Run_score] at valid
  let result : DiscScore := ⟨(Solve34.source4Run board discAlpha).score⟩
  have range : -114 ≤ result.raw.toInt ∧ result.raw.toInt ≤ 114 := valid
  obtain ⟨score, converted, _, equiv, _, classified⟩ :=
    windowContract_proven checked alpha beta result (Solve34.exactFourValue board) window
      (by omega) (by omega)
  have sound := Solve34.source4Run_nws board discAlpha safe
  exact ⟨score, converted, equiv.mp sound, classified sound⟩

open PvsBoardTree (Position minimax)
open EndgameTreeCache (discValue)

private def DirectResult (value : Int) (checked : Bool) (alpha beta : Score) (result : DiscScore) : Prop :=
  ∃ score, fromDiscDiff checked result = .ok score ∧
    Solve2.NWS alpha.raw.toInt value score.raw.toInt ∧
    let bound := classify false (legacy score) (legacy alpha) (legacy beta)
    (bound = .lower → score.raw.toInt ≤ value) ∧ (bound = .upper → value ≤ score.raw.toInt)

private theorem directResult (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config)
    (board : Position) (checked : Bool) (alpha beta : Score) (result : DiscScore)
    (window : beta.raw.toInt = alpha.raw.toInt + 1)
    (bounds : EndgameCache.SearchRange result.raw.toInt ∧
      Solve2.NWS (toDiscDiff alpha).raw.toInt (discValue mc fc board.val) result.raw.toInt) :
    DirectResult (minimax mc fc board) checked alpha beta result := by
  obtain ⟨score, converted, _, equiv, _, classified⟩ :=
    windowContract_proven checked alpha beta result (discValue mc fc board.val) window bounds.1.1 bounds.1.2
  have scaled := EndgameTreeCache.scaled_discValue mc fc board
  unfold DirectResult
  rw [← scaled, Int.mul_comm]
  exact ⟨score, converted, equiv.mp bounds.2, classified bounds.2⟩

def DirectTargetContract : Prop :=
  ∀ (target : Flip.Dispatch.TargetConfig) (bmi2 : Bool) (mc : Bitboard.Moves.X86Config)
    (fc : Flip.Dispatch.X86Config) (checked : Bool) (alpha beta : Score),
    beta.raw.toInt = alpha.raw.toInt + 1 →
    let a := (toDiscDiff alpha).raw
    (∀ b : Solve34.EmptyBoard 2, DirectResult (minimax mc fc ⟨b.asBoard, b.disjoint⟩) checked alpha beta
      ⟨(Solve2.targetRun target bmi2 b.toTwoEmpty a).score⟩) ∧
    (∀ b : Solve34.EmptyBoard 3, DirectResult (minimax mc fc ⟨b.asBoard, b.disjoint⟩) checked alpha beta
      ⟨(Solve34.target3Run target bmi2 b a).score⟩) ∧
    (∀ b : Solve34.EmptyBoard 4, DirectResult (minimax mc fc ⟨b.asBoard, b.disjoint⟩) checked alpha beta
      ⟨(Solve34.target4Run target bmi2 b a).score⟩)

theorem directTargetContract_proven : DirectTargetContract := by
  intro target bmi2 mc fc checked alpha beta window
  let a := (toDiscDiff alpha).raw
  have safe : Solve34.WindowSafe a := by
    have floor := ScoreRefresh.disc_floor alpha
    have lo := alpha.raw.le_toInt
    have hi := alpha.raw.toInt_lt
    change -2147483648 < (toDiscDiff alpha).raw.toInt ∧ (toDiscDiff alpha).raw.toInt < 2147483647
    omega
  obtain ⟨two, three, four⟩ := Solve34.targetTreeContract_proven target bmi2 mc fc a safe
  exact ⟨fun b => directResult mc fc ⟨b.asBoard, b.disjoint⟩ checked alpha beta _ window (two b),
    fun b => directResult mc fc ⟨b.asBoard, b.disjoint⟩ checked alpha beta _ window (three b),
    fun b => directResult mc fc ⟨b.asBoard, b.disjoint⟩ checked alpha beta _ window (four b)⟩

end NeuralReversi.EndgameScoreWindow

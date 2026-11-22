import NeuralReversi.Board.ScaledProperties
import NeuralReversi.MidgameAspiration.Model

namespace NeuralReversi.MidgameScoreRange

open Aspiration (Window)
open MidgameAspiration (FiniteScore)

def isFinite (raw : BitVec 32) : Bool :=
  decide ((if raw.msb then (-raw).toNat else raw.toNat) < 32767)

def GateContract : Prop :=
  ∀ raw : BitVec 32, isFinite raw = true ↔ FiniteScore raw.toInt

theorem gateContract_proven : GateContract := by
  intro raw
  have magnitude : (if raw.msb then (-raw).toNat else raw.toNat) = raw.toInt.natAbs := by
    cases sign : raw.msb
    · simp only [Bool.false_eq_true, ite_false, BitVec.toInt_eq_toNat_of_msb sign, Int.natAbs_natCast]
    · simp only [ite_true, ScaledScore.negative_magnitude raw sign,
        Int.natAbs_neg, Int.natAbs_natCast]
  unfold isFinite FiniteScore
  rw [magnitude, decide_eq_true_eq]
  omega

theorem finite_zero : FiniteScore 0 := ⟨by decide, by decide⟩

theorem finite_neg {score : Int} (finite : FiniteScore score) : FiniteScore (-score) := by
  unfold FiniteScore at *
  omega

theorem finite_max {left right : Int} (leftFinite : FiniteScore left)
    (rightFinite : FiniteScore right) : FiniteScore (max left right) := by
  unfold FiniteScore at *
  omega

def evalClamp (raw : Int) : Int := min (max raw (-16383)) 16383

theorem eval_clamp_bounded (raw : Int) : -16383 ≤ evalClamp raw ∧ evalClamp raw ≤ 16383 := by
  unfold evalClamp
  omega

theorem eval_clamp_finite (raw : Int) : FiniteScore (evalClamp raw) := by
  have := eval_clamp_bounded raw
  unfold FiniteScore
  omega

theorem disc_score_finite (discs : Int) (lo : -127 ≤ discs) (hi : discs ≤ 127) :
    FiniteScore (discs * 256) := by
  unfold FiniteScore
  omega

def EvaluationContract : Prop :=
  (∀ raw : Int, FiniteScore (evalClamp raw)) ∧
  (∀ (board : Board.Board) checked, ∃ result,
    Board.solveScaled checked board (Board.emptyCount board) = .ok result ∧
      FiniteScore (ScaledScore.value result).toInt) ∧
  (∀ count : Nat, count ≤ 64 → FiniteScore ((64 - 2 * (count : Int)) * 256))

theorem evaluationContract_proven : EvaluationContract := by
  refine ⟨eval_clamp_finite, ?_, ?_⟩
  · intro board checked
    obtain ⟨result, debug, release, _, _, lo, hi⟩ :=
      (Board.solveScaledContract_proven board (Board.emptyCount board)).2.2.2.1 rfl
    refine ⟨result, ?_, ?_⟩
    · cases checked
      · exact release
      · exact debug
    · unfold FiniteScore
      omega
  · intro count range
    exact disc_score_finite _ (by omega) (by omega)

theorem reflected {alpha beta : Int} (window : Window alpha beta) : Window (-beta) (-alpha) := by
  obtain ⟨lo, order, hi⟩ := window
  exact ⟨by omega, by omega, by omega⟩

theorem scout_window {alpha beta : Int} (window : Window alpha beta) :
    Window (-(alpha + 1)) (-alpha) := by
  obtain ⟨lo, order, hi⟩ := window
  exact ⟨by omega, by omega, by omega⟩

theorem raised_window {alpha beta score : Int} (window : Window alpha beta)
    (raises : alpha < score) (below : score < beta) : Window score beta := by
  obtain ⟨lo, order, hi⟩ := window
  exact ⟨by omega, below, hi⟩

def WindowContract : Prop :=
  ∀ alpha beta : Int, Window alpha beta →
    Window (-beta) (-alpha) ∧ Window (-(alpha + 1)) (-alpha) ∧
    (∀ score, alpha < score → score < beta → Window score beta)

theorem windowContract_proven : WindowContract := by
  intro alpha beta valid
  exact ⟨reflected valid, scout_window valid, fun _ => raised_window valid⟩

def ttMargin : Int := 1024

theorem margin_finite {alpha beta stored : Int} (window : Window alpha beta)
    (cached : FiniteScore stored) (cuts : beta + ttMargin ≤ stored) :
    FiniteScore (beta + ttMargin) := by
  unfold ttMargin at *
  obtain ⟨lo, order, hi⟩ := window
  obtain ⟨storedLo, storedHi⟩ := cached
  exact ⟨by omega, by omega⟩

example : isFinite (BitVec.ofInt 32 (-32767)) = false := by decide
example : isFinite (BitVec.ofInt 32 (-32766)) = true := by decide
example : isFinite (BitVec.ofInt 32 (-2147483648)) = false := by decide

end NeuralReversi.MidgameScoreRange

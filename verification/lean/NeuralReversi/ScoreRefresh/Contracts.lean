import NeuralReversi.ScoreRefresh.BoardBridge
import NeuralReversi.ScoreRefresh.Properties

namespace NeuralReversi.ScoreRefresh

def ConstructorContract : Prop :=
  (∀ debugAssertions raw,
    (fromRaw debugAssertions raw).map Score.raw =
      if debugAssertions && decide (raw.toInt < -32767 ∨ 32767 < raw.toInt) then
        .error "assertion failed: raw_value >= -Self::INF.0 && raw_value <= Self::INF.0"
      else .ok raw) ∧
  ∀ debugAssertions discs,
    (fromDiscDiff debugAssertions discs).map Score.raw =
      if debugAssertions && decide (discs.raw.toInt < -127 ∨ 127 < discs.raw.toInt) then
        .error discAssertion
      else .ok (discs.raw <<< 8)

theorem constructorContract_proven : ConstructorContract := by
  constructor
  · intro debugAssertions raw
    unfold fromRaw
    split <;> rfl
  · intro debugAssertions discs
    unfold fromDiscDiff
    split <;> rfl

def DiscArithmeticContract : Prop :=
  (∀ raw, DiscScore.value (DiscScore.new raw) = raw) ∧
  ∀ checked (a : DiscScore) (b : Word),
    observeDisc (DiscScore.negate checked a) =
      ScaledScore.checkedMathSpec checked (-a.raw.toInt) "attempt to negate with overflow" ∧
    observeDisc (DiscScore.addInt checked a b) =
      ScaledScore.checkedMathSpec checked (a.raw.toInt + b.toInt) "attempt to add with overflow" ∧
    observeDisc (DiscScore.subInt checked a b) =
      ScaledScore.checkedMathSpec checked (a.raw.toInt - b.toInt) "attempt to subtract with overflow"

theorem discArithmeticContract_proven : DiscArithmeticContract :=
  ⟨disc_unrestricted, fun checked a b =>
    ⟨disc_negate_spec checked a, disc_addInt_spec checked a b, disc_subInt_spec checked a b⟩⟩

def BoardScoreContract : Prop :=
  (∀ checked (b : BoardBridge.Board),
    (BoardBridge.finalScore checked b).map (fun s => s.raw.toInt) =
      if checked && Board.emptySquares b != 0 then
        .error "final_disc_diff requires no empty squares; use solve_disc_diff() for boards with empties"
      else .ok (256 * (2 * ((Board.playerCount b).toNat : Int) - 64))) ∧
  ∀ (b : BoardBridge.Board), Board.wellFormed b →
    (BoardBridge.solve true b (Board.emptyCount b)).map (fun s => s.raw.toInt) =
      .ok (256 * Board.AwardedScore b)

theorem boardScoreContract_proven : BoardScoreContract :=
  ⟨BoardBridge.finalScore_scaled, BoardBridge.solve_scaled_awarded⟩

end NeuralReversi.ScoreRefresh

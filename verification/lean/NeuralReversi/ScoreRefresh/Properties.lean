module
public import NeuralReversi.ScoreRefresh.Model
public import NeuralReversi.ScaledScore.Properties
public import NeuralReversi.ScaledScore.ArithmeticProperties

@[expose] public section

namespace NeuralReversi.ScoreRefresh
set_option linter.unusedSimpArgs false

theorem legacy_inverse (score : Score) : ofLegacy (legacy score) = score := rfl
theorem ofLegacy_inverse (score : ScaledScore.ScaledScore) : legacy (ofLegacy score) = score := rfl

theorem fromRaw_bridge (checked : Bool) (raw : Word) :
    (fromRaw checked raw).map legacy = ScaledScore.fromRaw checked raw := by
  unfold fromRaw ScaledScore.fromRaw
  split <;> rfl

def refreshDiscResult : Except String ScaledScore.ScaledScore → Except String Score
  | .ok score => .ok (ofLegacy score)
  | .error _ => .error discAssertion

theorem fromDiscDiff_bridge (checked : Bool) (discs : DiscScore) :
    fromDiscDiff checked discs = refreshDiscResult (ScaledScore.fromDiscDiff checked discs.raw) := by
  unfold fromDiscDiff ScaledScore.fromDiscDiff
  split <;> rfl

theorem disc_assertion_changed : discAssertion ≠
    "assertion failed: disc_diff >= -SCORE_INF && disc_diff <= SCORE_INF" := by decide

theorem disc_constructor_rejection (discs : DiscScore)
    (outside : discs.raw.toInt < -127 ∨ discs.raw.toInt > 127) :
    fromDiscDiff true discs = .error discAssertion ∧
    ScaledScore.fromDiscDiff true discs.raw =
      .error "assertion failed: disc_diff >= -SCORE_INF && disc_diff <= SCORE_INF" := by
  simp [fromDiscDiff, ScaledScore.fromDiscDiff, outside]

theorem toDiscDiff_bridge (score : Score) :
    (toDiscDiff score).raw = ScaledScore.toDiscDiff (legacy score) := rfl

theorem float_bridge (score : Score) :
    toDiscDiffF32 score = ScaledScore.toDiscDiffF32 (legacy score) := rfl

theorem display_bridge {State Payload : Type} (score : Score) (options : ScaledScore.FormatOptions)
    (write : State → String → ScaledScore.WriteResult State Payload) (state : State) :
    display score options write state = ScaledScore.display score.raw options write state := rfl

def observe (result : Except String Score) : Except String Int :=
  result.map (fun score => score.raw.toInt)

def observeDisc (result : Except String DiscScore) : Except String Int :=
  result.map (fun score => score.raw.toInt)

theorem addWords_spec (checked : Bool) (a b : Word) :
    observe (addWords checked a b) =
      ScaledScore.checkedMathSpec checked (a.toInt + b.toInt) "attempt to add with overflow" := by
  unfold observe addWords
  rw [ScaledScore.add_overflow_spec]
  unfold ScaledScore.checkedMathSpec
  split <;> simp [Except.map, BitVec.toInt_add]

theorem subWords_spec (checked : Bool) (a b : Word) :
    observe (subWords checked a b) =
      ScaledScore.checkedMathSpec checked (a.toInt - b.toInt) "attempt to subtract with overflow" := by
  unfold observe subWords
  rw [ScaledScore.sub_overflow_spec]
  unfold ScaledScore.checkedMathSpec
  split <;> simp [Except.map, BitVec.toInt_sub]

theorem negate_spec (checked : Bool) (a : Score) :
    observe (negate checked a) =
      ScaledScore.checkedMathSpec checked (-a.raw.toInt) "attempt to negate with overflow" := by
  unfold observe negate
  rw [ScaledScore.neg_overflow_spec]
  unfold ScaledScore.checkedMathSpec
  split <;> simp [Except.map, BitVec.toInt_neg]

theorem divInt_bridge (checked : Bool) (a : Score) (b : Word) :
    (divInt checked a b).map legacy = ScaledScore.divInt checked (legacy a) b := by
  unfold divInt ScaledScore.divInt legacy
  split
  · rfl
  · split <;> rfl

theorem divInt_spec (checked : Bool) (a : Score) (b : Word) :
    observe (divInt checked a b) = ScaledScore.divisionSpec a.raw.toInt b.toInt := by
  have h := ScaledScore.divInt_spec checked (legacy a) b
  rw [← divInt_bridge] at h
  cases result : divInt checked a b <;>
    simpa [result, ScaledScore.observeScore, Except.map, observe, legacy, ScaledScore.value] using h

theorem arithmetic_contract (checked : Bool) (a b : Score) :
    observe (addScore checked a b) = ScaledScore.checkedMathSpec checked
      (a.raw.toInt+b.raw.toInt) "attempt to add with overflow" ∧
    observe (subScore checked a b) = ScaledScore.checkedMathSpec checked
      (a.raw.toInt-b.raw.toInt) "attempt to subtract with overflow" ∧
    observe (negate checked a) = ScaledScore.checkedMathSpec checked
      (-a.raw.toInt) "attempt to negate with overflow" ∧
    observe (addInt checked a b.raw) = ScaledScore.checkedMathSpec checked
      (a.raw.toInt+b.raw.toInt) "attempt to add with overflow" ∧
    observe (subInt checked a b.raw) = ScaledScore.checkedMathSpec checked
      (a.raw.toInt-b.raw.toInt) "attempt to subtract with overflow" ∧
    observe (divInt checked a b.raw) = ScaledScore.divisionSpec a.raw.toInt b.raw.toInt :=
  ⟨addWords_spec checked a.raw b.raw, subWords_spec checked a.raw b.raw,
   negate_spec checked a, addWords_spec checked a.raw b.raw,
   subWords_spec checked a.raw b.raw, divInt_spec checked a b.raw⟩

theorem disc_negate_spec (checked : Bool) (a : DiscScore) :
    observeDisc (DiscScore.negate checked a) = ScaledScore.checkedMathSpec checked
      (-a.raw.toInt) "attempt to negate with overflow" := by
  unfold observeDisc DiscScore.negate
  rw [ScaledScore.neg_overflow_spec]
  unfold ScaledScore.checkedMathSpec
  split <;> simp [Except.map, BitVec.toInt_neg]

theorem disc_addInt_spec (checked : Bool) (a : DiscScore) (b : Word) :
    observeDisc (DiscScore.addInt checked a b) = ScaledScore.checkedMathSpec checked
      (a.raw.toInt+b.toInt) "attempt to add with overflow" := by
  unfold observeDisc DiscScore.addInt
  rw [ScaledScore.add_overflow_spec]
  unfold ScaledScore.checkedMathSpec
  split <;> simp [Except.map, BitVec.toInt_add]

theorem disc_subInt_spec (checked : Bool) (a : DiscScore) (b : Word) :
    observeDisc (DiscScore.subInt checked a b) = ScaledScore.checkedMathSpec checked
      (a.raw.toInt-b.toInt) "attempt to subtract with overflow" := by
  unfold observeDisc DiscScore.subInt
  rw [ScaledScore.sub_overflow_spec]
  unfold ScaledScore.checkedMathSpec
  split <;> simp [Except.map, BitVec.toInt_sub]

theorem disc_unrestricted (raw : Word) : DiscScore.value (DiscScore.new raw) = raw := rfl

theorem disc_floor (score : Score) :
    (toDiscDiff score).raw.toInt = score.raw.toInt / 256 :=
  ScaledScore.disc_floor (legacy score)

theorem bounded_roundtrip (discs : DiscScore) (lo : -127 ≤ discs.raw.toInt)
    (hi : discs.raw.toInt ≤ 127) :
    ∃ score, fromDiscDiff true discs = .ok score ∧ toDiscDiff score = discs ∧
      score.raw.toInt = discs.raw.toInt * 256 ∧ fromRaw true score.raw = .ok score := by
  obtain ⟨old, ctor, roundtrip, scaled, raw⟩ := ScaledScore.roundTripContract_proven discs.raw lo hi
  refine ⟨ofLegacy old, ?_, ?_, scaled, ?_⟩
  · rw [fromDiscDiff_bridge, ctor]; rfl
  · cases discs
    exact congrArg DiscScore.mk roundtrip
  · change ScaledScore.fromRaw true old.raw = .ok old at raw
    have h := fromRaw_bridge true old.raw
    rw [raw] at h
    change fromRaw true old.raw = .ok (ofLegacy old)
    cases result : fromRaw true old.raw with
    | error reason => simp only [result, Except.map] at h; cases h
    | ok score => simp only [result, Except.map, Except.ok.injEq] at h
                  exact congrArg Except.ok (congrArg ofLegacy h)

end NeuralReversi.ScoreRefresh

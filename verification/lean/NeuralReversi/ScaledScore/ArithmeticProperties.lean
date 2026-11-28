module
public import NeuralReversi.ScaledScore.ArithmeticModel
public import NeuralReversi.ScaledScore.Division

@[expose] public section

namespace NeuralReversi.ScaledScore

def outsideI32 (n : Int) : Prop := n < -2147483648 ∨ 2147483647 < n
instance (n : Int) : Decidable (outsideI32 n) := inferInstanceAs (Decidable (_ ∨ _))

def checkedMathSpec (checked : Bool) (n : Int) (failure : String) : Except String Int :=
  if checked && decide (outsideI32 n) then .error failure else .ok (n.bmod 4294967296)

theorem add_overflow_spec (a b : Word) :
    a.saddOverflow b = decide (outsideI32 (a.toInt + b.toInt)) := by
  apply Bool.eq_iff_iff.mpr
  rw [decide_eq_true_eq]
  simp only [BitVec.saddOverflow, Bool.or_eq_true, decide_eq_true_eq, outsideI32]
  omega

theorem sub_overflow_spec (a b : Word) :
    a.ssubOverflow b = decide (outsideI32 (a.toInt - b.toInt)) := by
  apply Bool.eq_iff_iff.mpr
  rw [decide_eq_true_eq]
  simp only [BitVec.ssubOverflow, Bool.or_eq_true, decide_eq_true_eq, outsideI32]
  omega

theorem mul_overflow_spec (a b : Word) :
    a.smulOverflow b = decide (outsideI32 (a.toInt * b.toInt)) := by
  apply Bool.eq_iff_iff.mpr
  rw [decide_eq_true_eq]
  simp only [BitVec.smulOverflow, Bool.or_eq_true, decide_eq_true_eq, outsideI32]
  omega

theorem neg_overflow_spec (a : Word) :
    a.negOverflow = decide (outsideI32 (-a.toInt)) := by
  have lo := BitVec.le_toInt a
  have hi := BitVec.toInt_lt (x := a)
  apply Bool.eq_iff_iff.mpr
  rw [decide_eq_true_eq]
  simp only [BitVec.negOverflow, beq_iff_eq, outsideI32]
  omega

def observeScore (outcome : Except String ScaledScore) : Except String Int :=
  outcome.map (fun score => (value score).toInt)

theorem addWords_spec (checked : Bool) (a b : Word) :
    observeScore (addWords checked a b) =
      checkedMathSpec checked (a.toInt + b.toInt) "attempt to add with overflow" := by
  simp only [observeScore, addWords, add_overflow_spec, checkedMathSpec]
  split <;> simp [Except.map, value, BitVec.toInt_add]

theorem subWords_spec (checked : Bool) (a b : Word) :
    observeScore (subWords checked a b) =
      checkedMathSpec checked (a.toInt - b.toInt) "attempt to subtract with overflow" := by
  simp only [observeScore, subWords, sub_overflow_spec, checkedMathSpec]
  split <;> simp [Except.map, value, BitVec.toInt_sub]

theorem negate_spec (checked : Bool) (a : ScaledScore) :
    observeScore (negate checked a) =
      checkedMathSpec checked (-a.raw.toInt) "attempt to negate with overflow" := by
  simp only [observeScore, negate, neg_overflow_spec, checkedMathSpec]
  split <;> simp [Except.map, value, BitVec.toInt_neg]

def divisionSpec (a b : Int) : Except String Int :=
  if b = 0 then .error "attempt to divide by zero"
  else if a = -2147483648 ∧ b = -1 then .error "attempt to divide with overflow"
  else .ok (a.tdiv b)

theorem divInt_spec (checked : Bool) (a : ScaledScore) (b : Word) :
    observeScore (divInt checked a b) = divisionSpec a.raw.toInt b.toInt := by
  by_cases zero : b = 0#32
  · subst b
    rfl
  · have nonzero : b.toInt ≠ 0 := fun h => zero (BitVec.eq_of_toInt_eq h)
    by_cases overflow : a.raw = 0x80000000#32 ∧ b = 0xffffffff#32
    · simp [observeScore, divInt, overflow, divisionSpec, Except.map]
    · have safe : ¬ (a.raw = BitVec.intMin 32 ∧ b = -1) := overflow
      have specSafe : ¬ (a.raw.toInt = -2147483648 ∧ b.toInt = -1) := by
        rintro ⟨min, minusOne⟩
        exact overflow ⟨BitVec.eq_of_toInt_eq min, BitVec.eq_of_toInt_eq minusOne⟩
      simp [observeScore, divInt, zero, overflow, divisionSpec, nonzero, specSafe,
        Except.map, value, sdiv_no_wrap _ _ safe]

def observeAssignment (outcome : Except String Unit × ScaledScore) : Except String Unit × Int :=
  (outcome.1, (value outcome.2).toInt)

def assignmentSpec (before : Int) (outcome : Except String Int) : Except String Unit × Int :=
  match outcome with
  | .ok after => (.ok (), after)
  | .error reason => (.error reason, before)

theorem assignResult_spec (before : ScaledScore) (outcome : Except String ScaledScore) :
    observeAssignment (assignResult before outcome) =
      assignmentSpec before.raw.toInt (observeScore outcome) := by
  cases outcome <;> rfl

def ArithmeticContract : Prop :=
  ∀ (checked : Bool) (a b : ScaledScore),
    observeScore (addScore checked a b) =
      checkedMathSpec checked (a.raw.toInt + b.raw.toInt) "attempt to add with overflow" ∧
    observeScore (subScore checked a b) =
      checkedMathSpec checked (a.raw.toInt - b.raw.toInt) "attempt to subtract with overflow" ∧
    observeScore (negate checked a) =
      checkedMathSpec checked (-a.raw.toInt) "attempt to negate with overflow" ∧
    observeScore (addInt checked a b.raw) =
      checkedMathSpec checked (a.raw.toInt + b.raw.toInt) "attempt to add with overflow" ∧
    observeScore (subInt checked a b.raw) =
      checkedMathSpec checked (a.raw.toInt - b.raw.toInt) "attempt to subtract with overflow" ∧
    observeScore (divInt checked a b.raw) = divisionSpec a.raw.toInt b.raw.toInt

theorem arithmeticContract_proven : ArithmeticContract := by
  intro checked a b
  exact ⟨addWords_spec checked a.raw b.raw, subWords_spec checked a.raw b.raw,
    negate_spec checked a, addWords_spec checked a.raw b.raw, subWords_spec checked a.raw b.raw,
    divInt_spec checked a b.raw⟩

def AssignmentContract : Prop :=
  ∀ (checked : Bool) (a b : ScaledScore),
    observeAssignment (addAssign checked a b) = assignmentSpec a.raw.toInt
      (checkedMathSpec checked (a.raw.toInt + b.raw.toInt) "attempt to add with overflow") ∧
    observeAssignment (subAssign checked a b) = assignmentSpec a.raw.toInt
      (checkedMathSpec checked (a.raw.toInt - b.raw.toInt) "attempt to subtract with overflow")

theorem assignmentContract_proven : AssignmentContract := by
  intro checked a b
  simp only [addAssign, subAssign, assignResult_spec]
  exact ⟨congrArg (assignmentSpec a.raw.toInt) (addWords_spec checked a.raw b.raw),
    congrArg (assignmentSpec a.raw.toInt) (subWords_spec checked a.raw b.raw)⟩

theorem checkedMathSpec_safe (n : Int) (failure : String) (safe : ¬ outsideI32 n) :
    checkedMathSpec true n failure = .ok n := by
  have bounds : -2147483648 ≤ n ∧ n < 2147483648 := by unfold outsideI32 at safe; omega
  simp only [checkedMathSpec, safe, decide_false, Bool.true_and, Bool.false_eq_true, ↓reduceIte]
  exact congrArg Except.ok (Int.bmod_eq_of_le bounds.1 bounds.2)

theorem everyRaw_publicConstruction (checked : Bool) (input : Word) :
    (fromRaw checked 0).bind (fun zero => addInt checked zero input) = .ok ⟨input⟩ := by
  have lower := BitVec.le_toInt input
  have upper := BitVec.toInt_lt (x := input)
  have valid : ¬ outsideI32 input.toInt := by unfold outsideI32; omega
  have safe : (0#32).saddOverflow input = false := by
    rw [add_overflow_spec]
    simp [valid]
  simp [fromRaw, addInt, addWords, Except.bind, safe]

end NeuralReversi.ScaledScore

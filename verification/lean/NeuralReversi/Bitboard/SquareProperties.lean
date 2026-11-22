import NeuralReversi.Bitboard.SquareModel

namespace NeuralReversi.Bitboard

theorem squareMask_bit (sq : Square) (i : Nat) :
    (squareMask sq).getLsbD i = decide (i = sq.val) := by
  simp only [squareMask, BitVec.getLsbD_shiftLeft]
  have bound := sq.isLt
  by_cases h : i = sq.val
  · subst i
    simp [bound]
  · simp_all <;> omega

theorem squareMask_nonzero (sq : Square) : squareMask sq ≠ 0#64 := by
  intro zero
  have witness := squareMask_bit sq sq.val
  simp_all

theorem and_squareMask (bits : Bits) (sq : Square) :
    bits &&& squareMask sq = if bits.getLsbD sq.val then squareMask sq else 0#64 := by
  split
  · next present =>
    apply BitVec.eq_of_getLsbD_eq
    intro i hi
    simp only [BitVec.getLsbD_and, squareMask_bit]
    by_cases equal : i = sq.val
    · simp only [equal, present, decide_true, Bool.true_and]
    · simp only [equal, decide_false, Bool.and_false]
  · next absent =>
    have missing : bits.getLsbD sq.val = false := by
      cases bit : bits.getLsbD sq.val <;> simp_all
    apply BitVec.eq_of_getLsbD_eq
    intro i hi
    simp only [BitVec.getLsbD_and, squareMask_bit, BitVec.getLsbD_zero]
    by_cases equal : i = sq.val
    · simp only [equal, missing, Bool.false_and]
    · simp only [equal, decide_false, Bool.and_false]

theorem and_squareMask_zero (bits : Bits) (sq : Square) :
    bits &&& squareMask sq = 0#64 ↔ bits.getLsbD sq.val = false := by
  rw [and_squareMask]
  cases bit : bits.getLsbD sq.val <;> simp_all [squareMask_nonzero]

theorem empty_squareMask (bits : Bits) (sq : Square) :
    ~~~bits &&& squareMask sq ≠ 0#64 ↔ squareMask sq &&& bits = 0#64 := by
  rw [BitVec.and_comm (squareMask sq) bits]
  change (¬ (~~~bits &&& squareMask sq = 0#64)) ↔ bits &&& squareMask sq = 0#64
  rw [and_squareMask_zero, and_squareMask_zero]
  simp [sq.isLt]

def SquareMaskContract : Prop :=
  ∀ (sq : Square) (debugAssertions overflowChecks : Bool),
    fromSquare sq = squareMask sq ∧
    squareBitboard debugAssertions overflowChecks sq = .ok (squareMask sq) ∧
    ∀ i, (squareMask sq).getLsbD i = decide (i = sq.val)

theorem squareMaskContract_proven : SquareMaskContract := by
  intro sq debugAssertions overflowChecks
  exact ⟨rfl, rfl, squareMask_bit sq⟩

def SquareSetContract : Prop :=
  ∀ (bits : Bits) (sq : Square) (debugAssertions overflowChecks : Bool),
    ∃ result, setSquare debugAssertions overflowChecks bits sq = .ok result ∧
      ∀ i, result.getLsbD i = (bits.getLsbD i || decide (i = sq.val))

theorem squareSetContract_proven : SquareSetContract := by
  intro bits sq debugAssertions overflowChecks
  refine ⟨bits ||| squareMask sq, rfl, ?_⟩
  intro i
  simp [squareMask_bit]

def SquareMembershipContract : Prop :=
  ∀ (bits : Bits) (sq : Square) (debugAssertions overflowChecks : Bool),
    containsSquare debugAssertions overflowChecks bits sq = .ok (bits.getLsbD sq.val)

theorem squareMembershipContract_proven : SquareMembershipContract := by
  intro bits sq debugAssertions overflowChecks
  rw [containsSquare, (squareMaskContract_proven sq debugAssertions overflowChecks).2.1]
  simp only [Except.map, Except.ok.injEq]
  rw [and_squareMask]
  cases bit : bits.getLsbD sq.val <;> simp_all [squareMask_nonzero]

def EmptyBitsContract : Prop :=
  ∀ bits : Bits, isEmpty bits = true ↔ ∀ i, bits.getLsbD i = false

theorem emptyBitsContract_proven : EmptyBitsContract := by
  intro bits
  simpa [isEmpty] using (BitVec.zero_iff_eq_false (x := bits))

end NeuralReversi.Bitboard

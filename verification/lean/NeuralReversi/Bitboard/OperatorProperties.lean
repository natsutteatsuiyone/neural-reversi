module
public import NeuralReversi.Bitboard.OperatorModel

@[expose] public section

namespace NeuralReversi.Bitboard

def LogicOperatorContract : Prop :=
  ∀ (lhs rhs : Bits) (i : Nat), i < 64 →
    (bitAnd lhs rhs).getLsbD i = (lhs.getLsbD i && rhs.getLsbD i) ∧
    (bitOr lhs rhs).getLsbD i = (lhs.getLsbD i || rhs.getLsbD i) ∧
    (bitXor lhs rhs).getLsbD i = (lhs.getLsbD i ^^ rhs.getLsbD i) ∧
    (bitNot lhs).getLsbD i = !lhs.getLsbD i ∧
    (bitAndAssign lhs rhs).2.getLsbD i = (lhs.getLsbD i && rhs.getLsbD i) ∧
    (bitOrAssign lhs rhs).2.getLsbD i = (lhs.getLsbD i || rhs.getLsbD i) ∧
    (bitXorAssign lhs rhs).2.getLsbD i = (lhs.getLsbD i ^^ rhs.getLsbD i)

theorem logicOperatorContract_proven : LogicOperatorContract := by
  intro lhs rhs i hi
  simp [bitAnd, bitOr, bitXor, bitNot, bitAndAssign, bitOrAssign, bitXorAssign, hi]

theorem leftShift_bit (bits : Bits) (count i : Nat) (hi : i < 64) :
    (bits <<< count).getLsbD i = leftShiftBit bits count i := by
  rw [BitVec.getLsbD_shiftLeft]
  unfold leftShiftBit
  by_cases lower : count ≤ i
  · have notLower : ¬i < count := by omega
    simp [hi, lower, notLower]
  · have below : i < count := by omega
    simp [hi, lower, below]

theorem rightShift_bit (bits : Bits) (count i : Nat) :
    (bits >>> count).getLsbD i = rightShiftBit bits count i := by
  simp only [BitVec.getLsbD_ushiftRight, rightShiftBit, Nat.add_comm]

def ShiftOperatorContract : Prop :=
  ∀ (bits : Bits) (count : BitVec 32) (checked : Bool) (i : Nat), i < 64 →
    ((shiftLeft checked bits count).map (fun result => result.getLsbD i) =
      if checked && decide (64 ≤ count.toNat) then .error "attempt to shift left with overflow"
      else .ok (leftShiftBit bits (count.toNat % 64) i)) ∧
    ((shiftRight checked bits count).map (fun result => result.getLsbD i) =
      if checked && decide (64 ≤ count.toNat) then .error "attempt to shift right with overflow"
      else .ok (rightShiftBit bits (count.toNat % 64) i))

theorem shiftOperatorContract_proven : ShiftOperatorContract := by
  intro bits count checked i hi
  constructor
  · unfold shiftLeft
    split <;> simp only [Except.map, leftShift_bit bits _ i hi]
  · unfold shiftRight
    split <;> simp only [Except.map, rightShift_bit]

def ShiftAssignmentContract : Prop :=
  ∀ (bits : Bits) (count : BitVec 32) (checked : Bool),
    (∀ i : Nat, i < 64 →
      ((shiftLeftAssign checked bits count).1.map
          (fun _ => (shiftLeftAssign checked bits count).2.getLsbD i) =
        if checked && decide (64 ≤ count.toNat) then .error "attempt to shift left with overflow"
        else .ok (leftShiftBit bits (count.toNat % 64) i)) ∧
      ((shiftRightAssign checked bits count).1.map
          (fun _ => (shiftRightAssign checked bits count).2.getLsbD i) =
        if checked && decide (64 ≤ count.toNat) then .error "attempt to shift right with overflow"
        else .ok (rightShiftBit bits (count.toNat % 64) i))) ∧
    (checked = true → 64 ≤ count.toNat →
      (shiftLeftAssign checked bits count).2 = bits ∧
      (shiftRightAssign checked bits count).2 = bits)

theorem shiftAssignmentContract_proven : ShiftAssignmentContract := by
  intro bits count checked
  constructor
  · intro i hi
    constructor
    · by_cases overflow : (checked && decide (64 ≤ count.toNat)) = true <;>
        simp only [shiftLeftAssign, shiftLeft, overflow, Bool.false_eq_true, ↓reduceIte, Except.map,
          leftShift_bit bits _ i hi]
    · by_cases overflow : (checked && decide (64 ≤ count.toNat)) = true <;>
        simp only [shiftRightAssign, shiftRight, overflow, Bool.false_eq_true, ↓reduceIte,
          Except.map, rightShift_bit]
  · intro checks oversized
    simp [shiftLeftAssign, shiftRightAssign, shiftLeft, shiftRight, checks, oversized]

end NeuralReversi.Bitboard

import Lean

namespace NeuralReversi.Eval.InputLayer

def clip510 (v : Int) : Int := min (max v 0) 510

def scalarBase (lo hi : Int) : Int :=
  clip510 lo * clip510 hi / 1024

def scalarPhase (v : Int) : Int := scalarBase v v

private def sat8 (v : Int) : Int := min (max v 0) 255
private def satU16 (v : Int) : Int := min (max v 0) 65535
private def satI16 (v : Int) : Int := min (max v (-32768)) 32767

def x86Base (lo hi : Int) : Int :=
  sat8 ((64 * clip510 lo) * min hi 510 / 65536)

def x86Phase (v : Int) : Int :=
  sat8 ((64 * clip510 v) * clip510 v / 65536)

def neonBase (lo hi : Int) : Int :=
  let shifted := satU16 (32 * min lo 510)
  sat8 (satI16 ((2 * (shifted * min hi 510)) / 65536))

def neonPhase (v : Int) : Int := neonBase v v

private theorem clipRange (v : Int) : 0 ≤ clip510 v ∧ clip510 v ≤ 510 := by
  unfold clip510
  omega

private theorem productBounds {a b : Int} (aOk : 0 ≤ a ∧ a ≤ 510)
    (bOk : -32768 ≤ b ∧ b ≤ 510) : -16711680 ≤ a * b ∧ a * b ≤ 260100 := by
  have lower := Int.mul_le_mul_of_nonneg_left bOk.1 aOk.1
  have upper := Int.mul_le_mul_of_nonneg_left bOk.2 aOk.1
  rw [Int.mul_comm _ (-32768)] at lower
  rw [Int.mul_comm _ 510] at upper
  omega

private theorem packedProduct (a b : Int) :
    sat8 (clip510 a * min b 510 / 1024) = scalarBase a b := by
  have ar := clipRange a
  by_cases sign : b < 0
  · have clipped : clip510 b = 0 := by unfold clip510; omega
    have low : min b 510 ≤ 0 := by omega
    have prod := Int.mul_nonpos_of_nonneg_of_nonpos ar.1 low
    unfold scalarBase sat8
    rw [clipped, Int.mul_zero]
    omega
  · have clipped : clip510 b = min b 510 := by unfold clip510; omega
    have br := clipRange b
    have prod := productBounds ar ⟨by omega, br.2⟩
    have nonneg := Int.mul_nonneg ar.1 br.1
    unfold scalarBase sat8
    rw [← clipped]
    omega

private theorem x86Exact (a b : Int) : x86Base a b = scalarBase a b := by
  unfold x86Base
  rw [Int.mul_assoc]
  have shift : 64 * (clip510 a * min b 510) / 65536 =
      clip510 a * min b 510 / 1024 := by omega
  rw [shift, packedProduct]

private theorem neonExact (a b : Int) (hb : -32768 ≤ b) :
    neonBase a b = scalarBase a b := by
  have ar := clipRange a
  have br : -32768 ≤ min b 510 ∧ min b 510 ≤ 510 := by omega
  have prod := productBounds ar br
  have shift : satU16 (32 * min a 510) = 32 * clip510 a := by
    unfold satU16 clip510
    omega
  unfold neonBase
  rw [shift]
  dsimp only
  rw [Int.mul_assoc 32, ← Int.mul_assoc 2]
  have signed : satI16 (64 * (clip510 a * min b 510) / 65536) =
      clip510 a * min b 510 / 1024 := by
    unfold satI16
    omega
  rw [show (2 : Int) * 32 = 64 by decide, signed, packedProduct]

def ActivationContract : Prop :=
  (∀ a b : Int, -32768 ≤ b →
    x86Base a b = scalarBase a b ∧ neonBase a b = scalarBase a b ∧
    0 ≤ scalarBase a b ∧ scalarBase a b ≤ 254) ∧
  (∀ v : Int, -32768 ≤ v → x86Phase v = scalarPhase v ∧ neonPhase v = scalarPhase v)

theorem activationContract_proven : ActivationContract := by
  constructor
  · intro a b rb
    have ar := clipRange a
    have br := clipRange b
    have product := productBounds ar ⟨by omega, br.2⟩
    have lo := Int.mul_nonneg ar.1 br.1
    unfold scalarBase at *
    exact ⟨x86Exact a b, neonExact a b rb, by omega, by omega⟩
  · intro v h
    refine ⟨?_, neonExact v v h⟩
    have range := clipRange v
    have product := productBounds range ⟨by omega, range.2⟩
    have lo := Int.mul_nonneg range.1 range.1
    unfold x86Phase scalarPhase scalarBase sat8
    rw [Int.mul_assoc]
    omega

def ShiftedLanesContract : Prop :=
  ∀ v : Int,
    (BitVec.ofInt 16 (64 * clip510 v)).toInt = 64 * clip510 v ∧
    (BitVec.ofInt 16 (32 * clip510 v)).toInt = 32 * clip510 v

theorem shiftedLanesContract_proven : ShiftedLanesContract := by
  intro v
  have h := clipRange v
  constructor <;> apply BitVec.toInt_ofInt_eq_self (by decide) <;> simp <;> omega

def addRows (bias : BitVec 16) (weight : Nat → BitVec 16) : Nat → BitVec 16
  | 0 => bias
  | n + 1 => addRows bias weight n + weight n

def sumRows (bias : Int) (weight : Nat → Int) : Nat → Int
  | 0 => bias
  | n + 1 => sumRows bias weight n + weight n

def WrappingRowsContract : Prop :=
  ∀ bias weight count,
    addRows (BitVec.ofInt 16 bias) (fun i => BitVec.ofInt 16 (weight i)) count =
      BitVec.ofInt 16 (sumRows bias weight count)

theorem wrappingRowsContract_proven : WrappingRowsContract := by
  intro bias weight count
  induction count with
  | zero => rfl
  | succ n ih => rw [addRows, ih, sumRows, BitVec.ofInt_add]

def addGroups (bias : BitVec 16) (w : Nat → BitVec 16) : Nat → BitVec 16
  | 0 => bias
  | n + 1 =>
    addGroups bias w n + ((w (4 * n) + w (4 * n + 1)) + (w (4 * n + 2) + w (4 * n + 3)))

def GroupedAdditionContract : Prop :=
  ∀ bias weights groups, addGroups bias weights groups = addRows bias weights (4 * groups)

theorem groupedAdditionContract_proven : GroupedAdditionContract := by
  intro bias weights groups
  induction groups with
  | zero => rfl
  | succ n ih =>
    rw [addGroups, ih, show 4 * (n + 1) = 4 * n + 1 + 1 + 1 + 1 by omega]
    rw [addRows, addRows, addRows, addRows]
    simp only [BitVec.add_assoc]

end NeuralReversi.Eval.InputLayer

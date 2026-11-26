import NeuralReversi.Eval.PatternUpdate.Tables

namespace NeuralReversi.Eval.PatternUpdate

inductive ByteOrder where
  | little
  | big
  deriving DecidableEq

def laneBytes (order : ByteOrder) (value : Fin 65536) : Nat × Nat :=
  let low := value.val % 256
  let high := value.val / 256
  match order with
  | .little => (low, high)
  | .big => (high, low)

def buildLaneBytes (host target : ByteOrder) (value : Fin 65536) : Nat × Nat :=
  let bytes := laneBytes host value
  if host = target then bytes else bytes.swap

def readLane (order : ByteOrder) (bytes : Nat × Nat) : Nat :=
  match order with
  | .little => bytes.1 + 256 * bytes.2
  | .big => bytes.2 + 256 * bytes.1

def TableByteOrderContract : Prop :=
  ∀ host target value,
    let bytes := buildLaneBytes host target value
    bytes = laneBytes target value ∧ bytes.1 < 256 ∧ bytes.2 < 256 ∧
    readLane target bytes = value.val

theorem tableByteOrderContract_proven : TableByteOrderContract := by
  intro host target value
  have expected : buildLaneBytes host target value = laneBytes target value := by
    cases host <;> cases target <;> rfl
  let n := value.val
  have fits : n < 65536 := value.isLt
  have low := Nat.mod_lt n (by decide : 0 < 256)
  have reassemble := Nat.mod_add_div n 256
  refine ⟨expected, ?_⟩
  rw [expected]
  cases target <;> change (_ < 256 ∧ _ < 256 ∧ _) <;>
    simp only [laneBytes, readLane] <;> omega

def tableByteOffset (chunk mask lane : Nat) : Nat := (chunk * 65536 + mask) * 64 + lane * 2

def TableByteOffsetContract : Prop :=
  ∀ chunk mask lane, chunk < 4 → mask < 65536 → lane < 32 →
    let address := tableByteOffset chunk mask lane
    address + 1 < 16777216 ∧ address % 2 = 0 ∧
    address / 64 / 65536 = chunk ∧ address / 64 % 65536 = mask ∧ address % 64 / 2 = lane

theorem tableByteOffsetContract_proven : TableByteOffsetContract := by
  intro chunk mask lane hc hm hl
  unfold tableByteOffset
  omega

end NeuralReversi.Eval.PatternUpdate

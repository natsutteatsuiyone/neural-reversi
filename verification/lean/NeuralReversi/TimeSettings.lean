import Std

namespace NeuralReversi.TimeSettings

abbrev Positive := {n : Nat // 0 < n}

inductive Settings where
  | infinite
  | byoyomi (perMove : Nat)
  | fischer (bank increment : Nat)
  | japaneseByo (bank period : Nat) (stones : Positive)

inductive Mode where
  | infinite
  | byoyomi (perMove : Nat)
  | fischer (bank increment : Nat)
  | japaneseByo (bank perMove : Nat)

def maxMillis : Nat := 2 ^ 64 - 1

def millis (seconds : Nat) : Nat := min (seconds * 1000) maxMillis

def fromGtpSecs (main extra stones : Nat) : Settings :=
  let bank := millis main
  let period := millis extra
  if bank = 0 ∧ period = 0 then .infinite
  else if bank = 0 ∧ stones = 0 then .byoyomi period
  else if positive : 0 < stones ∧ 0 < period then .japaneseByo bank period ⟨stones, positive.1⟩
  else .fischer bank period

def enabled : Settings → Bool
  | .infinite => false
  | _ => true

def gtpFields : Settings → Nat × Nat × Nat
  | .infinite => (0, 0, 0)
  | .byoyomi perMove => (0, perMove / 1000, 0)
  | .fischer bank increment => (bank / 1000, increment / 1000, 0)
  | .japaneseByo bank period stones => (bank / 1000, period / 1000, stones.val)

def initialMode : Settings → Mode
  | .infinite => .infinite
  | .byoyomi perMove => .byoyomi perMove
  | .fischer bank increment => .fischer bank increment
  | .japaneseByo bank period stones => .japaneseByo bank (period / stones.val)

def moveMode (remaining stonesLeft : Nat) : Settings → Mode
  | .infinite => .infinite
  | .byoyomi perMove => .byoyomi perMove
  | .fischer _ increment => .fischer remaining increment
  | .japaneseByo _ period stones =>
    if stonesLeft = 0 then .japaneseByo remaining (period / stones.val)
    else .japaneseByo 0 (remaining / stonesLeft)

theorem millis_zero_iff (seconds : Nat) : millis seconds = 0 ↔ seconds = 0 := by
  unfold millis maxMillis
  omega

theorem millis_div (seconds : Nat) :
    millis seconds / 1000 = min seconds (maxMillis / 1000) := by
  unfold millis
  omega

def ConversionContract : Prop :=
  ∀ main extra stones,
    let settings := fromGtpSecs main extra stones
    (enabled settings = false ↔ main = 0 ∧ extra = 0) ∧
    gtpFields settings =
      (min main (maxMillis / 1000), min extra (maxMillis / 1000), if extra = 0 then 0 else stones)

theorem conversionContract_proven : ConversionContract := by
  intro main extra stones settings
  by_cases mainZero : main = 0 <;> by_cases extraZero : extra = 0 <;>
    by_cases noStones : stones = 0 <;>
      simp_all [settings, fromGtpSecs, enabled, gtpFields, millis_zero_iff, millis_div,
        Nat.pos_iff_ne_zero]

def OvertimeContract : Prop :=
  ∀ (bank period remaining stonesLeft : Nat) (stones : Positive),
    let settings := Settings.japaneseByo bank period stones
    initialMode settings = .japaneseByo bank (period / stones.val) ∧
    (stonesLeft = 0 → moveMode remaining stonesLeft settings = .japaneseByo remaining (period / stones.val)) ∧
    (0 < stonesLeft →
      moveMode remaining stonesLeft settings = .japaneseByo 0 (remaining / stonesLeft) ∧
        stonesLeft * (remaining / stonesLeft) + remaining % stonesLeft = remaining ∧
        remaining % stonesLeft < stonesLeft)

theorem overtimeContract_proven : OvertimeContract := by
  intro bank period remaining stonesLeft stones settings
  refine ⟨rfl, ?_, ?_⟩
  · intro empty
    simp [moveMode, settings, empty]
  · intro active
    refine ⟨by simp [moveMode, settings, Nat.ne_of_gt active], ?_, Nat.mod_lt _ active⟩
    simpa only [Nat.add_comm] using Nat.mod_add_div remaining stonesLeft

end NeuralReversi.TimeSettings

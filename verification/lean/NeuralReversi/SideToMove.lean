import Lean

namespace NeuralReversi.SideToMove

inductive Side where
  | player
  | opponent
  deriving DecidableEq

def switch : Side → Side
  | .player => .opponent
  | .opponent => .player

def switchAfter (initial : Side) : Nat → Side
  | 0 => initial
  | n + 1 => switch (switchAfter initial n)

private theorem switch_twice (side : Side) : switch (switch side) = side := by
  cases side <;> rfl

def SwitchContract : Prop :=
  ∀ initial, switch initial ≠ initial ∧
    ∀ count, switchAfter initial count = if count % 2 = 0 then initial else switch initial

theorem switchContract_proven : SwitchContract := by
  intro initial
  refine ⟨by cases initial <;> decide, ?_⟩
  intro count
  induction count with
  | zero => rfl
  | succ n ih =>
    rw [switchAfter, ih]
    by_cases even : n % 2 = 0
    · have odd : (n + 1) % 2 ≠ 0 := by omega
      simp [even, odd]
    · have nextEven : (n + 1) % 2 = 0 := by omega
      simp [even, nextEven, switch_twice]

end NeuralReversi.SideToMove

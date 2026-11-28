module
public import NeuralReversi.EndgameProgress

@[expose] public section

namespace NeuralReversi.RootDispatch

inductive Phase where
  | midgame (maxDepth : Nat)
  | endgame (last : Nat)
  deriving DecidableEq

def fixedPhase (mid cap : Nat) (depthFor : Nat → Nat) (empties : Nat) : Phase :=
  match EndgameProgress.endSelectivity cap depthFor empties with
  | none => .midgame mid
  | some last => .endgame last

def timedPhase (empties : Nat) (tracked : Option Nat) : Phase :=
  if empties ≤ 14 ∨ tracked.any (fun start => decide (empties ≤ start)) then .endgame 5
  else .midgame empties

def updateTracking (empties : Nat) (tracked : Option Nat) (endgame : Bool) (selectivity : Nat) :
    Option Nat :=
  match tracked.filter (fun start => decide (empties ≤ start)) with
  | some start => some start
  | none => if 0 < empties ∧ endgame = true ∧ 2 ≤ selectivity then some (empties - 1) else none

def DispatchContract : Prop :=
  (∀ mid cap depthFor empties,
    fixedPhase mid cap depthFor empties = .midgame mid ↔ cap < 1 ∨ depthFor 1 < empties) ∧
  (∀ mid cap depthFor empties last,
    fixedPhase mid cap depthFor empties = .endgame last ↔
      EndgameProgress.endSelectivity cap depthFor empties = some last) ∧
  (∀ empties tracked,
    timedPhase empties tracked = .endgame 5 ↔
      empties ≤ 14 ∨ ∃ start, tracked = some start ∧ empties ≤ start) ∧
  (∀ empties tracked,
    timedPhase empties tracked = .midgame empties ↔
      14 < empties ∧ ∀ start, tracked = some start → start < empties)

theorem dispatchContract_proven : DispatchContract := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro mid cap depthFor empties
    rw [← EndgameProgress.endSelectivity_eq_none_iff]
    cases choice : EndgameProgress.endSelectivity cap depthFor empties <;> simp [fixedPhase, choice]
  · intro mid cap depthFor empties last
    cases choice : EndgameProgress.endSelectivity cap depthFor empties <;> simp [fixedPhase, choice]
  · intro empties tracked
    by_cases near : empties ≤ 14 <;> cases tracked <;> simp [timedPhase, near]
  · intro empties tracked
    cases tracked <;> simp [timedPhase, not_or, Nat.not_le]

def TrackingContract : Prop :=
  (∀ empties start endgame selectivity, empties ≤ start →
    updateTracking empties (some start) endgame selectivity = some start) ∧
  (∀ empties tracked endgame selectivity,
    (∀ start, tracked = some start → start < empties) →
    updateTracking empties tracked endgame selectivity =
      (if 0 < empties ∧ endgame = true ∧ 2 ≤ selectivity then some (empties - 1) else none)) ∧
  (∀ empties tracked endgame selectivity start, empties ≤ 60 →
    updateTracking empties tracked endgame selectivity = some start →
    (∀ old, tracked = some old → old < empties) →
    start < 60 ∧ timedPhase start (some start) = .endgame 5)

theorem trackingContract_proven : TrackingContract := by
  have expired (empties : Nat) (tracked : Option Nat) (endgame : Bool) (selectivity : Nat)
      (old : ∀ start, tracked = some start → start < empties) :
      updateTracking empties tracked endgame selectivity =
        (if 0 < empties ∧ endgame = true ∧ 2 ≤ selectivity then some (empties - 1) else none) := by
    cases tracked with
    | none => rfl
    | some start =>
      have passed := old start rfl
      simp [updateTracking, Option.filter, Nat.not_le_of_gt passed]
  refine ⟨?_, expired, ?_⟩
  · intro empties start endgame selectivity within
    simp [updateTracking, Option.filter, within]
  · intro empties tracked endgame selectivity start valid recorded old
    rw [expired _ _ _ _ old] at recorded
    split at recorded
    · cases recorded
      exact ⟨by omega, (dispatchContract_proven.2.2.1 _ _).mpr (.inr ⟨_, rfl, Nat.le_refl _⟩)⟩
    · cases recorded

end NeuralReversi.RootDispatch

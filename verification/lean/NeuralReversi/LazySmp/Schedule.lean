module

@[expose] public section

namespace NeuralReversi.LazySmp

structure Pattern where
  size : Nat
  phase : Nat

def patterns : Vector Pattern 20 :=
  #v[⟨1, 0⟩, ⟨1, 1⟩, ⟨2, 0⟩, ⟨2, 1⟩, ⟨2, 2⟩, ⟨2, 3⟩,
     ⟨3, 0⟩, ⟨3, 1⟩, ⟨3, 2⟩, ⟨3, 3⟩, ⟨3, 4⟩, ⟨3, 5⟩,
     ⟨4, 0⟩, ⟨4, 1⟩, ⟨4, 2⟩, ⟨4, 3⟩, ⟨4, 4⟩, ⟨4, 5⟩, ⟨4, 6⟩, ⟨4, 7⟩]

def pattern (helper : Nat) : Pattern := patterns[(helper - 1) % 20]'(Nat.mod_lt _ (by decide))

def Pattern.skips (p : Pattern) (depthPly : Nat) : Bool := (depthPly + p.phase) / p.size % 2 == 1

def skipsDepth (helper depthPly : Nat) : Bool := (pattern helper).skips depthPly

private theorem pattern_bounds (helper : Nat) :
    0 < (pattern helper).size ∧ (pattern helper).size ≤ 4 ∧ (pattern helper).phase ≤ 7 := by
  let slot : Fin 20 := ⟨(helper - 1) % 20, Nat.mod_lt _ (by decide)⟩
  have bounds : ∀ i : Fin 20, 0 < patterns[i].size ∧ patterns[i].size ≤ 4 ∧ patterns[i].phase ≤ 7 := by decide
  exact bounds slot

private theorem pattern_repeat {helper : Nat} (positive : 0 < helper) :
    pattern (helper + 20) = pattern helper := by
  have index : helper + 20 - 1 = (helper - 1) + 20 := by omega
  simp [pattern, index]

private theorem opposite {p : Pattern} (positive : 0 < p.size) (depthPly : Nat) :
    p.skips (depthPly + p.size) = !p.skips depthPly := by
  unfold Pattern.skips
  rw [show depthPly + p.size + p.phase = depthPly + p.phase + p.size by omega,
    Nat.add_div_right _ positive]
  by_cases odd : (depthPly + p.phase) / p.size % 2 = 1
  · simp [odd, show ((depthPly + p.phase) / p.size + 1) % 2 ≠ 1 by omega]
  · simp [odd, show ((depthPly + p.phase) / p.size + 1) % 2 = 1 by omega]

def ScheduleContract : Prop :=
  ∀ (helper depth ply empties : Nat), 0 < helper → 0 < depth → depth ≤ empties → ply + empties = 60 →
    0 < (pattern helper).size ∧ (pattern helper).size ≤ 4 ∧
    depth + ply + (pattern helper).phase ≤ 67 ∧
    skipsDepth (helper + 20) (depth + ply) = skipsDepth helper (depth + ply) ∧
    ∃ offset ≤ 4, skipsDepth helper (depth + ply + offset) = false

theorem scheduleContract_proven : ScheduleContract := by
  intro helper depth ply empties h _ d e
  obtain ⟨pos, size, phase⟩ := pattern_bounds helper
  refine ⟨pos, size, by omega, by simp [skipsDepth, pattern_repeat h], ?_⟩
  cases step : skipsDepth helper (depth + ply)
  · exact ⟨0, by decide, step⟩
  · refine ⟨(pattern helper).size, size, ?_⟩
    change (pattern helper).skips (depth + ply) = true at step
    change (pattern helper).skips (depth + ply + (pattern helper).size) = false
    rw [opposite pos, step]
    rfl

def CoverageContract : Prop :=
  ∀ depthPly : Nat, skipsDepth 1 depthPly = !(skipsDepth 2 depthPly)

theorem coverageContract_proven : CoverageContract := by
  intro depthPly
  change (depthPly / 1 % 2 == 1) = !((depthPly + 1) / 1 % 2 == 1)
  simp only [Nat.div_one]
  by_cases odd : depthPly % 2 = 1
  · simp [odd, show (depthPly + 1) % 2 ≠ 1 by omega]
  · simp [odd, show (depthPly + 1) % 2 = 1 by omega]

def eligibleDepths (helper rootPly maxDepth : Nat) : List Nat :=
  (List.range' 1 maxDepth).filter fun depth => !skipsDepth helper (depth + rootPly)

theorem eligibleDepths_spec (helper rootPly maxDepth depth : Nat) :
    depth ∈ eligibleDepths helper rootPly maxDepth ↔
      1 ≤ depth ∧ depth ≤ maxDepth ∧ skipsDepth helper (depth + rootPly) = false := by
  simp only [eligibleDepths, List.mem_filter, List.mem_range', Nat.one_mul, Bool.not_eq_true']
  constructor
  · rintro ⟨⟨i, hi, equals⟩, s⟩; exact ⟨by omega, by omega, s⟩
  · rintro ⟨h, d, s⟩; exact ⟨⟨depth - 1, by omega, by omega⟩, s⟩

def GapContract : Prop :=
  ∀ (helper depth ply maxDepth empties : Nat), 0 < helper → 0 < depth →
    depth + 4 ≤ maxDepth → maxDepth ≤ empties → ply + empties = 60 →
    ∃ next ∈ eligibleDepths helper ply maxDepth, depth ≤ next ∧ next ≤ depth + 4

theorem gapContract_proven : GapContract := by
  intro helper depth ply maxDepth empties h d enough remaining board
  obtain ⟨_, _, _, _, offset, off, visited⟩ :=
    scheduleContract_proven helper depth ply empties h d (by omega) board
  refine ⟨depth + offset, (eligibleDepths_spec ..).2 ⟨by omega, by omega, ?_⟩, by omega, by omega⟩
  rwa [show depth + offset + ply = depth + ply + offset by omega]

example : eligibleDepths 2 1 5 = [2, 4] := by decide

end NeuralReversi.LazySmp

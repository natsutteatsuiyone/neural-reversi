import NeuralReversi.Solve2.Control

namespace NeuralReversi.LateMoveReduction

structure Node where
  depth : Nat
  moveCount : Nat
  moves : Nat
  endgame : Bool
  pv : Bool
  selective : Bool

def disabled (n : Node) : Bool :=
  n.endgame || n.pv || !n.selective || n.depth < 5 || n.moveCount ≤ 2 || n.moves < 4

def maxReduction (depth : Nat) : Nat :=
  min (max (depth / 3) 1) (depth - 2)

def reduction (n : Node) (proposed : Int) : Nat :=
  if disabled n then 0 else
    min proposed.toNat (maxReduction n.depth)

abbrev Search := Nat → Int → Int → Bool → Int

def scout (search : Search) (depth r : Nat) (alpha : Int) (cut : Bool) : Int :=
  -search (depth - 1 - r) (-(alpha + 1)) (-alpha) (decide (0 < r) || !cut)

def searchMove (search : Search) (depth r : Nat) (alpha : Int) (cut : Bool) : Int :=
  let first := scout search depth r alpha cut
  if 0 < r ∧ alpha < first then
    -search (depth - 1) (-(alpha + 1)) (-alpha) (!cut)
  else first

def Contract : Prop :=
  ∀ (n : Node) (proposed : Int), 0 < n.depth →
    let r := reduction n proposed
    let child := n.depth - 1 - r
    (n.depth < 2^32 → maxReduction n.depth < 2^31) ∧
    (disabled n = true → r = 0) ∧
    r ≤ n.depth / 3 ∧ child + r + 1 = n.depth ∧ child < n.depth ∧ (0 < r → 0 < child) ∧
    ∀ (search : Search) (alpha : Int) (cut : Bool),
      let first := scout search n.depth r alpha cut
      let fullChild := search (n.depth - 1) (-(alpha + 1)) (-alpha) (!cut)
      let result := searchMove search n.depth r alpha cut
      (first ≤ alpha → result = first) ∧
      (alpha < first → result = -fullChild) ∧
      (alpha < result → result = -fullChild ∧
        ∀ value, Solve2.NWS (-(alpha + 1)) value fullChild → result ≤ -value)

theorem contract_proven : Contract := by
  intro n proposed positive
  dsimp only
  let r := reduction n proposed
  have bounds : r ≤ n.depth / 3 ∧ n.depth - 1 - r + r + 1 = n.depth ∧
      n.depth - 1 - r < n.depth ∧ (0 < r → 0 < n.depth - 1 - r) := by
    dsimp [r, reduction, maxReduction]
    split
    · omega
    · rename_i enabled
      have minDepth : 5 ≤ n.depth := by
        simp [disabled] at enabled
        omega
      omega
  refine ⟨by simp only [maxReduction]; omega, fun h => by simp [reduction, h], bounds.1, bounds.2.1, bounds.2.2.1,
    bounds.2.2.2, ?_⟩
  intro search alpha cut
  generalize reduction n proposed = k
  by_cases zero : k = 0
  · simp [zero, searchMove, scout]
    intro high value sound
    have upper := sound.1 (by omega)
    omega
  · have pos : 0 < k := Nat.pos_of_ne_zero zero
    by_cases high : alpha < scout search n.depth k alpha cut
    · simp [searchMove, pos, high, Int.not_le.mpr high]
      intro above value sound
      have upper := sound.1 (by omega)
      omega
    · simp [searchMove, pos, high]

end NeuralReversi.LateMoveReduction

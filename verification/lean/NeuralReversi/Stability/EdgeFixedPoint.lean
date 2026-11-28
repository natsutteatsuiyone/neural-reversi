module
public import NeuralReversi.Stability.EdgeRecursiveSafety

@[expose] public section

namespace NeuralReversi.Stability.Edge
set_option maxRecDepth 100000
set_option maxHeartbeats 0

theorem stableFuel_full (fuel : Nat) (p o : Byte) (full : ~~~(p ||| o) = 0) :
    stableFuel fuel p o = p := by
  cases fuel <;> simp [stableFuel, findWith, full]

private theorem full_of_cpop_finite : ∀ n : Fin 256,
    (BitVec.ofNat 8 n.val).cpop.toNat = 8 → ~~~(BitVec.ofNat 8 n.val) = 0 := by decide

theorem full_of_cpop (bits : Byte) (count : bits.cpop.toNat = 8) : ~~~bits = 0 := by
  have h := full_of_cpop_finite ⟨bits.toNat, bits.isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at h
  exact h count

theorem stableFuel_sufficient (fuel extra : Nat) (p o : Byte)
    (valid : p &&& o = 0) (enough : 8 ≤ (p ||| o).cpop.toNat + fuel) :
    stableFuel (fuel + extra) p o = stableFuel fuel p o := by
  induction fuel generalizing extra p o with
  | zero =>
      have bound := BitVec.toNat_cpop_le (p ||| o)
      have full := full_of_cpop (p ||| o) (by omega)
      simp only [Nat.zero_add, stableFuel_full _ _ _ full]
  | succ fuel ih =>
      have eq : (fuel + 1) + extra = (fuel + extra) + 1 := by omega
      rw [eq]
      change findWith (stableFuel (fuel + extra)) p o = findWith (stableFuel fuel) p o
      apply findWith_congr
      intro x empty
      have first : Step (p,o) (firstChild p o x) := ⟨x, empty, Or.inl rfl⟩
      have second : Step (p,o) (secondChild p o x) := ⟨x, empty, Or.inr rfl⟩
      have hp := step_progress valid first
      have ho := step_progress valid second
      exact ⟨ih extra _ _ hp.1 (by dsimp only at hp; omega),
        ih extra _ _ ho.1 (by dsimp only at ho; omega)⟩

theorem recursiveStable_fixedpoint (p o : Byte) (valid : p &&& o = 0) :
    findWith recursiveStable p o = recursiveStable p o := by
  change stableFuel (8 + 1) p o = stableFuel 8 p o
  exact stableFuel_sufficient 8 1 p o valid (by omega)

theorem source_update_eq_fixedpoint (p o : Byte) (table : EdgeTable) (valid : p &&& o = 0)
    (children : ∀ x : Fin 8, emptyAt p o x = true →
      edgeStability table (firstChild p o x).1 (firstChild p o x).2 =
        recursiveStable (firstChild p o x).1 (firstChild p o x).2 ∧
      edgeStability table (secondChild p o x).1 (secondChild p o x).2 =
        recursiveStable (secondChild p o x).1 (secondChild p o x).2) :
    findEdgeStable p o table = recursiveStable p o := by
  rw [findEdgeStable, findWith_congr _ recursiveStable p o children]
  exact recursiveStable_fixedpoint p o valid

end NeuralReversi.Stability.Edge

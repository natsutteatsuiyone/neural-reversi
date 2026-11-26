import NeuralReversi.EndgameSearch.CacheMir.Budgets

namespace NeuralReversi.EndgameSearch.CacheMir

/-- Takes the byte budgets as arguments so that tuning them only needs regeneration. -/
private theorem shared_fits (width total threads floor : Nat) (wordSize : 32 ≤ width)
    (small : total < 2^32) (nonempty : 18 ≤ floor) (bounded : floor < 2^32) :
    0 < max (total / max threads 1) floor / 18 ∧ max (total / max threads 1) floor / 18 < 2^64 ∧
      max (total / max threads 1) floor / 18 ≤ 2^width := by
  have native : 2^32 ≤ 2^width := Nat.pow_le_pow_right (by decide) wordSize
  have := Nat.div_le_self total (max threads 1)
  omega

theorem budget_fits (threads width : Nat) (wordSize : 32 ≤ width) :
    Fits width (fun t => budget threads t / 18) := by
  have native : 2^32 ≤ 2^width := Nat.pow_le_pow_right (by decide) wordSize
  intro t
  cases t <;> simp only [budget]
  all_goals first
    | omega
    | exact shared_fits width _ threads _ wordSize (by decide) (by decide) (by decide)

theorem initialize_threads (width : Nat) (wordSize : 32 ≤ width)
    (poolSize : Nat → Nat) (key : Board.Board → BitVec 64) :
    let bytes := fun owner t => budget (poolSize owner) t
    (∀ n t, newCache (bytes n t) = some (CacheState.empty (fun t => bytes n t / 18) t)) ∧
      Aligned width (fun n t => bytes n t / 18) key (fun _ => [])
        (fun n => CacheState.empty (fun t => bytes n t / 18)) := by
  exact ⟨fun n t => newCache_eq _ (budget_fits _ width wordSize t).1,
    allocated_owned _ _ _ fun n => budget_fits _ width wordSize⟩

end NeuralReversi.EndgameSearch.CacheMir

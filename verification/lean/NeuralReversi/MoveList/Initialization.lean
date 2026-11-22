import NeuralReversi.MoveList.Capacity

namespace NeuralReversi.MoveList.Initialization
open Capacity

abbrev Memory (α : Type) := Nat → Option α

def write (mem : Memory α) (index : Nat) (value : α) : Memory α :=
  fun i => if i = index then some value else mem i

def scalar (values : Nat → α) : Nat → Nat → Memory α → Memory α
  | 0, _, mem => mem
  | n + 1, start, mem => scalar values n (start + 1) (write mem start (values start))

def paired (values : Nat → α) (tail : Bool) : Nat → Nat → Memory α → Memory α
  | 0, start, mem => if tail then write mem start (values start) else mem
  | p + 1, start, mem =>
      paired values tail p (start + 2)
        (write (write mem start (values start)) (start + 1) (values (start + 1)))

theorem scalar_exact (values : Nat → α) (n start : Nat) (mem : Memory α) (i : Nat) :
    scalar values n start mem i =
      if start ≤ i ∧ i < start + n then some (values i) else mem i := by
  induction n generalizing start mem with
  | zero =>
    have h : ¬ (start ≤ i ∧ i < start + 0) := by omega
    simp only [scalar, ite_eq_right h]
  | succ n ih =>
    rw [scalar, ih]
    unfold write
    by_cases eq : i = start
    · subst i; simp
    · by_cases interval : start + 1 ≤ i ∧ i < start + 1 + n
      · have larger : start ≤ i ∧ i < start + (n + 1) := by omega
        simp [interval, larger]
      · have outside : ¬ (start ≤ i ∧ i < start + (n + 1)) := by omega
        simp [interval, outside, eq]

theorem paired_eq_scalar (values : Nat → α) (tail : Bool) (p start : Nat)
    (mem : Memory α) :
    paired values tail p start mem = scalar values (2 * p + tail.toNat) start mem := by
  induction p generalizing start mem with
  | zero => cases tail <;> rfl
  | succ p ih =>
    rw [paired, ih]
    have arithmetic : 2 * (p + 1) + tail.toNat = (2 * p + tail.toNat) + 2 := by omega
    rw [arithmetic]
    rfl

def pairRun (values : Nat → α) (n : Nat) (mem : Memory α) : Memory α :=
  paired values (decide (n % 2 = 1)) (n / 2) 0 mem

theorem tail_matches_remaining (b : Bitboard.Bits) :
    decide (remaining b (2 * (population b / 2)) ≠ 0) =
      decide (population b % 2 = 1) := by
  have rem := pairs_remainder b
  by_cases zero : remaining b (2 * (population b / 2)) = 0
  · have z := (population_zero_iff _).mpr zero
    have even : population b % 2 ≠ 1 := by omega
    simp [zero, even]
  · have nz : population (remaining b (2 * (population b / 2))) ≠ 0 := by
      intro h; exact zero ((population_zero_iff _).mp h)
    have odd : population b % 2 = 1 := by omega
    simp only [odd, decide_true, decide_eq_true_eq]
    exact zero

theorem pairRun_eq_scalar (values : Nat → α) (n : Nat) (mem : Memory α) :
    pairRun values n mem = scalar values n 0 mem := by
  unfold pairRun
  rw [paired_eq_scalar]
  have arithmetic : 2 * (n / 2) + (decide (n % 2 = 1)).toNat = n := by
    by_cases odd : n % 2 = 1 <;> simp [odd] <;> omega
  rw [arithmetic]

def sourcePairRun (values : Nat → α) (b : Bitboard.Bits) (mem : Memory α) : Memory α :=
  paired values (decide (remaining b (2 * (population b / 2)) ≠ 0))
    (population b / 2) 0 mem

theorem sourcePairRun_eq_scalar (values : Nat → α) (b : Bitboard.Bits) (mem : Memory α) :
    sourcePairRun values b mem = scalar values (population b) 0 mem := by
  unfold sourcePairRun
  rw [tail_matches_remaining]
  exact pairRun_eq_scalar values (population b) mem

theorem scalar_prefix (values : Nat → α) (n : Nat) (mem : Memory α) (i : Nat) :
    scalar values n 0 mem i = if i < n then some (values i) else mem i := by
  simpa using scalar_exact values n 0 mem i

theorem paired_prefix (values : Nat → α) (n : Nat) (mem : Memory α) (i : Nat) :
    pairRun values n mem i = if i < n then some (values i) else mem i := by
  rw [pairRun_eq_scalar]
  exact scalar_prefix values n mem i

theorem initialized_exactly (values : Nat → α) (n i : Nat) :
    pairRun values n (fun _ => none) i ≠ none ↔ i < n := by
  rw [paired_prefix]
  split <;> simp_all

structure Result (α σ : Type) where
  slots : Memory α
  len : Option Nat
  wipeout : Option (Option σ)

def finish (slots : Memory α) (n : Nat) (wipeout : Option σ) : Result α σ :=
  ⟨slots, some n, some wipeout⟩

def scalarReturn (values : Nat → α) (n : Nat) (mem : Memory α) (wipeout : Option σ) :=
  finish (scalar values n 0 mem) n wipeout

def pairReturn (values : Nat → α) (n : Nat) (mem : Memory α) (wipeout : Option σ) :=
  finish (pairRun values n mem) n wipeout

theorem normal_returns_equal (values : Nat → α) (n : Nat) (mem : Memory α)
    (wipeout : Option σ) : pairReturn values n mem wipeout = scalarReturn values n mem wipeout := by
  simp only [pairReturn, scalarReturn, pairRun_eq_scalar]

theorem return_fields (values : Nat → α) (n : Nat) (mem : Memory α) (wipeout : Option σ) :
    (pairReturn values n mem wipeout).len = some n ∧
    (pairReturn values n mem wipeout).wipeout = some wipeout := ⟨rfl, rfl⟩

theorem filtered_initialization (m : LegalMoves.Raw) (initial : LegalMoves.bounded m)
    (fs : List LegalMoves.Filter) (values : Nat → α) (mem : Memory α) :
    let n := population (LegalMoves.applyFilters fs m).bits
    n ≤ 35 ∧
    (∀ i, i < n → i < 35 ∧ pairRun values n mem i = some (values i)) ∧
    (∀ i, n ≤ i → pairRun values n mem i = mem i) := by
  dsimp
  have bound := LegalMoves.filters_bounded fs m initial
  refine ⟨bound, ?_, ?_⟩
  · intro i hi
    refine ⟨Nat.lt_of_lt_of_le hi bound, ?_⟩
    simp [paired_prefix, hi]
  · intro i hi
    simp [paired_prefix, Nat.not_lt_of_ge hi]

example (values : Nat → α) (mem : Memory α) : pairRun values 0 mem = mem := rfl
example (values : Nat → α) (mem : Memory α) :
    pairRun values 1 mem = write mem 0 (values 0) := rfl
example (values : Nat → α) : pairRun values 35 (fun _ => none) 34 = some (values 34) := by
  rw [paired_prefix]; rfl

example (values : Nat → α) : paired values false 17 0 (fun _ => none) 34 = none := by
  rw [paired_eq_scalar, scalar_prefix]
  rfl

end NeuralReversi.MoveList.Initialization

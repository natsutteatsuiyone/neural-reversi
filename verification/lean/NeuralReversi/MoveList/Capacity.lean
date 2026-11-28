module
public import NeuralReversi.LegalMoves.Preservation
public import NeuralReversi.Bitboard.IteratorProperties

@[expose] public section

namespace NeuralReversi.MoveList.Capacity
open Bitboard

def population (b : Bits) : Nat := (count b).toNat

theorem population_zero_iff (b : Bits) : population b = 0 ↔ b = 0 := by
  constructor
  · intro h
    apply (populationCountContract_proven b).2.2.1.mp
    apply BitVec.eq_of_toNat_eq
    exact h
  · intro h
    subst b
    rfl

theorem pop_progress (b : Bits) (h : b ≠ 0) :
    population (clearLsb b) + 1 = population b ∧ b.ctz.toNat < 64 := by
  have bound : b.ctz.toNat < 64 := by
    simpa [BitVec.lt_def] using (BitVec.ctz_lt_iff_ne_zero (x := b)).mpr h
  let sq : NeuralReversi.Square.Square := ⟨b.ctz.toNat, by omega⟩
  have least : IsLeastSetSquare b sq :=
    ⟨bound, BitVec.getLsbD_true_ctz_of_ne_zero h,
      fun _ hi => BitVec.getLsbD_false_of_lt_ctz hi⟩
  have decomp := setIndices_pop b (clearLsb b) sq least (clearLsb_ctz b h)
  have lens := congrArg List.length decomp
  have left := (iteratorEnumerationContract_proven b false).2.2.2.1
  have right := (iteratorEnumerationContract_proven (clearLsb b) false).2.2.2.1
  rw [List.length_cons, left, right] at lens
  exact ⟨lens.symm, bound⟩

def remaining (b : Bits) : Nat → Bits
  | 0 => b
  | k + 1 => clearLsb (remaining b k)

theorem scalar_invariant (b : Bits) (k : Nat) (hk : k ≤ population b) :
    k + population (remaining b k) = population b := by
  induction k with
  | zero => simp [remaining]
  | succ k ih =>
    have prev := ih (by omega)
    have nz : remaining b k ≠ 0 := by
      intro hz
      have z := (population_zero_iff _).mpr hz
      omega
    have step := (pop_progress _ nz).1
    simp only [remaining]
    omega

theorem remaining_nonzero_iff (b : Bits) (k : Nat) (hk : k ≤ population b) :
    remaining b k ≠ 0 ↔ k < population b := by
  have inv := scalar_invariant b k hk
  have zero := population_zero_iff (remaining b k)
  constructor
  · intro nz
    have nonzeroCount : population (remaining b k) ≠ 0 := by
      intro h; exact nz (zero.mp h)
    omega
  · intro lt hz
    have := zero.mpr hz
    omega

theorem scalar_write_safe (b : Bits) (bounded : population b ≤ 35)
    (k : Nat) (active : k < population b) :
    k < 35 ∧ (remaining b k).ctz.toNat < 64 ∧
    (k + 1) + population (remaining b (k + 1)) = population b := by
  have inv := scalar_invariant b k (by omega)
  have nz : remaining b k ≠ 0 := by
    intro hz; have := (population_zero_iff _).mpr hz; omega
  exact ⟨by omega, (pop_progress _ nz).2, scalar_invariant b (k + 1) (by omega)⟩

theorem scalar_finished (b : Bits) : remaining b (population b) = 0 := by
  apply (population_zero_iff _).mp
  have := scalar_invariant b (population b) (by omega)
  omega

theorem pair_write_safe (b : Bits) (bounded : population b ≤ 35)
    (k : Nat) (active : k < population b / 2) :
    2 * k < 35 ∧ 2 * k + 1 < 35 ∧
    (remaining b (2 * k)).ctz.toNat < 64 ∧
    (remaining b (2 * k + 1)).ctz.toNat < 64 ∧
    (2 * k + 2) + population (remaining b (2 * k + 2)) = population b := by
  have first := scalar_write_safe b bounded (2 * k) (by omega)
  have second := scalar_write_safe b bounded (2 * k + 1) (by omega)
  exact ⟨first.1, second.1, first.2.1, second.2.1, second.2.2⟩

theorem pairs_remainder (b : Bits) :
    population (remaining b (2 * (population b / 2))) = population b % 2 := by
  have := scalar_invariant b (2 * (population b / 2)) (by omega)
  omega

theorem odd_tail_safe (b : Bits) (bounded : population b ≤ 35)
    (tail : remaining b (2 * (population b / 2)) ≠ 0) :
    2 * (population b / 2) < 35 ∧
    (remaining b (2 * (population b / 2))).ctz.toNat < 64 ∧
    2 * (population b / 2) + 1 = population b := by
  have rem := pairs_remainder b
  have nz : population (remaining b (2 * (population b / 2))) ≠ 0 := by
    intro h; exact tail ((population_zero_iff _).mp h)
  exact ⟨by omega, (pop_progress _ tail).2, by omega⟩

def pairFinalLen (b : Bits) : Nat :=
  2 * (population b / 2) + if remaining b (2 * (population b / 2)) = 0 then 0 else 1

theorem pair_final_length (b : Bits) : pairFinalLen b = population b := by
  have rem := pairs_remainder b
  unfold pairFinalLen
  split
  · have := (population_zero_iff _).mpr ‹_›; omega
  · rename_i notzero
    have nz : population (remaining b (2 * (population b / 2))) ≠ 0 := by
      intro h; exact notzero ((population_zero_iff _).mp h)
    omega

inductive Dispatch where
  | empty | single | bulk
  deriving DecidableEq, Repr

def dispatch (skipEmpty : Bool) (b : Bits) : Dispatch :=
  if !skipEmpty && b == 0 then .empty
  else if clearLsb b == 0 then .single else .bulk

theorem single_dispatch_safe (skip : Bool) (b : Bits)
    (precondition : skip = true → b ≠ 0) (single : dispatch skip b = .single) :
    population b = 1 ∧ b.ctz.toNat < 64 := by
  have nz : b ≠ 0 := by
    intro hz
    cases skip with
    | false => simp [dispatch, hz] at single
    | true => exact precondition rfl hz
  have cleared : clearLsb b = 0 := by
    by_cases hc : clearLsb b = 0
    · exact hc
    · have bn : b ≠ 0#64 := nz
      have cn : clearLsb b ≠ 0#64 := hc
      simp [dispatch, bn, cn] at single
  have progress := pop_progress b nz
  have zero := (population_zero_iff _).mpr cleared
  exact ⟨by omega, progress.2⟩

theorem nonEmpty_single_dispatch_safe (m : LegalMoves.Raw) (n : LegalMoves.NonEmpty)
    (accepted : LegalMoves.nonEmpty m = some n)
    (single : dispatch true n.val.bits = .single) :
    n.val = m ∧ population m.bits = 1 ∧ m.bits.ctz.toNat < 64 := by
  have same := LegalMoves.nonEmpty_get m n accepted
  have safe := single_dispatch_safe true n.val.bits (fun _ => n.property) single
  rw [same] at safe
  exact ⟨same, safe⟩

theorem filtered_scalar_safe (m : LegalMoves.Raw) (initial : LegalMoves.bounded m)
    (fs : List LegalMoves.Filter) (k : Nat)
    (active : k < population (LegalMoves.applyFilters fs m).bits) :
    k < 35 ∧ ((remaining (LegalMoves.applyFilters fs m).bits k).ctz.toNat < 64) := by
  have bound := LegalMoves.filters_bounded fs m initial
  have safe := scalar_write_safe (LegalMoves.applyFilters fs m).bits bound k active
  exact ⟨safe.1, safe.2.1⟩

theorem filtered_pair_safe (m : LegalMoves.Raw) (initial : LegalMoves.bounded m)
    (fs : List LegalMoves.Filter) (k : Nat)
    (active : k < population (LegalMoves.applyFilters fs m).bits / 2) :
    2 * k < 35 ∧ 2 * k + 1 < 35 ∧
    (remaining (LegalMoves.applyFilters fs m).bits (2 * k)).ctz.toNat < 64 ∧
    (remaining (LegalMoves.applyFilters fs m).bits (2 * k + 1)).ctz.toNat < 64 := by
  have safe := pair_write_safe (LegalMoves.applyFilters fs m).bits
    (LegalMoves.filters_bounded fs m initial) k active
  exact ⟨safe.1, safe.2.1, safe.2.2.1, safe.2.2.2.1⟩

example : dispatch false 0 = .empty := rfl
example : dispatch true 0 = .single := rfl
example : (0 : Bits).ctz.toNat = 64 := trailingZeros_zero
example : pairFinalLen 0x7FFFFFFFF#64 = 35 := rfl
example : 2 * (35 / 2) = 34 := rfl

end NeuralReversi.MoveList.Capacity

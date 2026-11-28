module
public import NeuralReversi.MoveList.Initialization

@[expose] public section

namespace NeuralReversi.MoveList.Concurrent

def advance (len : Nat) (current : BitVec w) : Option (BitVec w) :=
  if current.toNat < len then some (current + 1) else none

def next (len : Nat) (current : BitVec w) : Option (Nat × BitVec w) :=
  (advance len current).map (fun post => (current.toNat, post))

theorem advance_some (len : Nat) {current updated : BitVec w}
    (success : advance len current = some updated) :
    current.toNat < len ∧ updated = current + 1 := by
  by_cases inside : current.toNat < len <;> simp [advance, inside, eq_comm] at success ⊢
  exact success

theorem advance_nat (len : Nat) {current updated : BitVec w}
    (capacity : len < 2 ^ w) (success : advance len current = some updated) :
    updated.toNat = current.toNat + 1 := by
  obtain ⟨inside, rfl⟩ := advance_some len success
  have positive : 1 < 2 ^ w := by omega
  have one : (1 : BitVec w).toNat = 1 := Nat.mod_eq_of_lt positive
  rw [BitVec.toNat_add, one, Nat.mod_eq_of_lt (by omega)]

theorem successful_bounds (len : Nat) (current : BitVec w) (i : Nat) (oneBased : BitVec w)
    (width : 35 < 2 ^ w) (capacity : len ≤ 35)
    (success : next len current = some (i, oneBased)) :
    i < len ∧ oneBased.toNat = i + 1 ∧ 1 ≤ oneBased.toNat ∧ oneBased.toNat ≤ 35 := by
  have inside : current.toNat < len := by
    by_cases h : current.toNat < len
    · exact h
    · simp [next, advance, h] at success
  have same : current.toNat = i ∧ current + 1 = oneBased := by
    simpa [next, advance, inside] using success
  have number := advance_nat len (by omega) (current := current)
    (updated := oneBased) (by
      unfold advance
      rw [ite_eq_left inside]
      exact congrArg some same.2)
  omega

theorem successful_initialized (len : Nat) (current : BitVec w) (i : Nat) (one : BitVec w)
    (mem : Initialization.Memory α) (initialized : ∀ j, j < len → ∃ v, mem j = some v)
    (success : next len current = some (i, one)) : ∃ v, mem i = some v := by
  by_cases inside : current.toNat < len
  · have index : current.toNat = i := by
      have pair : current.toNat = i ∧ current + 1 = one := by
        simpa [next, advance, inside] using success
      exact pair.1
    simpa [index] using initialized current.toNat inside
  · simp [next, advance, inside] at success

inductive Claims (len : Nat) : BitVec w → List (BitVec w) → BitVec w → Prop
  | done : Claims len current [] current
  | take : advance len current = some updated →
      Claims len updated issued final → Claims len current (current :: issued) final

def AtomicClaimsContract : Prop :=
  ∀ (w len : Nat) (start final : BitVec w) (issued : List (BitVec w)),
    len < 2 ^ w → Claims len start issued final →
    issued.map BitVec.toNat = List.range' start.toNat issued.length ∧
    final.toNat = start.toNat + issued.length ∧ issued.length ≤ len - start.toNat

theorem atomicClaimsContract_proven : AtomicClaimsContract := by
  intro w len start final issued capacity trace
  induction trace with
  | done => simp
  | @take current updated issued final success tail ih =>
    have number := advance_nat len capacity success
    have inside := (advance_some len success).1
    obtain ⟨tickets, count, bound⟩ := ih
    constructor
    · simp [List.range'_succ, tickets, number]
    · simp only [List.length_cons]
      omega

theorem Claims.nodup {start final : BitVec w} {issued : List (BitVec w)}
    (trace : Claims len start issued final) (capacity : len < 2 ^ w) :
    (issued.map BitVec.toNat).Nodup := by
  rw [(atomicClaimsContract_proven w len start final issued capacity trace).1]
  exact List.nodup_range'

def ExhaustionContract : Prop :=
  ∀ (w len : Nat) (start final : BitVec w) (issued : List (BitVec w)),
    len < 2 ^ w → Claims len start issued final → advance len final = none →
    issued.map BitVec.toNat = List.range' start.toNat (len - start.toNat) ∧
    (∀ more after, Claims len final more after → more = [] ∧ after = final)

theorem exhaustionContract_proven : ExhaustionContract := by
  intro w len start final issued capacity trace exhausted
  obtain ⟨tickets, count, bound⟩ := atomicClaimsContract_proven w len start final issued capacity trace
  have atEnd : len ≤ final.toNat := by
    by_cases inside : final.toNat < len
    · simp [advance, inside] at exhausted
    · exact Nat.le_of_not_gt inside
  have filled : issued.length = len - start.toNat := by omega
  refine ⟨filled ▸ tickets, ?_⟩
  intro more after further
  cases further with
  | done => exact ⟨rfl, rfl⟩
  | take success _ => rw [exhausted] at success; cases success

def ReturnContract : Prop :=
  ∀ (α : Type) (w len : Nat) (counter : BitVec w) (mem : Initialization.Memory α),
    35 < 2 ^ w → len ≤ 35 → (∀ i, i < len → ∃ v, mem i = some v) →
    (next len counter = none ↔ len ≤ counter.toNat) ∧
    ∀ i ordinal, next len counter = some (i, ordinal) →
      i < len ∧ ordinal.toNat = i + 1 ∧ 1 ≤ ordinal.toNat ∧ ordinal.toNat ≤ 35 ∧
      ∃ v, mem i = some v

theorem returnContract_proven : ReturnContract := by
  intro α w len counter mem width capacity init
  refine ⟨?_, ?_⟩
  · by_cases inside : counter.toNat < len <;> simp [next, advance, inside, Nat.not_le.mpr] <;> omega
  · intro i ordinal success
    obtain ⟨inside, exactNumber, positive, bounded⟩ :=
      successful_bounds len counter i ordinal width capacity success
    exact ⟨inside, exactNumber, positive, bounded, successful_initialized len counter i ordinal mem init success⟩

example : advance 2 (0xFFFFFFFF#32) = none := rfl
example : advance 2 (0xFFFFFFFFFFFFFFFF#64) = none := rfl

end NeuralReversi.MoveList.Concurrent

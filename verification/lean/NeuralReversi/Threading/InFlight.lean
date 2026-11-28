module

@[expose] public section

namespace NeuralReversi.Threading.InFlight

abbrev Hash := BitVec 64

def slot (hash : Hash) : BitVec 16 := hash.setWidth 16
def key (hash : Hash) : Hash := hash ||| 1

def nativeIndex (width : Nat) (hash : Hash) : Nat :=
  (hash.setWidth width).toNat &&& (2 ^ 16 - 1)

private theorem nativeIndex_eq (hash : Hash) {width : Nat} (wide : 16 ≤ width) :
    nativeIndex width hash = (slot hash).toNat := by
  rw [nativeIndex, Nat.and_two_pow_sub_one_eq_mod, BitVec.toNat_setWidth,
    Nat.mod_mod_of_dvd _ (Nat.pow_dvd_pow 2 wide), slot, BitVec.toNat_setWidth]

private theorem key_ne_empty (hash : Hash) : key hash ≠ 0#64 := by
  intro empty
  have bit := congrArg (fun word : Hash => word.getLsbD 0) empty
  simp [key] at bit

private theorem hash_eq {a b : Hash} (sameSlot : slot a = slot b) (sameKey : key a = key b) :
    a = b := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  by_cases zero : i = 0
  · subst i
    simpa [slot] using congrArg (fun word : BitVec 16 => word.getLsbD 0) sameSlot
  · simpa [key, zero] using congrArg (fun word : Hash => word.getLsbD i) sameKey

def EncodingContract : Prop :=
  ∀ hash width, 16 ≤ width →
    nativeIndex width hash = (slot hash).toNat ∧ nativeIndex width hash < 2 ^ 16 ∧ key hash ≠ 0 ∧
    ∀ other, (slot hash = slot other ∧ key hash = key other) ↔ hash = other

theorem encodingContract_proven : EncodingContract := by
  intro hash width wide
  refine ⟨nativeIndex_eq hash wide, ?_, key_ne_empty hash, ?_⟩
  · rw [nativeIndex_eq hash wide]
    exact (slot hash).isLt
  · intro other
    exact ⟨fun h => hash_eq h.1 h.2, fun h => h ▸ ⟨rfl, rfl⟩⟩

inductive Claim where
  | owned
  | busy
  | untracked
  deriving DecidableEq, Repr

def claim (word : Hash) (hash : Hash) : Hash × Claim :=
  if word = 0#64 then (key hash, .owned)
  else if word = key hash then (word, .busy)
  else (word, .untracked)

def ClaimContract : Prop :=
  (∀ hash, claim 0 hash = (key hash, .owned)) ∧
  (∀ owner request, slot owner = slot request →
    claim (key owner) request = (key owner, if owner = request then .busy else .untracked))

theorem claimContract_proven : ClaimContract := by
  refine ⟨fun hash => by simp [claim], ?_⟩
  intro owner request sameSlot
  by_cases equal : owner = request
  · simp [equal, claim, key_ne_empty]
  · have different : key owner ≠ key request := fun same => equal (hash_eq sameSlot same)
    simp [claim, key_ne_empty, equal, different]

def claimChild (endgame : Bool) (depth minDepth : Nat) (tableWord : Option Hash)
    (hash : Hash) : Option Hash × Claim :=
  if endgame && decide (minDepth ≤ depth) then
    match tableWord with
    | none => (none, .untracked)
    | some word =>
      let (next, result) := claim word hash
      (some next, result)
  else (tableWord, .untracked)

def DispatchContract : Prop :=
  ∀ endgame depth minDepth tableWord hash,
    (endgame = false ∨ depth < minDepth ∨ tableWord = none →
      claimChild endgame depth minDepth tableWord hash = (tableWord, .untracked)) ∧
    ∀ word, endgame = true → minDepth ≤ depth → tableWord = some word →
      claimChild endgame depth minDepth tableWord hash =
        (some (claim word hash).1, (claim word hash).2)

theorem dispatchContract_proven : DispatchContract := by
  intro endgame depth minDepth tableWord hash
  constructor
  · rintro (midgame | shallow | absent)
    · simp [claimChild, midgame]
    · simp [claimChild, show ¬ minDepth ≤ depth by omega]
    · simp [claimChild, absent]
  · rintro word rfl deep rfl
    simp [claimChild, deep]

structure Guard where
  id : Nat
  hash : Hash

-- Only successful CAS operations and guard drops modify the slot.
inductive Reachable (index : BitVec 16) : Hash → List Guard → Prop
  | initial : Reachable index 0 []
  | acquire (before : Reachable index 0 guards) (claimant : Guard)
      (atSlot : slot claimant.hash = index) :
      Reachable index (claim 0 claimant.hash).1 (claimant :: guards)
  | release {retiring : Guard} (before : Reachable index word (left ++ retiring :: right)) :
      Reachable index 0 (left ++ right)

def OwnershipContract : Prop :=
  ∀ index word guards, Reachable index word guards →
    (word = 0 ∧ guards = []) ∨
    ∃ guard, guards = [guard] ∧ word = key guard.hash ∧ slot guard.hash = index

theorem ownershipContract_proven : OwnershipContract := by
  intro index word guards reachable
  induction reachable with
  | initial => exact .inl ⟨rfl, rfl⟩
  | acquire _ claimant atSlot ih =>
    rcases ih with ⟨_, rfl⟩ | ⟨owner, _, empty, _⟩
    · exact .inr ⟨claimant, rfl, by simp [claim], atSlot⟩
    · exact (key_ne_empty owner.hash empty.symm).elim
  | release _ ih =>
    rcases ih with ⟨_, empty⟩ | ⟨_, single, _, _⟩
    · have len := congrArg List.length empty
      simp at len
    · have len := congrArg List.length single
      simp only [List.length_append, List.length_cons, List.length_nil] at len
      exact .inl ⟨rfl, List.length_eq_zero_iff.mp (by simp only [List.length_append]; omega)⟩

end NeuralReversi.Threading.InFlight

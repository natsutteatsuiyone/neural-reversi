module

@[expose] public section

namespace NeuralReversi.PublicationMir.ReturnedScores

abbrev Move := Fin 64 × BitVec 32
abbrev Value := Fin 64 × Int

def value (r : Move) : Value := (r.1, r.2.toInt)

/-- Score-bearing fields of the Rust `BestMove` variant, including its first PV move. -/
structure Report where
  best : Move
  moves : List Move
  depth : BitVec 32
  selectivity : Nat
  endgame : Bool
  deriving DecidableEq

def Matches (roots : List Value) (saved : List Move) (depth : Nat) (out : Report) : Prop :=
  (saved.map value).Perm roots ∧ saved.Pairwise (fun a b => b.2.toInt ≤ a.2.toInt) ∧
  saved.head? = some out.best ∧ out.moves = saved ∧ out.depth.toNat = depth ∧
  out.selectivity = 5 ∧ out.endgame = true

instance (roots saved depth out) : Decidable (Matches roots saved depth out) := by
  unfold Matches
  infer_instance

theorem captured (original : List Move) (head : Move) (rest : List Move) (depth : BitVec 32)
    (permutation : (head :: rest).Perm original)
    (sorted : (head :: rest).Pairwise (fun a b => b.2.toInt ≤ a.2.toInt)) :
    Matches (original.map value) (head :: rest) depth.toNat ⟨head,head :: rest,depth,5,true⟩ :=
  ⟨permutation.map value,sorted,rfl,rfl,rfl,rfl,rfl⟩

def Maximum (roots : List Value) (score : Int) : Prop :=
  (∃ r ∈ roots, r.2 = score) ∧ ∀ r ∈ roots, r.2 ≤ score

theorem maximum_unique (a : Maximum roots x) (b : Maximum roots y) : x = y := by
  obtain ⟨ra, ha, rfl⟩ := a.1
  obtain ⟨rb, hb, rfl⟩ := b.1
  exact Int.le_antisymm (b.2 _ ha) (a.2 _ hb)

theorem observed (h : Matches roots saved depth out) :
    Maximum roots out.best.2.toInt ∧ (out.moves.map value).Perm roots := by
  obtain ⟨perm, order, first, moves, _⟩ := h
  rw [moves]
  refine ⟨?_, perm⟩
  cases saved with
  | nil => contradiction
  | cons head tail =>
    have same : head = out.best := Option.some.inj first
    subst head
    constructor
    · exact ⟨value out.best, perm.mem_iff.mp (by simp), rfl⟩
    · intro r member
      obtain ⟨m, seen, rfl⟩ := List.mem_map.mp (perm.mem_iff.mpr member)
      rcases List.mem_cons.mp seen with rfl | rest
      · exact Int.le_refl _
      · exact (List.pairwise_cons.mp order).1 _ rest

end NeuralReversi.PublicationMir.ReturnedScores

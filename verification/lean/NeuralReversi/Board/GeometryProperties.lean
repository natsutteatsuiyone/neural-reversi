import NeuralReversi.Board.Symmetry

namespace NeuralReversi.Board
open Bitboard

def BoardGeometryContract : Prop :=
  ∀ (b : Board) (i : Nat), i < 64 →
    ((flipVertical b).player.getLsbD i = b.player.getLsbD (verticalIndex i) ∧
     (flipVertical b).opponent.getLsbD i = b.opponent.getLsbD (verticalIndex i)) ∧
    ((flipHorizontal b).player.getLsbD i = b.player.getLsbD (horizontalIndex i) ∧
     (flipHorizontal b).opponent.getLsbD i = b.opponent.getLsbD (horizontalIndex i)) ∧
    ((flipDiagA1H8 b).player.getLsbD i = b.player.getLsbD (mainDiagonalIndex i) ∧
     (flipDiagA1H8 b).opponent.getLsbD i = b.opponent.getLsbD (mainDiagonalIndex i)) ∧
    ((flipDiagA8H1 b).player.getLsbD i = b.player.getLsbD (antiDiagonalIndex i) ∧
     (flipDiagA8H1 b).opponent.getLsbD i = b.opponent.getLsbD (antiDiagonalIndex i)) ∧
    ((rotate90 b).player.getLsbD i = b.player.getLsbD (rotate90Index i) ∧
     (rotate90 b).opponent.getLsbD i = b.opponent.getLsbD (rotate90Index i)) ∧
    ((rotate180 b).player.getLsbD i = b.player.getLsbD (rotate180Index i) ∧
     (rotate180 b).opponent.getLsbD i = b.opponent.getLsbD (rotate180Index i)) ∧
    ((rotate270 b).player.getLsbD i = b.player.getLsbD (rotate270Index i) ∧
     (rotate270 b).opponent.getLsbD i = b.opponent.getLsbD (rotate270Index i))

theorem boardGeometryContract_proven : BoardGeometryContract := by
  intro b i within
  exact ⟨⟨vertical_geometry _ i within, vertical_geometry _ i within⟩,
    ⟨horizontal_geometry _ i within, horizontal_geometry _ i within⟩,
    ⟨mainDiagonal_geometry _ i within, mainDiagonal_geometry _ i within⟩,
    ⟨antiDiagonal_geometry _ i within, antiDiagonal_geometry _ i within⟩,
    ⟨rotate90_geometry _ i within, rotate90_geometry _ i within⟩,
    ⟨rotate180_geometry _ i within, rotate180_geometry _ i within⟩,
    ⟨rotate270_geometry _ i within, rotate270_geometry _ i within⟩⟩

def BoardLE (a b : Board) : Prop :=
  a.player.toNat < b.player.toNat ∨
    a.player.toNat = b.player.toNat ∧ a.opponent.toNat ≤ b.opponent.toNat

theorem boardLE_refl (a : Board) : BoardLE a a := by simp [BoardLE]

theorem boardLE_trans {a b c : Board} (ab : BoardLE a b) (bc : BoardLE b c) :
    BoardLE a c := by unfold BoardLE at *; omega

theorem boardLE_antisymm {a b : Board} (ab : BoardLE a b) (ba : BoardLE b a) :
    a = b := by
  have hp : a.player.toNat = b.player.toNat := by unfold BoardLE at *; omega
  have ho : a.opponent.toNat = b.opponent.toNat := by unfold BoardLE at *; omega
  cases a; cases b
  simp_all only [BitVec.toNat_inj]

theorem selectCandidate_bounds (a b : Board) :
    BoardLE (selectCandidate a b) a ∧ BoardLE (selectCandidate a b) b ∧
      (selectCandidate a b = a ∨ selectCandidate a b = b) := by
  by_cases h : tupleLess b a
  · simp only [selectCandidate, h, ↓reduceIte]
    exact ⟨by unfold tupleLess BoardLE at *; omega, boardLE_refl _, by simp⟩
  · simp only [selectCandidate, h, ↓reduceIte]
    exact ⟨boardLE_refl _, by unfold tupleLess BoardLE at *; omega, by simp⟩

theorem fold_candidates_min (xs : List Board) (initial : Board) :
    xs.foldl selectCandidate initial ∈ initial :: xs ∧
      ∀ candidate ∈ initial :: xs, BoardLE (xs.foldl selectCandidate initial) candidate := by
  induction xs generalizing initial with
  | nil => simp [boardLE_refl]
  | cons head tail ih =>
    obtain ⟨member, lower⟩ := ih (selectCandidate initial head)
    obtain ⟨toInitial, toHead, selected⟩ := selectCandidate_bounds initial head
    constructor
    · simp only [List.foldl_cons] at *
      rcases List.mem_cons.mp member with same | intail
      · rcases selected with left | right
        · exact List.mem_cons.mpr (Or.inl (same.trans left))
        · exact List.mem_cons_of_mem _ (List.mem_cons.mpr (Or.inl (same.trans right)))
      · exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ intail)
    · intro candidate member
      simp only [List.foldl_cons]
      rcases List.mem_cons.mp member with same | member
      · subst candidate
        exact boardLE_trans (lower _ (List.mem_cons_self ..)) toInitial
      · rcases List.mem_cons.mp member with same | intail
        · subst candidate
          exact boardLE_trans (lower _ (List.mem_cons_self ..)) toHead
        · exact lower _ (List.mem_cons_of_mem _ intail)

def UniqueMinimumContract : Prop :=
  ∀ b : Board, unique b ∈ b :: symmetryCandidates b ∧
    ∀ candidate ∈ b :: symmetryCandidates b, BoardLE (unique b) candidate

theorem uniqueMinimumContract_proven : UniqueMinimumContract := by
  intro b
  exact fold_candidates_min (symmetryCandidates b) b

theorem orientation_membership (x b : Board) :
    x ∈ b :: symmetryCandidates b ↔ Geometry.InOrbit x b := by
  constructor
  · intro member
    simp only [symmetryCandidates, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with h | h | h | h | h | h | h | h
    · exact ⟨.identity, h⟩
    · exact ⟨.rotate90, h⟩
    · exact ⟨.rotate180, h⟩
    · exact ⟨.rotate270, h⟩
    · exact ⟨.horizontal, h⟩
    · exact ⟨.vertical, h⟩
    · exact ⟨.mainDiagonal, h⟩
    · exact ⟨.antiDiagonal, h⟩
  · rintro ⟨s, rfl⟩
    cases s <;> simp [symmetryCandidates, Geometry.orient, Geometry.word,
      rotate90, rotate180, rotate270, flipVertical, flipHorizontal, flipDiagA1H8,
      flipDiagA8H1, transformPure]

theorem unique_inOrbit (b : Board) : Geometry.InOrbit (unique b) b :=
  (orientation_membership _ b).mp (uniqueMinimumContract_proven b).1

theorem unique_lower (b x : Board) (member : Geometry.InOrbit x b) :
    BoardLE (unique b) x :=
  (uniqueMinimumContract_proven b).2 x ((orientation_membership x b).mpr member)

def UniqueCanonicalContract : Prop :=
  (∀ (s : Geometry.Symmetry) (b : Board), unique (Geometry.orient s b) = unique b) ∧
    ∀ b : Board, unique (unique b) = unique b

theorem uniqueCanonicalContract_proven : UniqueCanonicalContract := by
  exact ⟨Geometry.canonical_invariant unique BoardLE (fun _ _ => boardLE_antisymm)
      unique_inOrbit unique_lower,
    Geometry.canonical_idempotent unique BoardLE (fun _ _ => boardLE_antisymm)
      unique_inOrbit unique_lower⟩

theorem orient_wellFormed (s : Geometry.Symmetry) (b : Board) (valid : wellFormed b) :
    wellFormed (Geometry.orient s b) := by
  change Geometry.word s b.player &&& Geometry.word s b.opponent = 0#64
  have disjoint : b.player &&& b.opponent = 0#64 := valid
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  have original := congrArg (fun bits : Bits => bits.getLsbD (Geometry.index s i)) disjoint
  simpa only [Geometry.word_geometry s _ i hi,
    BitVec.getLsbD_and, BitVec.getLsbD_zero] using original

def GeometrySeparationContract : Prop :=
  ∀ b : Board, wellFormed b →
    (∀ s : Geometry.Symmetry, wellFormed (Geometry.orient s b)) ∧ wellFormed (unique b)

theorem geometrySeparationContract_proven : GeometrySeparationContract := by
  intro b valid
  refine ⟨fun s => orient_wellFormed s b valid, ?_⟩
  obtain ⟨s, hs⟩ := unique_inOrbit b
  rw [hs]
  exact orient_wellFormed s b valid

end NeuralReversi.Board

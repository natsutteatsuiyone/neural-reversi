module
public import NeuralReversi.Bitboard.GeometryProperties
public import NeuralReversi.Board.GeometryModel

@[expose] public section

namespace NeuralReversi.Board.Geometry
open NeuralReversi.Bitboard

inductive Symmetry where
  | identity
  | vertical
  | horizontal
  | mainDiagonal
  | antiDiagonal
  | rotate90
  | rotate180
  | rotate270
  deriving DecidableEq, Repr

def index : Symmetry → Nat → Nat
  | .identity => id
  | .vertical => verticalIndex
  | .horizontal => horizontalIndex
  | .mainDiagonal => mainDiagonalIndex
  | .antiDiagonal => antiDiagonalIndex
  | .rotate90 => rotate90Index
  | .rotate180 => rotate180Index
  | .rotate270 => rotate270Index

def word : Symmetry → Bits → Bits
  | .identity => id
  | .vertical => NeuralReversi.Bitboard.flipVertical
  | .horizontal => NeuralReversi.Bitboard.flipHorizontal
  | .mainDiagonal => NeuralReversi.Bitboard.flipDiagA1H8
  | .antiDiagonal => NeuralReversi.Bitboard.flipDiagA8H1
  | .rotate90 => NeuralReversi.Bitboard.rotate90
  | .rotate180 => NeuralReversi.Bitboard.rotate180
  | .rotate270 => NeuralReversi.Bitboard.rotate270

def compose : Symmetry → Symmetry → Symmetry
  | .identity, .identity => .identity
  | .identity, .vertical => .vertical
  | .identity, .horizontal => .horizontal
  | .identity, .mainDiagonal => .mainDiagonal
  | .identity, .antiDiagonal => .antiDiagonal
  | .identity, .rotate90 => .rotate90
  | .identity, .rotate180 => .rotate180
  | .identity, .rotate270 => .rotate270
  | .vertical, .identity => .vertical
  | .vertical, .vertical => .identity
  | .vertical, .horizontal => .rotate180
  | .vertical, .mainDiagonal => .rotate270
  | .vertical, .antiDiagonal => .rotate90
  | .vertical, .rotate90 => .antiDiagonal
  | .vertical, .rotate180 => .horizontal
  | .vertical, .rotate270 => .mainDiagonal
  | .horizontal, .identity => .horizontal
  | .horizontal, .vertical => .rotate180
  | .horizontal, .horizontal => .identity
  | .horizontal, .mainDiagonal => .rotate90
  | .horizontal, .antiDiagonal => .rotate270
  | .horizontal, .rotate90 => .mainDiagonal
  | .horizontal, .rotate180 => .vertical
  | .horizontal, .rotate270 => .antiDiagonal
  | .mainDiagonal, .identity => .mainDiagonal
  | .mainDiagonal, .vertical => .rotate90
  | .mainDiagonal, .horizontal => .rotate270
  | .mainDiagonal, .mainDiagonal => .identity
  | .mainDiagonal, .antiDiagonal => .rotate180
  | .mainDiagonal, .rotate90 => .vertical
  | .mainDiagonal, .rotate180 => .antiDiagonal
  | .mainDiagonal, .rotate270 => .horizontal
  | .antiDiagonal, .identity => .antiDiagonal
  | .antiDiagonal, .vertical => .rotate270
  | .antiDiagonal, .horizontal => .rotate90
  | .antiDiagonal, .mainDiagonal => .rotate180
  | .antiDiagonal, .antiDiagonal => .identity
  | .antiDiagonal, .rotate90 => .horizontal
  | .antiDiagonal, .rotate180 => .mainDiagonal
  | .antiDiagonal, .rotate270 => .vertical
  | .rotate90, .identity => .rotate90
  | .rotate90, .vertical => .mainDiagonal
  | .rotate90, .horizontal => .antiDiagonal
  | .rotate90, .mainDiagonal => .horizontal
  | .rotate90, .antiDiagonal => .vertical
  | .rotate90, .rotate90 => .rotate180
  | .rotate90, .rotate180 => .rotate270
  | .rotate90, .rotate270 => .identity
  | .rotate180, .identity => .rotate180
  | .rotate180, .vertical => .horizontal
  | .rotate180, .horizontal => .vertical
  | .rotate180, .mainDiagonal => .antiDiagonal
  | .rotate180, .antiDiagonal => .mainDiagonal
  | .rotate180, .rotate90 => .rotate270
  | .rotate180, .rotate180 => .identity
  | .rotate180, .rotate270 => .rotate90
  | .rotate270, .identity => .rotate270
  | .rotate270, .vertical => .antiDiagonal
  | .rotate270, .horizontal => .mainDiagonal
  | .rotate270, .mainDiagonal => .vertical
  | .rotate270, .antiDiagonal => .horizontal
  | .rotate270, .rotate90 => .identity
  | .rotate270, .rotate180 => .rotate90
  | .rotate270, .rotate270 => .rotate180

theorem index_bound (s : Symmetry) (i : Nat) (hi : i < 64) : index s i < 64 := by
  cases s <;> simp only [index, id, verticalIndex, horizontalIndex, mainDiagonalIndex,
    antiDiagonalIndex, rotate90Index, rotate180Index, rotate270Index] <;> omega

theorem word_geometry (s : Symmetry) (bits : Bits) (i : Nat) (hi : i < 64) :
    (word s bits).getLsbD i = bits.getLsbD (index s i) := by
  cases s
  · rfl
  · exact vertical_geometry bits i hi
  · exact horizontal_geometry bits i hi
  · exact mainDiagonal_geometry bits i hi
  · exact antiDiagonal_geometry bits i hi
  · exact rotate90_geometry bits i hi
  · exact rotate180_geometry bits i hi
  · exact rotate270_geometry bits i hi

private def allSymmetries : List Symmetry :=
  [.identity, .vertical, .horizontal, .mainDiagonal, .antiDiagonal, .rotate90, .rotate180, .rotate270]

private theorem mem_allSymmetries (s : Symmetry) : s ∈ allSymmetries := by
  cases s <;> decide

private theorem index_compose_all :
    ∀ s ∈ allSymmetries, ∀ t ∈ allSymmetries, ∀ i : Fin 64,
      index t (index s i.val) = index (compose s t) i.val := by decide +kernel

theorem index_compose (s t : Symmetry) (i : Nat) (hi : i < 64) :
    index t (index s i) = index (compose s t) i :=
  index_compose_all s (mem_allSymmetries s) t (mem_allSymmetries t) ⟨i, hi⟩

theorem word_compose (s t : Symmetry) (bits : Bits) :
    word s (word t bits) = word (compose s t) bits := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  rw [word_geometry s _ i hi, word_geometry t _ _ (index_bound s i hi),
    word_geometry (compose s t) _ i hi, index_compose s t i hi]

def orient (s : Symmetry) (b : Board) : Board := ⟨word s b.player, word s b.opponent⟩

theorem orient_compose (s t : Symmetry) (b : Board) :
    orient s (orient t b) = orient (compose s t) b := by
  simp only [orient, word_compose]

def inverse : Symmetry → Symmetry
  | .rotate90 => .rotate270
  | .rotate270 => .rotate90
  | s => s

theorem compose_surjective (s t : Symmetry) :
    compose (compose s (inverse t)) t = s := by
  cases s <;> cases t <;> rfl

def InOrbit (x b : Board) : Prop := ∃ s, x = orient s b

theorem orbit_orientation (x b : Board) (t : Symmetry) :
    InOrbit x (orient t b) ↔ InOrbit x b := by
  constructor
  · rintro ⟨s, hs⟩
    exact ⟨compose s t, hs.trans (orient_compose s t b)⟩
  · rintro ⟨s, hs⟩
    refine ⟨compose s (inverse t), ?_⟩
    rw [orient_compose, compose_surjective]
    exact hs

theorem canonical_invariant (canonical : Board → Board) (le : Board → Board → Prop)
    (antisymm : ∀ a b, le a b → le b a → a = b)
    (member : ∀ b, InOrbit (canonical b) b)
    (minimum : ∀ b x, InOrbit x b → le (canonical b) x)
    (t : Symmetry) (b : Board) : canonical (orient t b) = canonical b := by
  apply antisymm
  · apply minimum
    exact (orbit_orientation _ b t).mpr (member b)
  · apply minimum
    exact (orbit_orientation _ b t).mp (member (orient t b))

theorem canonical_idempotent (canonical : Board → Board) (le : Board → Board → Prop)
    (antisymm : ∀ a b, le a b → le b a → a = b)
    (member : ∀ b, InOrbit (canonical b) b)
    (minimum : ∀ b x, InOrbit x b → le (canonical b) x)
    (b : Board) : canonical (canonical b) = canonical b := by
  obtain ⟨s, hs⟩ := member b
  rw [hs]
  rw [canonical_invariant canonical le antisymm member minimum]
  exact hs

end NeuralReversi.Board.Geometry

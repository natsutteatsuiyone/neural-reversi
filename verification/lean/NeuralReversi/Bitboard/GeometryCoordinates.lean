module

@[expose] public section

namespace NeuralReversi.Bitboard

def verticalIndex (i : Nat) : Nat := 8 * (7 - i / 8) + i % 8
def horizontalIndex (i : Nat) : Nat := 8 * (i / 8) + (7 - i % 8)
def mainDiagonalIndex (i : Nat) : Nat := 8 * (i % 8) + i / 8
def antiDiagonalIndex (i : Nat) : Nat := 8 * (7 - i % 8) + (7 - i / 8)
def rotate90Index (i : Nat) : Nat := 8 * (7 - i % 8) + i / 8
def rotate180Index (i : Nat) : Nat := 8 * (7 - i / 8) + (7 - i % 8)
def rotate270Index (i : Nat) : Nat := 8 * (i % 8) + (7 - i / 8)

theorem geometryIndices_bounded (i : Nat) (within : i < 64) :
    verticalIndex i < 64 ∧ horizontalIndex i < 64 ∧ mainDiagonalIndex i < 64 ∧
    antiDiagonalIndex i < 64 ∧ rotate90Index i < 64 ∧ rotate180Index i < 64 ∧
    rotate270Index i < 64 := by
  unfold verticalIndex horizontalIndex mainDiagonalIndex antiDiagonalIndex
    rotate90Index rotate180Index rotate270Index
  omega

end NeuralReversi.Bitboard

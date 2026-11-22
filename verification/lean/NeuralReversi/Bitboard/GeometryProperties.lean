import NeuralReversi.Bitboard.GeometryModel
import NeuralReversi.Bitboard.GeometryCoordinates

namespace NeuralReversi.Bitboard

private theorem xorABA (a b : Bool) : (a ^^ (b ^^ a)) = b := by
  cases a <;> cases b <;> rfl

set_option maxRecDepth 4096 in
set_option maxHeartbeats 2000000 in
theorem vertical_geometry (bits : Bits) : ∀ i : Nat, i < 64 →
    (flipVertical bits).getLsbD i = bits.getLsbD (verticalIndex i) := by
  simp only [Nat.forall_lt_succ_right]
  simp only [flipVertical, byteAt, BitVec.getLsbD_append, BitVec.getLsbD_extractLsb']
  simp [verticalIndex]

set_option maxRecDepth 4096 in
set_option maxHeartbeats 2000000 in
theorem horizontal_geometry (bits : Bits) : ∀ i : Nat, i < 64 →
    (flipHorizontal bits).getLsbD i = bits.getLsbD (horizontalIndex i) := by
  simp only [Nat.forall_lt_succ_right]
  simp [flipHorizontal, horizontalIndex]

set_option maxRecDepth 4096 in
set_option maxHeartbeats 2000000 in
theorem mainDiagonal_geometry (bits : Bits) : ∀ i : Nat, i < 64 →
    (flipDiagA1H8 bits).getLsbD i = bits.getLsbD (mainDiagonalIndex i) := by
  simp only [Nat.forall_lt_succ_right]
  simp [flipDiagA1H8, deltaSwapRaw, mainDiagonalIndex, xorABA]

set_option maxRecDepth 4096 in
set_option maxHeartbeats 2000000 in
theorem antiDiagonal_geometry (bits : Bits) : ∀ i : Nat, i < 64 →
    (flipDiagA8H1 bits).getLsbD i = bits.getLsbD (antiDiagonalIndex i) := by
  simp only [Nat.forall_lt_succ_right]
  simp [flipDiagA8H1, deltaSwapRaw, antiDiagonalIndex, xorABA]

theorem rotate90_geometry (bits : Bits) (i : Nat) (within : i < 64) :
    (rotate90 bits).getLsbD i = bits.getLsbD (rotate90Index i) := by
  rw [rotate90, vertical_geometry _ i within,
    antiDiagonal_geometry _ _ (geometryIndices_bounded i within).1]
  congr 1
  unfold antiDiagonalIndex verticalIndex rotate90Index
  omega

theorem rotate180_geometry (bits : Bits) (i : Nat) (within : i < 64) :
    (rotate180 bits).getLsbD i = bits.getLsbD (rotate180Index i) := by
  have index : 63 - i = rotate180Index i := by unfold rotate180Index; omega
  rw [rotate180, BitVec.getLsbD_reverse]
  simp only [BitVec.getMsbD, within, decide_true, Bool.true_and]
  exact congrArg bits.getLsbD index

theorem rotate270_geometry (bits : Bits) (i : Nat) (within : i < 64) :
    (rotate270 bits).getLsbD i = bits.getLsbD (rotate270Index i) := by
  rw [rotate270, vertical_geometry _ i within,
    mainDiagonal_geometry _ _ (geometryIndices_bounded i within).1]
  congr 1
  unfold mainDiagonalIndex verticalIndex rotate270Index
  omega

def GeometryContract : Prop :=
  ∀ (bits : Bits) (i : Nat), i < 64 →
    (flipVertical bits).getLsbD i = bits.getLsbD (verticalIndex i) ∧
    (flipHorizontal bits).getLsbD i = bits.getLsbD (horizontalIndex i) ∧
    (flipDiagA1H8 bits).getLsbD i = bits.getLsbD (mainDiagonalIndex i) ∧
    (flipDiagA8H1 bits).getLsbD i = bits.getLsbD (antiDiagonalIndex i) ∧
    (rotate90 bits).getLsbD i = bits.getLsbD (rotate90Index i) ∧
    (rotate180 bits).getLsbD i = bits.getLsbD (rotate180Index i) ∧
    (rotate270 bits).getLsbD i = bits.getLsbD (rotate270Index i)

theorem geometryContract_proven : GeometryContract := by
  intro bits i within
  exact ⟨vertical_geometry bits i within, horizontal_geometry bits i within,
    mainDiagonal_geometry bits i within, antiDiagonal_geometry bits i within,
    rotate90_geometry bits i within, rotate180_geometry bits i within,
    rotate270_geometry bits i within⟩

end NeuralReversi.Bitboard

import NeuralReversi.Flip.Avx512.Properties
import NeuralReversi.Simd.ZmmModel

namespace NeuralReversi.Flip.Avx512.WidePair
open NeuralReversi.Simd.X86
open NeuralReversi.Flip (mask value Square)

private def shuffleI64x2 (a b : Vec512) (imm : BitVec 8) : Vec512 := fun lane =>
  let block := lane.val / 2
  let sel := (imm.toNat / 2 ^ (2 * block)) % 4
  let index : Fin 8 := ⟨2 * sel + lane.val % 2, by dsimp [sel]; omega⟩
  if block < 2 then a index else b index

private def shuffleDwords (a : Vec512) (imm : BitVec 8) : Vec512 := fun lane =>
  let block : Vec128 := fun j => a ⟨2 * (lane.val / 2) + j.val, by omega⟩
  shuffleEpi32 block imm ⟨lane.val % 2, by omega⟩

private def reducePair (flips : Vec512) : Word × Word :=
  let or64 := or512 flips (shuffleDwords flips 0x4e)
  let reduced := or512 or64 (shuffleI64x2 or64 or64 0xb1)
  (reduced 0, extractI64x4 reduced 1 0)

private theorem reducePair_eq (v : Vec512) :
    reducePair v = ((v 0 ||| v 1) ||| (v 2 ||| v 3), (v 4 ||| v 5) ||| (v 6 ||| v 7)) := by
  let paired := or512 v (shuffleDwords v 0x4e)
  change (paired 0 ||| paired 2, paired 4 ||| paired 6) = _
  simp only [paired, or512, shuffleDwords, shuffleEpi32_4e]
  rfl

def flipPair (x0 x1 : Square) (player opponent : Word) : Word × Word :=
  let left := shuffleI64x2 (mask x0) (mask x1) 0x44
  let right := shuffleI64x2 (mask x0) (mask x1) 0xee
  reducePair fun i => Algebra.combinedMin (left i) (right i) player (~~~opponent)

private def mergeDirections (sq : Square) (player opponent : Word) : Word :=
  let lane := fun (l r : Fin 8) => Algebra.combinedMin (mask sq l) (mask sq r) player (~~~opponent)
  (lane 0 4 ||| lane 1 5) ||| (lane 2 6 ||| lane 3 7)

private theorem mergeDirections_eq (sq : Square) (player opponent : Word) :
    mergeDirections sq player opponent = value sq player opponent := by
  simp only [mergeDirections, Algebra.combinedMin_scalar, value]
  ac_rfl

def PairContract : Prop := ∀ (x0 x1 : Square) (player opponent : Word),
  flipPair x0 x1 player opponent = (value x0 player opponent, value x1 player opponent)

theorem pairContract_proven : PairContract := by
  intro x0 x1 player opponent
  rw [flipPair, reducePair_eq]
  change (mergeDirections x0 player opponent, mergeDirections x1 player opponent) = _
  rw [mergeDirections_eq, mergeDirections_eq]

/-- Cast the low YMM half and insert the high YMM half with immediate 1.
    Every output lane is defined after the insertion. -/
private def insertUpper (low high : Vec256) : Vec512 := fun i =>
  if h : i.val < 4 then low ⟨i.val, h⟩ else high ⟨i.val - 4, by omega⟩

def narrowPair (x0 x1 : Square) (player opponent : Word) : Word × Word :=
  let l0 := loadLeft (squareIndex x0) (squareIndex_valid x0)
  let l1 := loadLeft (squareIndex x1) (squareIndex_valid x1)
  let r0 := loadRight (squareIndex x0) (squareIndex_valid x0)
  let r1 := loadRight (squareIndex x1) (squareIndex_valid x1)
  let left := insertUpper l0 l1
  let right := insertUpper r0 r1
  reducePair fun i => Algebra.combinedMin (left i) (right i) player (~~~opponent)

private theorem loadLeft_eq (sq : Square) (lane : Fin 4) :
    loadLeft (squareIndex sq) (squareIndex_valid sq) lane = mask sq ⟨lane.val, by omega⟩ := by
  simp only [loadLeft, squareIndex_value, mask]

private theorem loadRight_eq (sq : Square) (lane : Fin 4) :
    loadRight (squareIndex sq) (squareIndex_valid sq) lane = mask sq ⟨4 + lane.val, by omega⟩ := by
  simp only [loadRight, squareIndex_value, mask]

theorem narrowPair_eq (x0 x1 : Square) (player opponent : Word) :
    narrowPair x0 x1 player opponent = (value x0 player opponent, value x1 player opponent) := by
  rw [narrowPair, reducePair_eq]
  simp only [loadLeft_eq, loadRight_eq, insertUpper]
  change (mergeDirections x0 player opponent, mergeDirections x1 player opponent) = _
  rw [mergeDirections_eq, mergeDirections_eq]

def singleMin (sq : Square) (player opponent : Word) : Word :=
  let left := loadLeft (squareIndex sq) (squareIndex_valid sq)
  let right := loadRight (squareIndex sq) (squareIndex_valid sq)
  let flips : Vec256 := fun i => Algebra.combinedMin (left i) (right i) player (~~~opponent)
  let half := or128 (cast256Si128 flips) (extractI128 flips 1)
  cvtsi128Si64 (or128 half (shuffleEpi32 half 0x4e))

theorem singleMin_eq (sq : Square) (player opponent : Word) :
    singleMin sq player opponent = value sq player opponent := by
  simp only [singleMin, cvtsi128Si64, shuffleEpi32_4e, or128, cast256Si128, extractI128,
    loadLeft_eq, loadRight_eq]
  let lane := fun (l r : Fin 8) => Algebra.combinedMin (mask sq l) (mask sq r) player (~~~opponent)
  change (lane 0 4 ||| lane 2 6) ||| (lane 1 5 ||| lane 3 7) = _
  rw [← mergeDirections_eq sq player opponent]
  change _ = (lane 0 4 ||| lane 1 5) ||| (lane 2 6 ||| lane 3 7)
  ac_rfl

def triple (a b c : Square) (player opponent : Word) : Word × Word × Word :=
  (singleMin a player opponent, narrowPair b c player opponent)

def quad (a b c d : Square) (player opponent : Word) : Word × Word × Word × Word :=
  let first := narrowPair a b player opponent
  let second := narrowPair c d player opponent
  (first.1, first.2, second.1, second.2)

theorem triple_eq (a b c : Square) (player opponent : Word) :
    triple a b c player opponent =
      (value a player opponent, value b player opponent, value c player opponent) := by
  simp only [triple, singleMin_eq, narrowPair_eq]

theorem quad_eq (a b c d : Square) (player opponent : Word) :
    quad a b c d player opponent =
      (value a player opponent, value b player opponent, value c player opponent, value d player opponent) := by
  simp only [quad, narrowPair_eq]

end NeuralReversi.Flip.Avx512.WidePair

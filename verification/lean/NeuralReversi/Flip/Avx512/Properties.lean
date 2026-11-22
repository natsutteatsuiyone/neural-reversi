import NeuralReversi.Flip.Avx512.Model
import NeuralReversi.Flip.Avx512.Spec
import NeuralReversi.Flip.Avx512.Word
import NeuralReversi.Flip.Properties
import NeuralReversi.Simd.X86Properties

namespace NeuralReversi.Flip.Avx512
open NeuralReversi.Flip
open NeuralReversi.Simd.X86 (horizontalOrU64 horizontalOrU64_exact)

set_option maxRecDepth 8192
set_option maxHeartbeats 2000000

def mask66 (pos : Fin 66) (d : Fin 8) : Word := (lrmask.get pos).get d

theorem mask66_typed (pos : Fin 66) (h : pos.val < 64) (d : Fin 8) :
    mask66 pos d = mask ⟨pos.val,h⟩ d := rfl

theorem mask66_placeholders : ∀ pos : Fin 66, 64 ≤ pos.val → ∀ d : Fin 8, mask66 pos d = 0 := by
  decide

theorem mask66_sentinel : ∀ d : Fin 8, mask66 ⟨65,by decide⟩ d = 0 := by decide

def indexedValue (pos : Fin 66) (player opponent : Word) : Word :=
  let n := ~~~opponent
  Scalar.flipLeft (mask66 pos 0) player n ||| Scalar.flipLeft (mask66 pos 1) player n |||
    Scalar.flipLeft (mask66 pos 2) player n ||| Scalar.flipLeft (mask66 pos 3) player n |||
    Scalar.flipRightValue (mask66 pos 4) player n ||| Scalar.flipRightValue (mask66 pos 5) player n |||
    Scalar.flipRightValue (mask66 pos 6) player n ||| Scalar.flipRightValue (mask66 pos 7) player n

theorem indexedValue_typed (pos : Fin 66) (h : pos.val < 64) (player opponent : Word) :
    indexedValue pos player opponent = value ⟨pos.val,h⟩ player opponent := rfl

theorem indexedValue_placeholder (pos : Fin 66) (hge : 64 ≤ pos.val) (player opponent : Word) :
    indexedValue pos player opponent = 0 := by
  unfold indexedValue
  rw [mask66_placeholders pos hge 0, mask66_placeholders pos hge 1, mask66_placeholders pos hge 2,
    mask66_placeholders pos hge 3, mask66_placeholders pos hge 4, mask66_placeholders pos hge 5,
    mask66_placeholders pos hge 6, mask66_placeholders pos hge 7]
  simp [Scalar.flipLeft, Scalar.flipRightValue]

theorem indexValue_coordinate (pos : Index64) (valid : pos.toNat < 66)
    (player opponent : Word) (i : Nat) :
    (indexedValue ⟨pos.toNat,valid⟩ player opponent).getLsbD i = true ↔
      IndexCoordinateBit pos player opponent i := by
  by_cases hp : pos.toNat < 64
  · rw [indexedValue_typed _ hp]
    simp only [IndexCoordinateBit,hp,↓reduceDIte]
    exact value_coordinate ⟨pos.toNat,hp⟩ player opponent i
  · rw [indexedValue_placeholder ⟨pos.toNat, valid⟩ (by dsimp; omega)]
    simp [IndexCoordinateBit,hp]

theorem index_word_kernel (pos : Index64) (valid : pos.toNat < 66) (player opponent : Word) :
    flipIndex pos valid player opponent = horizontalOrU64 (fun lane =>
      NeuralReversi.Flip.Avx512.Algebra.combinedCmp (loadLeft pos valid lane) (loadRight pos valid lane)
        player (~~~opponent)) := rfl

theorem index_eq_scalarValue (pos : Index64) (valid : pos.toNat < 66)
    (player opponent : Word) :
    flipIndex pos valid player opponent = indexedValue ⟨pos.toNat,valid⟩ player opponent := by
  rw [index_word_kernel,horizontalOrU64_exact]
  simp only [NeuralReversi.Flip.Avx512.Algebra.combinedCmp_scalar]
  simp only [indexedValue,loadLeft,loadRight,mask66]
  ac_rfl

theorem flipIndex_coordinate (pos : Index64) (valid : pos.toNat < 66)
    (player opponent : Word) (i : Nat) :
    (flipIndex pos valid player opponent).getLsbD i = true ↔
      IndexCoordinateBit pos player opponent i := by
  rw [index_eq_scalarValue]
  exact indexValue_coordinate pos valid player opponent i

theorem flip_eq_scalarValue (sq : Square) (player opponent : Word) :
    flip sq player opponent = value sq player opponent := by
  rw [flip,index_eq_scalarValue]
  have hp : (squareIndex sq).toNat < 64 := by rw [squareIndex_value]; exact sq.isLt
  rw [indexedValue_typed _ hp]
  congr 1
  apply Fin.ext
  exact squareIndex_value sq

theorem flip_coordinate (sq : Square) (player opponent : Word) (i : Nat) :
    (flip sq player opponent).getLsbD i = true ↔ CoordinateBit sq player opponent i := by
  rw [flip_eq_scalarValue]
  exact value_coordinate sq player opponent i

theorem flip_eq_scalar (checks : Bool) (sq : Square) (player opponent : Word) :
    NeuralReversi.Flip.flip checks sq player opponent = .ok (flip sq player opponent) := by
  rw [flip_eq_scalarValue]
  exact operational_eq_value checks sq player opponent

def FlipContract : Prop := ∀ sq player opponent i,
  (flip sq player opponent).getLsbD i = true ↔ CoordinateBit sq player opponent i

def FlipIndexContract : Prop := ∀ pos valid player opponent i,
  (flipIndex pos valid player opponent).getLsbD i = true ↔
    IndexCoordinateBit pos player opponent i

theorem flipContract_proven : FlipContract := flip_coordinate
theorem flipIndexContract_proven : FlipIndexContract := flipIndex_coordinate

end NeuralReversi.Flip.Avx512

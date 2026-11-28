module
public import NeuralReversi.Simd.Avx512FlipModel
public import NeuralReversi.Flip.LrmaskModel

@[expose] public section

namespace NeuralReversi.Flip.Avx512
open NeuralReversi.Flip (Square lrmask)
open NeuralReversi.Simd.X86

abbrev Index64 := BitVec 64

def loadLeft (pos : Index64) (valid : pos.toNat < 66) : Vec256 := fun lane =>
  (lrmask.get ⟨pos.toNat,valid⟩).get ⟨lane.val,by omega⟩

def loadRight (pos : Index64) (valid : pos.toNat < 66) : Vec256 := fun lane =>
  (lrmask.get ⟨pos.toNat,valid⟩).get ⟨4+lane.val,by omega⟩

def flipPrepared (pos : Index64) (valid : pos.toNat < 66)
    (pp no zero msb : Vec256) : Word :=
  let rightMask := loadRight pos valid
  let rightBit := lzcntEpi64 (and256 no rightMask)
  let leftMask := loadLeft pos valid
  let leftBit := and256 no leftMask
  let leftBit := ternarylogicEpi64 leftBit (subEpi64 zero leftBit) pp 0x80
  let leftFlank := subEpi64 (cmpeqEpi64 leftBit zero) leftBit
  let rightBit := and256 (rawSrlvEpi64 msb rightBit) pp
  let rightFlips := ternarylogicEpi64 (subEpi64 zero rightBit) rightBit rightMask 0x28
  let flips := ternarylogicEpi64 rightFlips leftFlank leftMask 0xf2
  let half := or128 (cast256Si128 flips) (extractI128 flips 1)
  cvtsi128Si64 (or128 half (shuffleEpi32 half 0x4e))

def flipIndex (pos : Index64) (valid : pos.toNat < 66) (player opponent : Word) : Word :=
  let pp := set1Epi64x player
  let no := set1Epi64x (~~~opponent)
  let zero := set1Epi64x 0
  let msb := set1Epi64x 0x8000000000000000#64
  flipPrepared pos valid pp no zero msb

def squareIndex (sq : Square) : Index64 := BitVec.ofNat 64 sq.val

theorem squareIndex_value (sq : Square) : (squareIndex sq).toNat = sq.val := by
  rw [squareIndex,BitVec.toNat_ofNat]
  exact Nat.mod_eq_of_lt (by have := sq.isLt; omega)

theorem squareIndex_valid (sq : Square) : (squareIndex sq).toNat < 66 := by
  rw [squareIndex_value]
  have := sq.isLt
  omega

def flip (sq : Square) (player opponent : Word) : Word :=
  flipIndex (squareIndex sq) (squareIndex_valid sq) player opponent

end NeuralReversi.Flip.Avx512

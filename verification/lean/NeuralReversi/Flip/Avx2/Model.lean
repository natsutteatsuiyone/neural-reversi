module
public import NeuralReversi.Simd.FlipModel
public import NeuralReversi.Flip.LrmaskModel

@[expose] public section

namespace NeuralReversi.Flip.Avx2
open NeuralReversi.Simd.X86

abbrev Index64 := BitVec 64

def loadLeft (pos : Index64) (valid : pos.toNat < 66) : Vec256 := fun lane =>
  (lrmask.get ⟨pos.toNat,valid⟩).get ⟨lane.val,by omega⟩

def loadRight (pos : Index64) (valid : pos.toNat < 66) : Vec256 := fun lane =>
  (lrmask.get ⟨pos.toNat,valid⟩).get ⟨4+lane.val,by omega⟩

def mmFlipPrepared (pp oo : Vec256) (pos : Index64) (valid : pos.toNat < 66) : Vec128 :=
  let rightMask := loadRight pos valid
  let leftMask := loadLeft pos valid
  let lo := andnot256 oo leftMask
  let lo := and256 (xor256 (addEpi64 lo (set1Epi64x (-1#64))) lo) leftMask
  let rp := and256 pp rightMask
  let rs := or256 rp (srlvEpi64 rp (setEpi64x 7 9 8 1))
  let lf := andnot256 pp lo
  let leftFlip := andnot256 (cmpeqEpi64 lf lo) lf
  let rs := or256 rs (srlvEpi64 rs (setEpi64x 14 18 16 2))
  let rs := or256 rs (srlvEpi64 rs (setEpi64x 28 36 32 4))
  let re := xor256 (andnot256 oo rightMask) rp
  let rightFlip := andnot256 rs (and256 rightMask (cmpgtEpi64 rp re))
  let flips := or256 rightFlip leftFlip
  or128 (cast256Si128 flips) (extractI128 flips 1)

def flipPrepared (pp oo : Vec256) (pos : Index64) (valid : pos.toNat < 66) : Word :=
  let half := mmFlipPrepared pp oo pos valid
  cvtsi128Si64 (or128 half (shuffleEpi32 half 0x4e))

structure BoardCtx where
  pp : Vec256
  oo : Vec256

def newContext (player opponent : Word) : BoardCtx :=
  ⟨set1Epi64x player,set1Epi64x opponent⟩

def flip1 (context : BoardCtx) (pos : Index64) (valid : pos.toNat < 66) : Word :=
  flipPrepared context.pp context.oo pos valid

def squareIndex (sq : Square) : Index64 := BitVec.ofNat 64 sq.val

theorem squareIndex_value (sq : Square) : (squareIndex sq).toNat = sq.val := by
  rw [squareIndex,BitVec.toNat_ofNat]
  exact Nat.mod_eq_of_lt (by have := sq.isLt; omega)

theorem squareIndex_valid (sq : Square) : (squareIndex sq).toNat < 66 := by
  rw [squareIndex_value]
  have := sq.isLt
  omega

def flip (sq : Square) (player opponent : Word) : Word :=
  flip1 (newContext player opponent) (squareIndex sq) (squareIndex_valid sq)

end NeuralReversi.Flip.Avx2

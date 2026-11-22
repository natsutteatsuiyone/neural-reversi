import NeuralReversi.Flip.LrmaskModel
import NeuralReversi.Flip.Types

namespace NeuralReversi.Flip.WasmSimd
open NeuralReversi.Flip (Word Square lrmask)

abbrev Vec128 := Word × Word

def u64x2Splat (x : Word) : Vec128 := (x, x)
def v128And (a b : Vec128) : Vec128 := (a.1 &&& b.1, a.2 &&& b.2)
def v128Or (a b : Vec128) : Vec128 := (a.1 ||| b.1, a.2 ||| b.2)
def v128Not (a : Vec128) : Vec128 := (~~~a.1, ~~~a.2)
def u64x2Sub (a b : Vec128) : Vec128 := (a.1 - b.1, a.2 - b.2)

def u64EqMask (a b : Word) : Word :=
  if a == b then 0xffffffffffffffff#64 else 0#64

def u64x2Eq (a b : Vec128) : Vec128 := (u64EqMask a.1 b.1, u64EqMask a.2 b.2)

def lutHi (idx : BitVec 8) : BitVec 8 :=
  match idx.toNat with
  | 0 => 0x00#8 | 1 => 0x08#8 | 2 => 0x04#8 | 3 => 0x0c#8
  | 4 => 0x02#8 | 5 => 0x0a#8 | 6 => 0x06#8 | 7 => 0x0e#8
  | 8 => 0x01#8 | 9 => 0x09#8 | 10 => 0x05#8 | 11 => 0x0d#8
  | 12 => 0x03#8 | 13 => 0x0b#8 | 14 => 0x07#8 | 15 => 0x0f#8
  | _ => 0x00#8

def lutLo (idx : BitVec 8) : BitVec 8 :=
  match idx.toNat with
  | 0 => 0x00#8 | 1 => 0x80#8 | 2 => 0x40#8 | 3 => 0xc0#8
  | 4 => 0x20#8 | 5 => 0xa0#8 | 6 => 0x60#8 | 7 => 0xe0#8
  | 8 => 0x10#8 | 9 => 0x90#8 | 10 => 0x50#8 | 11 => 0xd0#8
  | 12 => 0x30#8 | 13 => 0xb0#8 | 14 => 0x70#8 | 15 => 0xf0#8
  | _ => 0x00#8

def bitReverseByte (b : BitVec 8) : BitVec 8 :=
  let lo := b &&& 0x0f#8
  let hi := (b >>> 4) &&& 0x0f#8
  lutLo lo ||| lutHi hi

def getByte64 (x : Word) (k : Nat) : BitVec 8 :=
  (x >>> (k * 8)).setWidth 8

def packBytes64 (b0 b1 b2 b3 b4 b5 b6 b7 : BitVec 8) : Word :=
  (b0.setWidth 64) |||
  ((b1.setWidth 64) <<< 8) |||
  ((b2.setWidth 64) <<< 16) |||
  ((b3.setWidth 64) <<< 24) |||
  ((b4.setWidth 64) <<< 32) |||
  ((b5.setWidth 64) <<< 40) |||
  ((b6.setWidth 64) <<< 48) |||
  ((b7.setWidth 64) <<< 56)

def bitReverseU64 (x : Word) : Word :=
  packBytes64
    (bitReverseByte (getByte64 x 7))
    (bitReverseByte (getByte64 x 6))
    (bitReverseByte (getByte64 x 5))
    (bitReverseByte (getByte64 x 4))
    (bitReverseByte (getByte64 x 3))
    (bitReverseByte (getByte64 x 2))
    (bitReverseByte (getByte64 x 1))
    (bitReverseByte (getByte64 x 0))

def bitReverseU64x2 (x : Vec128) : Vec128 :=
  (bitReverseU64 x.1, bitReverseU64 x.2)

def wasmMask (pos : Fin 66) (d : Fin 8) : Word :=
  if d.val < 4 then (lrmask.get pos).get d
  else ((lrmask.get pos).get d).reverse

def flipLeftPair (mask pp no zero one : Vec128) : Vec128 :=
  let nonOpponent := v128And mask no
  let outflank := v128And (v128And nonOpponent (u64x2Sub zero nonOpponent)) pp
  let nonzero := v128Not (u64x2Eq outflank zero)
  v128And (v128And mask (u64x2Sub outflank one)) nonzero

def foldOrPair (x : Vec128) : Word := x.1 ||| x.2

def flipIndexPrepared (pos : Fin 66) (pp no ppRev noRev zero one : Vec128) : Word :=
  let maskLA : Vec128 := (wasmMask pos 0, wasmMask pos 1)
  let maskLB : Vec128 := (wasmMask pos 2, wasmMask pos 3)
  let flipLA := flipLeftPair maskLA pp no zero one
  let flipLB := flipLeftPair maskLB pp no zero one
  let maskRrA : Vec128 := (wasmMask pos 4, wasmMask pos 5)
  let maskRrB : Vec128 := (wasmMask pos 6, wasmMask pos 7)
  let flipRrA := flipLeftPair maskRrA ppRev noRev zero one
  let flipRrB := flipLeftPair maskRrB ppRev noRev zero one
  let flipL := v128Or flipLA flipLB
  let flipRr := bitReverseU64x2 (v128Or flipRrA flipRrB)
  foldOrPair (v128Or flipL flipRr)

structure BoardCtx where
  pp : Vec128
  no : Vec128
  ppRev : Vec128
  noRev : Vec128
  zero : Vec128
  one : Vec128

def BoardCtx.new (player opponent : Word) : BoardCtx :=
  let notOpponent := ~~~opponent
  let rev := bitReverseU64x2 (player, notOpponent)
  { pp := u64x2Splat player
    no := u64x2Splat notOpponent
    ppRev := (rev.1, rev.1)
    noRev := (rev.2, rev.2)
    zero := u64x2Splat 0#64
    one := u64x2Splat 1#64 }

def BoardCtx.flip1 (ctx : BoardCtx) (pos : Fin 66) : Word :=
  flipIndexPrepared pos ctx.pp ctx.no ctx.ppRev ctx.noRev ctx.zero ctx.one

def BoardCtx.flip2 (ctx : BoardCtx) (x0 x1 : Fin 66) : Word × Word :=
  (ctx.flip1 x0, ctx.flip1 x1)

def BoardCtx.flip3 (ctx : BoardCtx) (x0 x1 x2 : Fin 66) : Word × Word × Word :=
  (ctx.flip1 x0, ctx.flip1 x1, ctx.flip1 x2)

def BoardCtx.flip4 (ctx : BoardCtx) (x0 x1 x2 x3 : Fin 66) : Word × Word × Word × Word :=
  (ctx.flip1 x0, ctx.flip1 x1, ctx.flip1 x2, ctx.flip1 x3)

def flipIndex (pos : Fin 66) (player opponent : Word) : Word :=
  (BoardCtx.new player opponent).flip1 pos

def flip (sq : Square) (player opponent : Word) : Word :=
  flipIndex ⟨sq.val, by have := sq.isLt; omega⟩ player opponent

end NeuralReversi.Flip.WasmSimd

module
public import NeuralReversi.Flip.LrmaskModel
public import NeuralReversi.Flip.Types

@[expose] public section

namespace NeuralReversi.Flip.Neon
open NeuralReversi.Flip (Word Square lrmask)

abbrev Vec128 := Word × Word

def vdupqNU64 (x : Word) : Vec128 := (x, x)
def vandqU64 (a b : Vec128) : Vec128 := (a.1 &&& b.1, a.2 &&& b.2)
def vorrqU64 (a b : Vec128) : Vec128 := (a.1 ||| b.1, a.2 ||| b.2)
def vbicqU64 (a b : Vec128) : Vec128 := (a.1 &&& ~~~b.1, a.2 &&& ~~~b.2)
def vaddqU64 (a b : Vec128) : Vec128 := (a.1 + b.1, a.2 + b.2)
def negU64 (x : Vec128) : Vec128 := (-x.1, -x.2)

def qsubU64 (a b : Word) : Word :=
  if a.toNat < b.toNat then 0#64 else a - b

def vqsubqU64 (a b : Vec128) : Vec128 := (qsubU64 a.1 b.1, qsubU64 a.2 b.2)

def vbcaxqU64 (a b c : Vec128) : Vec128 :=
  (a.1 ^^^ (b.1 &&& ~~~c.1), a.2 ^^^ (b.2 &&& ~~~c.2))

def vbslqU64 (sel a b : Vec128) : Vec128 :=
  ((sel.1 &&& a.1) ||| (~~~sel.1 &&& b.1), (sel.2 &&& a.2) ||| (~~~sel.2 &&& b.2))

def vpaddqU64 (a b : Vec128) : Vec128 := (a.1 + a.2, b.1 + b.2)

def neonMask (pos : Fin 66) (d : Fin 8) : Word :=
  match d.val with
  | 0 => (lrmask.get pos).get ⟨0, by decide⟩
  | 1 => (lrmask.get pos).get ⟨1, by decide⟩
  | 2 => ~~~((lrmask.get pos).get ⟨2, by decide⟩)
  | 3 => ~~~((lrmask.get pos).get ⟨3, by decide⟩)
  | 4 => ((lrmask.get pos).get ⟨4, by decide⟩).reverse
  | 5 => ((lrmask.get pos).get ⟨5, by decide⟩).reverse
  | 6 => ~~~(((lrmask.get pos).get ⟨6, by decide⟩).reverse)
  | _ => ~~~(((lrmask.get pos).get ⟨7, by decide⟩).reverse)

def flipSpanPair (mask pp oo one : Vec128) : Vec128 :=
  let nonOpponent := vbicqU64 mask oo
  let playerOnRay := vandqU64 mask pp
  let outflank := vandqU64 (negU64 nonOpponent) playerOnRay
  vqsubqU64 outflank one

def flipSpanPairInv (cmask pp oo one : Vec128) : Vec128 :=
  let carry := vaddqU64 (vorrqU64 oo cmask) one
  let playerOnRay := vbicqU64 pp cmask
  let outflank := vandqU64 carry playerOnRay
  vqsubqU64 outflank one

def mergeSpans (sha3 : Bool) (maskA wA cmaskB wB : Vec128) : Vec128 :=
  if sha3 then vbcaxqU64 (vandqU64 maskA wA) wB cmaskB
  else vbslqU64 cmaskB (vandqU64 maskA wA) wB

def foldAddp (flipL flipRr : Vec128) : Word :=
  let folded := vpaddqU64 flipL flipRr
  folded.1 ||| folded.2.reverse

structure BoardCtx where
  pp : Vec128
  oo : Vec128
  ppRev : Vec128
  ooRev : Vec128
  one : Vec128

def BoardCtx.new (player opponent : Word) : BoardCtx :=
  { pp := vdupqNU64 player
    oo := vdupqNU64 opponent
    ppRev := vdupqNU64 player.reverse
    ooRev := vdupqNU64 opponent.reverse
    one := vdupqNU64 1#64 }

def BoardCtx.flipPairs (sha3 : Bool) (ctx : BoardCtx) (pos : Fin 66) : Vec128 × Vec128 :=
  let maskLA : Vec128 := (neonMask pos 0, neonMask pos 1)
  let cmaskLB : Vec128 := (neonMask pos 2, neonMask pos 3)
  let maskRrA : Vec128 := (neonMask pos 4, neonMask pos 5)
  let cmaskRrB : Vec128 := (neonMask pos 6, neonMask pos 7)
  let wLA := flipSpanPair maskLA ctx.pp ctx.oo ctx.one
  let wRrA := flipSpanPair maskRrA ctx.ppRev ctx.ooRev ctx.one
  let wLB := flipSpanPairInv cmaskLB ctx.pp ctx.oo ctx.one
  let wRrB := flipSpanPairInv cmaskRrB ctx.ppRev ctx.ooRev ctx.one
  let flipL := mergeSpans sha3 maskLA wLA cmaskLB wLB
  let flipRr := mergeSpans sha3 maskRrA wRrA cmaskRrB wRrB
  (flipL, flipRr)

def BoardCtx.flip1 (sha3 : Bool) (ctx : BoardCtx) (pos : Fin 66) : Word :=
  let (flipL, flipRr) := ctx.flipPairs sha3 pos
  foldAddp flipL flipRr

def BoardCtx.flip2 (sha3 : Bool) (ctx : BoardCtx) (x0 x1 : Fin 66) : Word × Word :=
  (ctx.flip1 sha3 x0, ctx.flip1 sha3 x1)

def BoardCtx.flip3 (sha3 : Bool) (ctx : BoardCtx) (x0 x1 x2 : Fin 66) : Word × Word × Word :=
  (ctx.flip1 sha3 x0, ctx.flip1 sha3 x1, ctx.flip1 sha3 x2)

def BoardCtx.flip4 (sha3 : Bool) (ctx : BoardCtx) (x0 x1 x2 x3 : Fin 66) :
    Word × Word × Word × Word :=
  (ctx.flip1 sha3 x0, ctx.flip1 sha3 x1, ctx.flip1 sha3 x2, ctx.flip1 sha3 x3)

def flipIndex (sha3 : Bool) (pos : Fin 66) (player opponent : Word) : Word :=
  (BoardCtx.new player opponent).flip1 sha3 pos

def flip (sha3 : Bool) (sq : Square) (player opponent : Word) : Word :=
  flipIndex sha3 ⟨sq.val, by have := sq.isLt; omega⟩ player opponent

end NeuralReversi.Flip.Neon

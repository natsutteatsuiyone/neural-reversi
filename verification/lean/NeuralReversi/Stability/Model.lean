import Std
namespace NeuralReversi.Stability
abbrev Word := BitVec 64

def center : Word := 0x007e7e7e7e7e7e00

structure FullLines where
  h : Word
  v : Word
  d7 : Word
  d9 : Word
  deriving Repr, BEq, Inhabited

def FullLines.intersection (f : FullLines) : Word := f.h &&& f.v &&& f.d7 &&& f.d9

def rotateRight (x : Word) (n : Nat) : Word := (x >>> n) ||| (x <<< (64 - n))
def rotateLeft (x : Word) (n : Nat) : Word := (x <<< n) ||| (x >>> (64 - n))

def getFullLines (disc : Word) : FullLines := Id.run do
  let mut h := disc
  let mut v := disc
  let mut l7 := disc
  let mut l9 := disc
  let mut r7 := disc
  let mut r9 := disc
  h := h &&& (h >>> 1)
  h := h &&& (h >>> 2)
  h := h &&& (h >>> 4)
  let horizontal := (h &&& 0x0101010101010101) * 0xff
  v := v &&& rotateRight v 8
  v := v &&& rotateRight v 16
  v := v &&& rotateLeft v 32
  l7 := l7 &&& ((0xff01010101010101 : Word) ||| (l7 >>> 7))
  r7 := r7 &&& ((0x80808080808080ff : Word) ||| (r7 <<< 7))
  l7 := l7 &&& ((0xffff030303030303 : Word) ||| (l7 >>> 14))
  r7 := r7 &&& ((0xc0c0c0c0c0c0ffff : Word) ||| (r7 <<< 14))
  l7 := l7 &&& ((0xffffffff0f0f0f0f : Word) ||| (l7 >>> 28))
  r7 := r7 &&& ((0xf0f0f0f0ffffffff : Word) ||| (r7 <<< 28))
  l9 := l9 &&& ((0xff80808080808080 : Word) ||| (l9 >>> 9))
  r9 := r9 &&& ((0x01010101010101ff : Word) ||| (r9 <<< 9))
  l9 := l9 &&& ((0xffffc0c0c0c0c0c0 : Word) ||| (l9 >>> 18))
  r9 := r9 &&& ((0x030303030303ffff : Word) ||| (r9 <<< 18))
  return ⟨horizontal, v, l7 &&& r7,
    l9 &&& r9 &&& ((0x0f0f0f0ff0f0f0f0 : Word) ||| (l9 >>> 36) ||| (r9 <<< 36))⟩

def contactStep (central stable : Word) (full : FullLines) : Word :=
  stable |||
    (((stable >>> 1) ||| (stable <<< 1) ||| full.h) &&&
     ((stable >>> 8) ||| (stable <<< 8) ||| full.v) &&&
     ((stable >>> 7) ||| (stable <<< 7) ||| full.d7) &&&
     ((stable >>> 9) ||| (stable <<< 9) ||| full.d9) &&& central)


end NeuralReversi.Stability

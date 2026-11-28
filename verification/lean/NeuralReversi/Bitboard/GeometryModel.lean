module
public import NeuralReversi.Bitboard.Model

@[expose] public section

namespace NeuralReversi.Bitboard

def byteAt (bits : Bits) (k : Nat) : BitVec 8 := bits.extractLsb' (8 * k) 8

def flipVertical (bits : Bits) : Bits :=
  byteAt bits 0 ++ byteAt bits 1 ++ byteAt bits 2 ++ byteAt bits 3 ++
    byteAt bits 4 ++ byteAt bits 5 ++ byteAt bits 6 ++ byteAt bits 7

def flipHorizontal (bits : Bits) : Bits :=
  let bits := ((bits >>> 1) &&& 0x5555555555555555) ||| ((bits &&& 0x5555555555555555) <<< 1)
  let bits := ((bits >>> 2) &&& 0x3333333333333333) ||| ((bits &&& 0x3333333333333333) <<< 2)
  ((bits >>> 4) &&& 0x0f0f0f0f0f0f0f0f) ||| ((bits &&& 0x0f0f0f0f0f0f0f0f) <<< 4)

def deltaSwapRaw (bits mask : Bits) (delta : Nat) : Bits :=
  let tmp := mask &&& (bits ^^^ (bits <<< delta))
  bits ^^^ tmp ^^^ (tmp >>> delta)

def flipDiagA1H8 (bits : Bits) : Bits :=
  let bits := deltaSwapRaw bits 0x0f0f0f0f00000000 28
  let bits := deltaSwapRaw bits 0x3333000033330000 14
  deltaSwapRaw bits 0x5500550055005500 7

def flipDiagA8H1 (bits : Bits) : Bits :=
  let bits := deltaSwapRaw bits 0xf0f0f0f000000000 36
  let bits := deltaSwapRaw bits 0xcccc0000cccc0000 18
  deltaSwapRaw bits 0xaa00aa00aa00aa00 9

def rotate90 (bits : Bits) : Bits := flipVertical (flipDiagA8H1 bits)

def rotate180 (bits : Bits) : Bits := bits.reverse

def rotate270 (bits : Bits) : Bits := flipVertical (flipDiagA1H8 bits)

end NeuralReversi.Bitboard

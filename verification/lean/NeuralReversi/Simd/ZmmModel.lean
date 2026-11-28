module
public import NeuralReversi.Simd.TernaryFast

@[expose] public section

namespace NeuralReversi.Simd.X86
abbrev Vec512 := Fin 8 → Word

def setEpi64 (a7 a6 a5 a4 a3 a2 a1 a0 : Word) : Vec512 := fun i =>
  match i.val with
  | 0 => a0 | 1 => a1 | 2 => a2 | 3 => a3
  | 4 => a4 | 5 => a5 | 6 => a6 | _ => a7
def set1Epi64 (a : Word) : Vec512 := fun _ => a
def and512 (a b : Vec512) : Vec512 := fun i => a i &&& b i
def or512 (a b : Vec512) : Vec512 := fun i => a i ||| b i
def add512 (a b : Vec512) : Vec512 := fun i => a i + b i
def rolv512 (a counts : Vec512) : Vec512 := fun i =>
  (a i).rotateLeft ((counts i &&& 63).toNat)
def ternary512 (a b c : Vec512) (imm : BitVec 8) : Vec512 := fun i =>
  ternaryWordFast (a i) (b i) (c i) imm
def unpacklo512 (a b : Vec512) : Vec512 := fun i =>
  match i.val with
  | 0 => a 0 | 1 => b 0 | 2 => a 2 | 3 => b 2
  | 4 => a 4 | 5 => b 4 | 6 => a 6 | _ => b 6
def unpackhi512 (a b : Vec512) : Vec512 := fun i =>
  match i.val with
  | 0 => a 1 | 1 => b 1 | 2 => a 3 | 3 => b 3
  | 4 => a 5 | 5 => b 5 | 6 => a 7 | _ => b 7
def cast512Si256 (a : Vec512) : Vec256 := fun i => a ⟨i.val,by omega⟩
def extractI64x4 (a : Vec512) (imm : Fin 2) : Vec256 := fun i =>
  a ⟨4*imm.val+i.val,by omega⟩
def horizontalOr2Zmm (a b : Vec512) : Word × Word :=
  let lo := unpacklo512 a b
  let hi := unpackhi512 a b
  let pairs := or512 lo hi
  let m256 := or256 (cast512Si256 pairs) (extractI64x4 pairs 1)
  let m128 := or128 (cast256Si128 m256) (extractI128 m256 1)
  (cvtsi128Si64 m128,cvtsi128Si64 (shuffleEpi32 m128 0x4e))


def pack512 (a : Vec512) : BitVec 512 :=
  pack256 (extractI64x4 a 1) ++ pack256 (cast512Si256 a)

def unpack512 (a : BitVec 512) : Vec512 := fun i =>
  if h : i.val < 4 then unpack256 (a.extractLsb' 0 256) ⟨i.val,h⟩
  else unpack256 (a.extractLsb' 256 256) ⟨i.val-4,by omega⟩


end NeuralReversi.Simd.X86

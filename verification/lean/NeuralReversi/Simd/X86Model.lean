import NeuralReversi.Bitboard.Model

namespace NeuralReversi.Simd.X86
abbrev Word := BitVec 64
abbrev Vec256 := Fin 4 → Word
abbrev Vec128 := Fin 2 → Word

def setEpi64x (e3 e2 e1 e0 : Word) : Vec256 := fun i =>
  match i.val with
  | 0 => e0 | 1 => e1 | 2 => e2 | _ => e3

def set1Epi64x (word : Word) : Vec256 := fun _ => word
def and256 (a b : Vec256) : Vec256 := fun i => a i &&& b i
def or256 (a b : Vec256) : Vec256 := fun i => a i ||| b i
def addEpi64 (a b : Vec256) : Vec256 := fun i => a i + b i

/-- Packed variable shifts zero a lane on counts >=64; they do not mask counts
as Rust scalar runtime shifts do. Explicit guards also avoid huge Nat shifts. -/
def sllvEpi64 (a counts : Vec256) : Vec256 := fun i =>
  if (counts i).toNat < 64 then a i <<< (counts i).toNat else 0

def srlvEpi64 (a counts : Vec256) : Vec256 := fun i =>
  if (counts i).toNat < 64 then a i >>> (counts i).toNat else 0

def cast256Si128 (a : Vec256) : Vec128 := fun i => a ⟨i.val,by omega⟩

def extractI128 (a : Vec256) (immediate : Fin 2) : Vec128 := fun i =>
  a ⟨2*immediate.val+i.val,by omega⟩

def or128 (a b : Vec128) : Vec128 := fun i => a i ||| b i
def cvtsi128Si64 (a : Vec128) : Word := a 0

def dword (a : Vec128) (i : Fin 4) : BitVec 32 :=
  match i.val with
  | 0 => (a 0).extractLsb' 0 32
  | 1 => (a 0).extractLsb' 32 32
  | 2 => (a 1).extractLsb' 0 32
  | _ => (a 1).extractLsb' 32 32

def fromDwords (a : Fin 4 → BitVec 32) : Vec128 := fun i =>
  if i.val = 0 then a 1 ++ a 0 else a 3 ++ a 2

def shuffleEpi32 (a : Vec128) (immediate : BitVec 8) : Vec128 :=
  fromDwords fun i => dword a
    ⟨immediate.toNat / 2^(2*i.val) % 4,Nat.mod_lt _ (by decide)⟩

def pack256 (a : Vec256) : BitVec 256 := a 3 ++ a 2 ++ a 1 ++ a 0
def unpack256 (a : BitVec 256) : Vec256 := fun i => a.extractLsb' (64*i.val) 64

def horizontalOrU64 (a : Vec256) : Word :=
  let m128 := or128 (cast256Si128 a) (extractI128 a 1)
  cvtsi128Si64 (or128 m128 (shuffleEpi32 m128 0x4e))

end NeuralReversi.Simd.X86

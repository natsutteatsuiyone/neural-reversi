module

@[expose] public section

namespace NeuralReversi.Square

abbrev Square := Fin 64
abbrev Byte := BitVec 8

inductive Panic where
  | invalidFile (value : Nat)
  | invalidRank (value : Nat)
  deriving DecidableEq, Repr

def Panic.message : Panic → String
  | .invalidFile n => s!"Invalid file: {n}"
  | .invalidRank n => s!"Invalid rank: {n}"

def index (sq : Square) : Nat := sq.val

def fromU8 (value : Byte) : Option Square :=
  if h : value.toNat < 64 then some ⟨value.toNat, h⟩ else none

def file (sq : Square) : Nat := index sq % 8

def rank (sq : Square) : Nat := index sq / 8

def fromFileRank (file rank : Byte) : Except Panic Square :=
  if hf : file.toNat < 8 then
    if hr : rank.toNat < 8 then
      .ok ⟨rank.toNat * 8 + file.toNat, by omega⟩
    else .error (.invalidRank rank.toNat)
  else .error (.invalidFile file.toNat)

def iter : List Square := List.finRange 64

end NeuralReversi.Square

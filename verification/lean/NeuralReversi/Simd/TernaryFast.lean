module
public import NeuralReversi.Simd.TernaryProperties

@[expose] public section

namespace NeuralReversi.Simd.X86

def ternaryTerm (a b c : Word) (imm : BitVec 8) (row : Nat) : Word :=
  if imm.getLsbD row then
    (if row / 4 % 2 == 1 then a else ~~~a) &&&
    (if row / 2 % 2 == 1 then b else ~~~b) &&&
    (if row % 2 == 1 then c else ~~~c)
  else 0

def ternaryWordMinterms (a b c : Word) (imm : BitVec 8) : Word :=
  ternaryTerm a b c imm 0 ||| ternaryTerm a b c imm 1 |||
  ternaryTerm a b c imm 2 ||| ternaryTerm a b c imm 3 |||
  ternaryTerm a b c imm 4 ||| ternaryTerm a b c imm 5 |||
  ternaryTerm a b c imm 6 ||| ternaryTerm a b c imm 7

theorem wordGate_bit (flag : Bool) (x : Word) (i : Nat) :
    (if flag then x else 0).getLsbD i = (flag && x.getLsbD i) := by
  cases flag <;> simp

theorem ternaryWordMinterms_exact (a b c : Word) (imm : BitVec 8) :
    ternaryWordMinterms a b c imm = ternaryWord a b c imm := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  rw [ternaryWord,wordFromBits_bit _ i hi]
  simp only [ternaryWordMinterms,BitVec.getLsbD_or,ternaryTerm,wordGate_bit]
  cases ha : a.getLsbD i <;> cases hb : b.getLsbD i <;> cases hc : c.getLsbD i
  all_goals simp_all [ternaryBit,truthIndex]

def ternaryWordFast (a b c : Word) (imm : BitVec 8) : Word :=
  if imm = 0xf8 then a ||| (b &&& c) else ternaryWordMinterms a b c imm

theorem ternaryWordFast_exact (a b c : Word) (imm : BitVec 8) :
    ternaryWordFast a b c imm = ternaryWord a b c imm := by
  unfold ternaryWordFast
  split
  · next same => rw [same,ternaryWord_f8]
  · exact ternaryWordMinterms_exact a b c imm

def ternarylogicEpi64Fast (a b c : Vec256) (imm : BitVec 8) : Vec256 := fun i =>
  ternaryWordFast (a i) (b i) (c i) imm

theorem ternarylogicEpi64Fast_f8 (a b c : Vec256) :
    ternarylogicEpi64Fast a b c 0xf8 = or256 a (and256 b c) := by
  funext lane
  rw [ternarylogicEpi64Fast, ternaryWordFast_exact, ternaryWord_f8]
  rfl

end NeuralReversi.Simd.X86

import NeuralReversi.Simd.TernaryModel

namespace NeuralReversi.Simd.X86

theorem ternaryBit_f8 : ∀ a b c, ternaryBit 0xf8 a b c = (a || (b && c)) := by decide

theorem wordFromBits_bit (bits : Fin 64 → Bool) (i : Nat) (hi : i < 64) :
    (wordFromBits bits).getLsbD i = bits ⟨i,hi⟩ := by
  simp only [wordFromBits,BitVec.getLsbD_cast,BitVec.getLsbD_ofBoolListLE,List.getD_eq_getElem?_getD]
  rw [List.getElem?_eq_getElem (by simpa only [List.length_ofFn] using hi),List.getElem_ofFn]
  rfl

theorem ternaryWord_f8 (a b c : Word) : ternaryWord a b c 0xf8 = a ||| (b &&& c) := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  rw [ternaryWord,wordFromBits_bit _ i hi,ternaryBit_f8]
  simp only [BitVec.getLsbD_or,BitVec.getLsbD_and]

theorem ternarylogicEpi64_f8 (a b c : Vec256) :
    ternarylogicEpi64 a b c 0xf8 = or256 a (and256 b c) := by
  funext lane
  exact ternaryWord_f8 (a lane) (b lane) (c lane)

end NeuralReversi.Simd.X86

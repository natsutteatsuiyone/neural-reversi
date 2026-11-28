module
public import NeuralReversi.Board.ParseModel

@[expose] public section

namespace NeuralReversi.Board
open Bitboard

theorem scalarCount_exact (text : String) (bound : RepresentableText text) :
    (BitVec.ofNat 64 text.toList.length).toNat = text.toList.length := by
  exact Nat.mod_eq_of_lt bound

theorem fromString_short (checked : Bool) (text : String) (side : Disc.PlayerColor)
    (bound : RepresentableText text) (short : text.toList.length < 64) :
    fromString checked text side =
      .error (.error (.tooShort 64 (BitVec.ofNat 64 text.toList.length))) := by
  simp only [fromString, scalarCount_exact text bound, short, ↓reduceIte]

theorem fromString_long (checked : Bool) (text : String) (side : Disc.PlayerColor)
    (bound : RepresentableText text) (long : 64 < text.toList.length) :
    fromString checked text side =
      .error (.error (.tooLong 64 (BitVec.ofNat 64 text.toList.length))) := by
  have notShort : ¬text.toList.length < 64 := by omega
  simp only [fromString, scalarCount_exact text bound, notShort, long, ↓reduceIte]

theorem fromString_sized (checked : Bool) (text : String) (side : Disc.PlayerColor)
    (size : text.toList.length = 64) :
    fromString checked text side =
      parseTokens checked side ⟨0#64, 0#64⟩ (Square.iter.zip text.toList) := by
  simp only [fromString, size]
  rfl

def ParseGuardContract : Prop :=
  ∀ (checked : Bool) (text : String) (side : Disc.PlayerColor), RepresentableText text →
    (BitVec.ofNat 64 text.toList.length).toNat = text.toList.length ∧
    (text.toList.length < 64 →
      fromString checked text side = .error (.error (.tooShort 64 (BitVec.ofNat 64 text.toList.length)))) ∧
    (64 < text.toList.length →
      fromString checked text side = .error (.error (.tooLong 64 (BitVec.ofNat 64 text.toList.length)))) ∧
    (text.toList.length = 64 →
      fromString checked text side = parseTokens checked side ⟨0#64, 0#64⟩ (Square.iter.zip text.toList))

theorem parseGuardContract_proven : ParseGuardContract := by
  intro checked text side bound
  exact ⟨scalarCount_exact text bound,
    fun short => fromString_short checked text side bound short,
    fun long => fromString_long checked text side bound long,
    fromString_sized checked text side⟩

end NeuralReversi.Board

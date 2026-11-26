import NeuralReversi.Disc.Properties
import NeuralReversi.Square.TextModel

namespace NeuralReversi.Disc.PlayerColor

def asStr : PlayerColor → String
  | .black => "black"
  | .white => "white"

def reprByte : PlayerColor → BitVec 8
  | .black => 1
  | .white => 2

def index (color : PlayerColor) : Nat := color.reprByte.toNat - 1

def ofDisc : Disc → Option PlayerColor
  | .empty => none
  | .black => some .black
  | .white => some .white

def asciiLower (c : Char) : Char :=
  if 65 ≤ c.toNat ∧ c.toNat ≤ 90 then Char.ofNat (c.toNat + 32) else c

def normalize (input : String) : String :=
  String.ofList ((Square.trimWhitespace input.toList).map asciiLower)

def fromStr (input : String) : Option PlayerColor :=
  let token := normalize input
  if token = "b" ∨ token = "black" then some .black
  else if token = "w" ∨ token = "white" then some .white
  else none

def SyntaxContract : Prop :=
  (∀ text, let token := normalize text
    (fromStr text = some .black ↔ token = "b" ∨ token = "black") ∧
    (fromStr text = some .white ↔ token = "w" ∨ token = "white") ∧
    (fromStr text = none ↔ token ≠ "b" ∧ token ≠ "black" ∧ token ≠ "w" ∧ token ≠ "white")) ∧
  (∀ color, fromStr color.asStr = some color ∧ fromStr (String.singleton color.toChar) = none)

theorem syntaxContract_proven : SyntaxContract := by
  constructor
  · intro text
    dsimp only
    unfold fromStr
    generalize normalize text = token
    by_cases black : token = "b" ∨ token = "black"
    · rcases black with h | h <;> simp [h]
    · by_cases white : token = "w" ∨ token = "white"
      · rcases white with h | h <;> simp [h]
      · simp only [not_or] at black white
        simp [black, white]
  · intro color
    cases color <;> decide

def RepresentationContract : Prop :=
  (∀ color : PlayerColor,
    color.index < 2 ∧ color.opposite.index = 1 - color.index ∧
    ofDisc color.toDisc = some color ∧ color.toChar = Disc.toChar color.toDisc) ∧
  (∀ disc, ofDisc disc = none ↔ disc = .empty) ∧
  (∀ a b : PlayerColor, a.index = b.index ↔ a = b)

theorem representationContract_proven : RepresentationContract := by
  constructor
  · intro color
    cases color <;> decide
  · constructor
    · intro disc
      cases disc <;> decide
    · intro a b
      cases a <;> cases b <;> decide

end NeuralReversi.Disc.PlayerColor

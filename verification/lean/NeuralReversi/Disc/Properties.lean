module
public import NeuralReversi.Disc.Model

@[expose] public section

namespace NeuralReversi.Disc

def CharacterEncoding : Prop :=
  ∀ disc : Disc,
    (toChar disc = '-' ↔ disc = .empty) ∧
    (toChar disc = 'X' ↔ disc = .black) ∧
    (toChar disc = 'O' ↔ disc = .white)

theorem characterEncoding_proven : CharacterEncoding := by
  intro disc
  cases disc <;> decide

theorem toChar_injective (a b : Disc) (h : toChar a = toChar b) : a = b := by
  cases a <;> cases b <;> simp_all [toChar]

def ColorExchange : Prop :=
  ∀ color : PlayerColor,
    color.opposite.opposite = color ∧
    color.opposite ≠ color ∧
    (color.opposite = .black ↔ color = .white) ∧
    (color.opposite = .white ↔ color = .black)

theorem colorExchange_proven : ColorExchange := by
  intro color
  cases color <;> decide

end NeuralReversi.Disc

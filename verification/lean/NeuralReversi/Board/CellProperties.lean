import NeuralReversi.Board.TextModel
import NeuralReversi.Bitboard.SquareProperties
import NeuralReversi.Disc.Properties

namespace NeuralReversi.Board
open Bitboard

def cellDisc (b : Board) (index : Nat) (side : Disc.PlayerColor) : Disc.Disc :=
  if b.player.getLsbD index then side.toDisc
  else if b.opponent.getLsbD index then side.opposite.toDisc else .empty

theorem discAt_playable (checked : Bool) (b : Board) (sq : Bitboard.Square) (side : Disc.PlayerColor) :
    discAt checked b sq side = .ok (cellDisc b sq.val side) := by
  have player := squareMembershipContract_proven b.player sq checked checked
  have opponent := squareMembershipContract_proven b.opponent sq checked checked
  simp only [discAt, player, opponent]
  cases hp : b.player.getLsbD sq.val <;> cases ho : b.opponent.getLsbD sq.val <;>
    simp only [cellDisc, hp, ho, Bool.false_eq_true, ↓reduceIte]

theorem cellDisc_semantics (b : Board) (i : Nat) (side : Disc.PlayerColor) :
    (cellDisc b i side = side.toDisc ↔ b.player.getLsbD i = true) ∧
    (cellDisc b i side = side.opposite.toDisc ↔
      b.player.getLsbD i = false ∧ b.opponent.getLsbD i = true) ∧
    (cellDisc b i side = .empty ↔
      b.player.getLsbD i = false ∧ b.opponent.getLsbD i = false) := by
  cases side <;> cases hp : b.player.getLsbD i <;> cases ho : b.opponent.getLsbD i <;>
    simp_all [cellDisc, Disc.PlayerColor.opposite, Disc.PlayerColor.toDisc]

def DiscAtContract : Prop :=
  ∀ (b : Board) (sq : Bitboard.Square) (side : Disc.PlayerColor) (checked : Bool),
    discAt checked b sq side = .ok (cellDisc b sq.val side) ∧
    (cellDisc b sq.val side = side.toDisc ↔ b.player.getLsbD sq.val = true) ∧
    (cellDisc b sq.val side = side.opposite.toDisc ↔
      b.player.getLsbD sq.val = false ∧ b.opponent.getLsbD sq.val = true) ∧
    (cellDisc b sq.val side = .empty ↔
      b.player.getLsbD sq.val = false ∧ b.opponent.getLsbD sq.val = false)

theorem discAtContract_proven : DiscAtContract := by
  intro b sq side checked
  exact ⟨discAt_playable checked b sq side, cellDisc_semantics b sq.val side⟩

end NeuralReversi.Board

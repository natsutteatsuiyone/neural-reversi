import NeuralReversi.Board.Model
import NeuralReversi.Disc.Model
import NeuralReversi.Bitboard.SquareModel

namespace NeuralReversi.Game.History

abbrev Square := NeuralReversi.Bitboard.AnySquare
abbrev Board := NeuralReversi.Board.Board
abbrev Disc := NeuralReversi.Disc.PlayerColor

inductive Action where
  | play (square : Square)
  | explicitPass
  | automaticPass
  deriving DecidableEq, Repr

structure Entry where
  action : Action
  board : Board
  side : Disc
  deriving DecidableEq, Repr

structure ValueState where
  board : Board
  side : Disc
  history : List Entry
  deriving DecidableEq, Repr

end NeuralReversi.Game.History

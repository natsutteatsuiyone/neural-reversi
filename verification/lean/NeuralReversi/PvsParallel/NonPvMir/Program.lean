import NeuralReversi.EndgameSearch.Direct.Mir.Program
import NeuralReversi.ScoreRefresh.BoardBridge

namespace NeuralReversi.PvsParallel.NonPvMir

def pvZeroScore (checked : Bool) (board : Board.Board) : Except String ScoreRefresh.Score :=
  ScoreRefresh.BoardBridge.finalScore checked board

inductive Arm where
  | fixed (arm : EndgameSearch.Direct.Mir.Arm)
  | shallow

def dispatch : Nat → Option Arm
  | 0 => some (.fixed .terminal)
  | 1 => some (.fixed .solve1)
  | 2 => some (.fixed .solve2)
  | 3 => some (.fixed .solve3)
  | 4 => some (.fixed .solve4)
  | 5 => some (.shallow)
  | 6 => some (.shallow)
  | _ => none

end NeuralReversi.PvsParallel.NonPvMir

namespace NeuralReversi.EndgameSearch.Direct.Mir

inductive Arm where
  | terminal | solve1 | solve2 | solve3 | solve4

def dispatch : Nat → Option Arm
  | 0 => some .terminal
  | 1 => some .solve1
  | 2 => some .solve2
  | 3 => some .solve3
  | 4 => some .solve4
  | _ => none

end NeuralReversi.EndgameSearch.Direct.Mir

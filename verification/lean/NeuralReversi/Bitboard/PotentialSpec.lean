import NeuralReversi.Bitboard.AdjacencyModel

namespace NeuralReversi.Bitboard.Moves

def BracketableOpponent (opponent : Bits) (target : Nat) : Prop :=
  ∃ cell, cell < 64 ∧ opponent.getLsbD cell = true ∧ bracketableAdjacent target cell = true

def PotentialMove (player opponent : Bits) (target : Nat) : Prop :=
  target < 64 ∧ player.getLsbD target = false ∧ opponent.getLsbD target = false ∧
    BracketableOpponent opponent target

end NeuralReversi.Bitboard.Moves

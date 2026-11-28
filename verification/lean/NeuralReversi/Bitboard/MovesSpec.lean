module
public import NeuralReversi.Bitboard.Model

@[expose] public section

namespace NeuralReversi.Bitboard.Moves

def OnRay (anchor index k : Nat) (dr dc : Int) : Prop :=
  (index / 8 : Int) = (anchor / 8 : Int) + (k : Int) * dr ∧
  (index % 8 : Int) = (anchor % 8 : Int) + (k : Int) * dc

def Ray (player opponent : Bits) (target : Nat) (dr dc : Int) : Prop :=
  ∃ anchor distance, anchor < 64 ∧ 2 ≤ distance ∧ distance ≤ 7 ∧
    player.getLsbD anchor = true ∧ OnRay anchor target distance dr dc ∧
    ∀ k, 0 < k → k < distance →
      ∃ index, index < 64 ∧ OnRay anchor index k dr dc ∧ opponent.getLsbD index = true

def Bracketed (player opponent : Bits) (target : Nat) : Prop :=
  Ray player opponent target 0 1 ∨ Ray player opponent target 0 (-1) ∨
  Ray player opponent target 1 0 ∨ Ray player opponent target (-1) 0 ∨
  Ray player opponent target 1 1 ∨ Ray player opponent target 1 (-1) ∨
  Ray player opponent target (-1) 1 ∨ Ray player opponent target (-1) (-1)

def LegalMove (player opponent : Bits) (target : Nat) : Prop :=
  target < 64 ∧ player.getLsbD target = false ∧ opponent.getLsbD target = false ∧
    Bracketed player opponent target

end NeuralReversi.Bitboard.Moves

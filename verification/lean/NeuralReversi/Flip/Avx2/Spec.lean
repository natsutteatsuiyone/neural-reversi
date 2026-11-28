module
public import NeuralReversi.Flip.Spec

@[expose] public section

namespace NeuralReversi.Flip.Avx2

abbrev Index := Fin 66
abbrev Lanes := Fin 4 → Word
abbrev Pair := Fin 2 → Word

def CoordinatePrefix (sq : Square) (d : Fin 8) (opponent : Word) (i : Nat) : Prop :=
  Cell sq d i ∧ ∀ k, Cell sq d k → k < i → opponent.getLsbD k = true

def RightAnchor (sq : Square) (d : Fin 8) (player opponent : Word) : Prop :=
  ∃ j, Cell sq d j ∧ player.getLsbD j = true ∧ opponent.getLsbD j = false ∧
    ∀ k, Cell sq d k → j < k → opponent.getLsbD k = true

def LeftRawBit (sq : Square) (d : Fin 8) (player opponent : Word) (i : Nat) : Prop :=
  CoordinatePrefix sq d opponent i ∧ player.getLsbD i = false ∧
    ∃ j, CoordinatePrefix sq d opponent j ∧ player.getLsbD j = true

def RightRawBit (sq : Square) (d : Fin 8) (player opponent : Word) (i : Nat) : Prop :=
  RightAnchor sq d player opponent ∧ Cell sq d i ∧
    ¬ ∃ j, Cell sq d j ∧ player.getLsbD j = true ∧ i ≤ j

def LaneSpec (sq : Square) (d : Fin 8) (player opponent : Word) (i : Nat) : Prop :=
  if d.val < 4 then LeftRawBit sq d player opponent i
  else RightRawBit sq d player opponent i

def RawSpec (sq : Square) (player opponent : Word) (i : Nat) : Prop :=
  ∃ d : Fin 8, LaneSpec sq d player opponent i

def LaneSpec66 (pos : Index) (d : Fin 8) (player opponent : Word) (i : Nat) : Prop :=
  if h : pos.val < 64 then LaneSpec ⟨pos.val, h⟩ d player opponent i else False

def inputLane (d : Fin 8) : Fin 4 := ⟨d.val % 4, Nat.mod_lt _ (by decide)⟩

def PairBitSpec (pos : Index) (pp oo : Lanes) (j : Fin 2) (i : Nat) : Prop :=
  ∃ d : Fin 8, d.val % 2 = j.val ∧
    LaneSpec66 pos d (pp (inputLane d)) (oo (inputLane d)) i

end NeuralReversi.Flip.Avx2

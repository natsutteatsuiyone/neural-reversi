module
public import NeuralReversi.Flip.Types
public import NeuralReversi.Bitboard.MovesSpec

@[expose] public section

namespace NeuralReversi.Flip
open NeuralReversi.Bitboard.Moves (OnRay)

def delta (d : Fin 8) : Int × Int :=
  match d.val with
  | 0 => (0, 1)
  | 1 => (1, 0)
  | 2 => (1, 1)
  | 3 => (1, -1)
  | 4 => (0, -1)
  | 5 => (-1, 0)
  | 6 => (-1, -1)
  | _ => (-1, 1)

def Cell (sq : Square) (d : Fin 8) (i : Nat) : Prop :=
  sq.val < 64 ∧ i < 64 ∧ ∃ k : Fin 8,
    0 < k.val ∧ OnRay sq.val i k.val (delta d).1 (delta d).2

def RayMask (sq : Square) (d : Fin 8) (m : Word) : Prop :=
  ∀ i, m.getLsbD i = true ↔ Cell sq d i

def DirectionalBit (sq : Square) (d : Fin 8) (player opponent : Word) (i : Nat) : Prop :=
  sq.val < 64 ∧ i < 64 ∧ ∃ anchor distance,
    anchor < 64 ∧ 2 ≤ distance ∧ distance ≤ 7 ∧
    player.getLsbD anchor = true ∧ opponent.getLsbD anchor = false ∧
    OnRay sq.val anchor distance (delta d).1 (delta d).2 ∧
    (∀ k, 0 < k → k < distance →
      ∃ index, index < 64 ∧ OnRay sq.val index k (delta d).1 (delta d).2 ∧
        opponent.getLsbD index = true) ∧
    ∃ k, 0 < k ∧ k < distance ∧ OnRay sq.val i k (delta d).1 (delta d).2

def CoordinateBit (sq : Square) (player opponent : Word) (i : Nat) : Prop :=
  ∃ d : Fin 8, DirectionalBit sq d player opponent i

instance (sq : Square) (d : Fin 8) (i : Nat) : Decidable (Cell sq d i) := by
  unfold Cell OnRay
  infer_instance

end NeuralReversi.Flip

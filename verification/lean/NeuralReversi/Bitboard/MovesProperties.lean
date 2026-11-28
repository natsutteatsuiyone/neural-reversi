module
public import NeuralReversi.Bitboard.MovesGeometry

@[expose] public section

namespace NeuralReversi.Bitboard.Moves

def PortableMovesContract : Prop :=
  ∀ (player opponent : Bits) (target : Nat),
    (getMovesPortable player opponent).getLsbD target = true ↔
      LegalMove player opponent target

theorem portableMovesContract_proven : PortableMovesContract := getMovesPortable_exact

theorem opponent_excludes_player (p o : Bits) (disjoint : p &&& o = 0) (i : Nat)
    (opponent : o.getLsbD i = true) : p.getLsbD i = false := by
  have bit := congrArg (fun bits : Bits => bits.getLsbD i) disjoint
  simpa [opponent] using bit

theorem portable_bits_iff (p o : Bits) (i : Nat) (ib : i < 64) :
    (getMovesPortable p o).getLsbD i = true ↔
      p.getLsbD i = false ∧ o.getLsbD i = false ∧
        (portableDirections p o).getLsbD i = true := by
  simp only [portable_decomposition, BitVec.getLsbD_and, BitVec.getLsbD_not,
    BitVec.getLsbD_or, ib, decide_true, Bool.true_and]
  cases (portableDirections p o).getLsbD i <;>
    cases p.getLsbD i <;> cases o.getLsbD i <;> decide

theorem portable_empty_target (p o : Bits) (i : Nat) (ib : i < 64)
    (move : (getMovesPortable p o).getLsbD i = true) :
    p.getLsbD i = false ∧ o.getLsbD i = false := by
  have exactBits := (portable_bits_iff p o i ib).mp move
  exact ⟨exactBits.1, exactBits.2.1⟩

end NeuralReversi.Bitboard.Moves

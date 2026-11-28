module
public import NeuralReversi.Bitboard.DispatchModel
public import NeuralReversi.Bitboard.Avx512Properties
public import NeuralReversi.Bitboard.Avx512PotentialProperties

@[expose] public section

namespace NeuralReversi.Bitboard.Moves

theorem getMovesX86_exact (config : X86Config) (player opponent : Bits) (i : Nat) :
    (getMovesX86 config player opponent).getLsbD i = true ↔ LegalMove player opponent i := by
  unfold getMovesX86
  split
  · exact getMovesAvx512_exact player opponent i
  · split
    · exact getMovesAvx2_exact player opponent i
    · exact getMovesPortable_exact player opponent i

def BothCoordinateOutputs (player opponent : Bits) (i : Nat) (result : Bits × Bits) : Prop :=
  (result.1.getLsbD i = true ↔ LegalMove player opponent i) ∧
  (result.2.getLsbD i = true ↔ PotentialMove player opponent i)

theorem getMovesAndPotentialX86_exact (config : X86Config) (player opponent : Bits) (i : Nat) :
    BothCoordinateOutputs player opponent i (getMovesAndPotentialX86 config player opponent) := by
  unfold getMovesAndPotentialX86
  split
  · exact avx512MovesAndPotentialContract_proven player opponent i
  · split
    · exact avx2MovesAndPotentialContract_proven player opponent i
    · exact portableMovesAndPotentialContract_proven player opponent i

end NeuralReversi.Bitboard.Moves

namespace NeuralReversi.Bitboard

theorem getMoves_exact (config : Moves.X86Config) (player opponent : Bits) (i : Nat) :
    (getMoves config player opponent).getLsbD i = true ↔ Moves.LegalMove player opponent i :=
  Moves.getMovesX86_exact config player opponent i

theorem getMovesAndPotential_exact (config : Moves.X86Config) (player opponent : Bits) (i : Nat) :
    Moves.BothCoordinateOutputs player opponent i (getMovesAndPotential config player opponent) :=
  Moves.getMovesAndPotentialX86_exact config player opponent i

theorem legalMoves_exact (config : Moves.X86Config) (player opponent : Bits) (i : Nat) :
    (legalMoves config player opponent).getLsbD i = true ↔ Moves.LegalMove player opponent i :=
  getMoves_exact config player opponent i

theorem legalMovesAndPotential_exact (config : Moves.X86Config) (player opponent : Bits) (i : Nat) :
    Moves.BothCoordinateOutputs player opponent i (legalMovesAndPotential config player opponent) :=
  getMovesAndPotential_exact config player opponent i

def X86CombinedDispatchContract : Prop := ∀ config player opponent i,
  Moves.BothCoordinateOutputs player opponent i (Moves.getMovesAndPotentialX86 config player opponent) ∧
  Moves.BothCoordinateOutputs player opponent i (getMovesAndPotential config player opponent)

theorem x86CombinedDispatchContract_proven : X86CombinedDispatchContract := by
  intro config player opponent i
  exact ⟨Moves.getMovesAndPotentialX86_exact config player opponent i,
    getMovesAndPotential_exact config player opponent i⟩

def X86DispatchContract : Prop := ∀ config player opponent i,
  ((Moves.getMovesX86 config player opponent).getLsbD i = true ↔
    Moves.LegalMove player opponent i) ∧
  ((getMoves config player opponent).getLsbD i = true ↔ Moves.LegalMove player opponent i)

theorem x86DispatchContract_proven : X86DispatchContract := by
  intro config player opponent i
  exact ⟨Moves.getMovesX86_exact config player opponent i, getMoves_exact config player opponent i⟩

theorem word_zero_iff_no_bits (bits : Bits) : bits = 0 ↔ ∀ i, bits.getLsbD i = false := by
  constructor
  · intro h i
    simp [h]
  · intro h
    apply BitVec.eq_of_getLsbD_eq
    intro i _
    simp [h]

end NeuralReversi.Bitboard

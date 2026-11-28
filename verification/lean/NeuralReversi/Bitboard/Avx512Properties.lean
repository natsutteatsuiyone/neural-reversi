module
public import NeuralReversi.Simd.TernaryProperties
public import NeuralReversi.Bitboard.Avx512Model
public import NeuralReversi.Bitboard.Avx2Properties

@[expose] public section

namespace NeuralReversi.Bitboard.Moves
open NeuralReversi.Simd.X86

theorem getMovesAvx512_eq_avx2 (p o : Bits) : getMovesAvx512 p o = getMovesAvx2 p o := by
  simp only [getMovesAvx512,getMovesAvx2,ternarylogicEpi64Fast_f8]

theorem getMovesAvx512_exact (p o : Bits) (i : Nat) :
    (getMovesAvx512 p o).getLsbD i = true ↔ LegalMove p o i := by
  rw [getMovesAvx512_eq_avx2]
  exact getMovesAvx2_exact p o i

theorem getMovesAvx512_eq_portable (p o : Bits) : getMovesAvx512 p o = getMovesPortable p o :=
  (getMovesAvx512_eq_avx2 p o).trans (getMovesAvx2_eq_portable p o)

def Avx512MovesContract : Prop :=
  ∀ p o i, (getMovesAvx512 p o).getLsbD i = true ↔ LegalMove p o i

theorem avx512MovesContract_proven : Avx512MovesContract := by
  intro p o i
  exact getMovesAvx512_exact p o i

end NeuralReversi.Bitboard.Moves

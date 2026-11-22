import NeuralReversi.Stability.Semantics

namespace NeuralReversi.Stability.Semantics

def boardIndex (x : Square) : Fin 64 := ⟨8 * x.1.val + x.2.val, by omega⟩
def boardSquare (i : Fin 64) : Square :=
  (⟨i.val / 8, by omega⟩, ⟨i.val % 8, by omega⟩)

theorem boardSquare_index (x : Square) : boardSquare (boardIndex x) = x := by
  apply Prod.ext <;> apply Fin.ext <;> simp only [boardSquare, boardIndex] <;> omega

theorem boardIndex_square (i : Fin 64) : boardIndex (boardSquare i) = i := by
  apply Fin.ext
  simp only [boardSquare, boardIndex]
  omega

theorem boardIndex_row (x : Square) : (boardIndex x).val / 8 = x.1.val := by
  change (8 * x.1.val + x.2.val) / 8 = x.1.val
  omega

theorem boardIndex_col (x : Square) : (boardIndex x).val % 8 = x.2.val := by
  change (8 * x.1.val + x.2.val) % 8 = x.2.val
  omega


end NeuralReversi.Stability.Semantics

import NeuralReversi.Stability.Model
namespace NeuralReversi.Stability

def sameLine (axis : Fin 4) (s t : Nat) : Prop :=
  match axis.val with
  | 0 => s / 8 = t / 8
  | 1 => s % 8 = t % 8
  | 2 => s / 8 + s % 8 = t / 8 + t % 8
  | _ => s / 8 + t % 8 = t / 8 + s % 8
instance (a : Fin 4) (s t : Nat) : Decidable (sameLine a s t) := by
  unfold sameLine; split <;> infer_instance

def lineCells (axis : Fin 4) (sq : Nat) : List Nat :=
  (List.range 64).filter (fun t => decide (sameLine axis sq t))
def lineBits (disc : Word) (axis : Fin 4) (sq : Nat) : Bool :=
  (lineCells axis sq).all disc.getLsbD

def FullAt (disc : Word) (axis : Fin 4) (sq : Fin 64) : Prop :=
  ∀ t : Fin 64, sameLine axis sq.val t.val → disc.getLsbD t.val = true

def field (f : FullLines) (axis : Fin 4) : Word :=
  match axis.val with | 0 => f.h | 1 => f.v | 2 => f.d7 | _ => f.d9

theorem lineBits_iff (disc : Word) (axis : Fin 4) (sq : Fin 64) :
    lineBits disc axis sq.val = true ↔ FullAt disc axis sq := by
  simp only [lineBits, lineCells, List.all_eq_true, List.mem_filter, List.mem_range,
    decide_eq_true_eq, FullAt]
  constructor
  · intro h t hl; exact h t.val ⟨t.isLt, hl⟩
  · intro h t ht; exact h ⟨t, ht.1⟩ ht.2

end NeuralReversi.Stability

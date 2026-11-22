import NeuralReversi.Stability.Structural
import NeuralReversi.Stability.StructuralCoordinates
import NeuralReversi.Stability.Geometry

namespace NeuralReversi.Stability.Semantics

def directionAxis : Direction → Fin 4
  | .e | .w => 0
  | .n | .s => 1
  | .ne | .sw => 2
  | .nw | .se => 3

def Interior (x : Square) : Prop :=
  1 ≤ x.1.val ∧ x.1.val ≤ 6 ∧ 1 ≤ x.2.val ∧ x.2.val ≤ 6

def axisStride (axis : Fin 4) : Nat :=
  match axis.val with | 0 => 1 | 1 => 8 | 2 => 7 | _ => 9

theorem axis_offset_neighbor {x y : Square} (d : Direction) (inside : Interior x)
    (offset : (boardIndex y).val = (boardIndex x).val + axisStride (directionAxis d) ∨
      (boardIndex y).val + axisStride (directionAxis d) = (boardIndex x).val) :
    Along x y d 1 ∨ Along y x d 1 := by
  have yr := y.1.isLt
  have yc := y.2.isLt
  unfold Interior at inside
  cases d <;>
    simp [boardIndex, axisStride, directionAxis, Along, Direction.delta] at offset ⊢ <;> omega

theorem along_sameLine {a x : Square} {d : Direction} {k : Nat} (ray : Along a x d k) :
    NeuralReversi.Stability.sameLine (directionAxis d) (boardIndex x).val (boardIndex a).val := by
  have ar := a.1.isLt
  have ac := a.2.isLt
  have xr := x.1.isLt
  have xc := x.2.isLt
  cases d <;>
    simp [NeuralReversi.Stability.sameLine, directionAxis, boardIndex_row, boardIndex_col]
    <;> simp only [Along, Direction.delta] at ray <;>
    omega

theorem fullAt_guard {s : State} {disc : NeuralReversi.Stability.Word} {x : Square}
    (occupied : ∀ y, disc.getLsbD (boardIndex y).val = true → Occupied s.board y)
    (d : Direction) (full : NeuralReversi.Stability.FullAt disc (directionAxis d) (boardIndex x)) :
    ∀ a k, Along a x d k → Occupied s.board a := by
  intro a k ray
  exact occupied a (full (boardIndex a) (along_sameLine ray))

theorem four_full_stable {s : State} {disc : NeuralReversi.Stability.Word} {c : Color} {x : Square}
    (initial : s.board x = some c)
    (occupied : ∀ y, disc.getLsbD (boardIndex y).val = true → Occupied s.board y)
    (full : ∀ axis, NeuralReversi.Stability.FullAt disc axis (boardIndex x)) : Stable s c x := by
  apply full_axes_stable initial
  intro d
  exact fullAt_guard occupied d (full (directionAxis d))

theorem fullAt_or_neighbor_guard {s : State} {disc : NeuralReversi.Stability.Word}
    {known : Square → Prop} {x : Square}
    (occupied : ∀ y, disc.getLsbD (boardIndex y).val = true → Occupied s.board y)
    (witness : ∀ d, NeuralReversi.Stability.FullAt disc (directionAxis d) (boardIndex x) ∨
      ∃ y, (Along x y d 1 ∨ Along y x d 1) ∧ known y) : GuardedBy s known x := by
  intro d
  rcases witness d with full | neighbor
  · exact Or.inl (fullAt_guard occupied d full)
  · exact Or.inr neighbor

theorem mask_guard {s : State} {disc known : NeuralReversi.Stability.Word} (sq : Fin 64)
    (inside : Interior (boardSquare sq))
    (occupied : ∀ y, disc.getLsbD (boardIndex y).val = true → Occupied s.board y)
    (support : ∀ axis : Fin 4,
      NeuralReversi.Stability.FullAt disc axis sq ∨
      ∃ j : Fin 64, (j.val = sq.val + axisStride axis ∨ j.val + axisStride axis = sq.val) ∧
        known.getLsbD j.val = true) :
    GuardedBy s (fun y => known.getLsbD (boardIndex y).val = true) (boardSquare sq) := by
  apply fullAt_or_neighbor_guard occupied
  intro d
  rcases support (directionAxis d) with full | ⟨j, offset, marked⟩
  · exact Or.inl (by simpa only [boardIndex_square] using full)
  · refine Or.inr ⟨boardSquare j, ?_, ?_⟩
    · apply axis_offset_neighbor d inside
      simpa only [boardIndex_square] using offset
    · simpa only [boardIndex_square] using marked

theorem mask_contact_stable {s : State} {c : Color}
    {disc known : NeuralReversi.Stability.Word} (sq : Fin 64)
    (inside : Interior (boardSquare sq)) (initial : s.board (boardSquare sq) = some c)
    (occupied : ∀ y, disc.getLsbD (boardIndex y).val = true → Occupied s.board y)
    (knownSound : ∀ j : Fin 64, known.getLsbD j.val = true → Stable s c (boardSquare j))
    (support : ∀ axis : Fin 4,
      NeuralReversi.Stability.FullAt disc axis sq ∨
      ∃ j : Fin 64, (j.val = sq.val + axisStride axis ∨ j.val + axisStride axis = sq.val) ∧
        known.getLsbD j.val = true) : Stable s c (boardSquare sq) := by
  apply contact_stable initial
  apply guardedBy_sound ?_ (mask_guard sq inside occupied support)
  intro y marked
  simpa only [boardSquare_index] using knownSound (boardIndex y) marked

end NeuralReversi.Stability.Semantics

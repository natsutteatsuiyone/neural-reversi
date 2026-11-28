module
public import NeuralReversi.Stability.StructuralBridge

@[expose] public section

namespace NeuralReversi.Stability.Semantics

def SafeAxis (s : State) (c : Color) (axis : Fin 4) (x : Square) : Prop :=
  ∀ t, Reach s t → ∀ a, LegalMove t.board t.turn a → t.turn = Opposite c →
    ∀ d, directionAxis d = axis → ¬CapturedAlong t.board t.turn a x d

private theorem empty_before {s t : State} (reach : Reach s t) {a : Square}
    (empty : t.board a = none) : s.board a = none := by
  cases h : s.board a with
  | none => rfl
  | some c =>
    obtain ⟨_, occupied⟩ := reach_occupied reach (show Occupied s.board a from ⟨c, h⟩)
    simp [empty] at occupied

private theorem color_or_move {s t : State} {c : Color} {x : Square} (reach : Reach s t)
    (initial : s.board x = some c) (origin : Square → Prop)
    (rule : ∀ u, Reach s u → ∀ a, LegalMove u.board u.turn a → u.turn = Opposite c →
      Captured u.board u.turn a x → origin a) :
    t.board x = some c ∨ ∃ a, origin a ∧ Occupied t.board a := by
  induction reach with
  | refl => exact Or.inl initial
  | @tail t u prior step ih =>
    rcases ih with color | ⟨a, allowed, occ⟩
    · cases step with
      | pass => exact Or.inl color
      | move a legal =>
        have ne : x ≠ a := fun eq => by simp [eq, legal.1] at color
        classical
        by_cases cap : Captured t.board t.turn a x
        · have opposite : t.turn = Opposite c := by
            have colors := color.symm.trans (captured_color cap)
            cases c <;> cases ht : t.turn <;> simp_all [Opposite]
          exact Or.inr ⟨a, rule t prior a legal opposite cap, t.turn, by simp [place]⟩
        · exact Or.inl (by simpa [place, ne, cap] using color)
    · exact Or.inr ⟨a, allowed, step_occupied step occ⟩

theorem stable_of_safe_axes {s : State} {c : Color} {x : Square}
    (initial : s.board x = some c) (safe : ∀ axis, SafeAxis s c axis x) : Stable s c x := by
  intro t reach
  apply (color_or_move reach initial (fun _ => False) ?_).resolve_right
  · rintro ⟨_, impossible, _⟩
    exact impossible
  · rintro u prior a legal turn ⟨d, cap⟩
    exact safe _ u prior a legal turn d rfl cap

theorem full_safe_axis {s : State} {c : Color} {x : Square} {axis : Fin 4}
    (full : ∀ a, sameLine axis (boardIndex x).val (boardIndex a).val → Occupied s.board a) :
    SafeAxis s c axis x := by
  rintro t reached a legal _ d rfl ⟨_, _, _, _, _, k, _, _, ray⟩
  obtain ⟨_, occ⟩ := full a (along_sameLine ray)
  simp [empty_before reached legal.1] at occ

private theorem bracket_offset_neighbor {a x y z : Square} {d : Direction} {k n : Nat}
    (hk : 1 ≤ k) (hkn : k ≤ n) (ray : Along a x d k) (endRay : Along a z d (n + 1))
    (offset : (boardIndex y).val = (boardIndex x).val + axisStride (directionAxis d) ∨
      (boardIndex y).val + axisStride (directionAxis d) = (boardIndex x).val) :
    Along x y d 1 ∨ Along y x d 1 := by
  have := a.1.isLt; have := a.2.isLt
  have := y.1.isLt; have := y.2.isLt
  have := z.1.isLt; have := z.2.isLt
  cases d <;> simp [boardIndex, axisStride, directionAxis, Along, Direction.delta] at * <;> omega

theorem contact_safe_axis {s : State} {c : Color} {x y : Square} {axis : Fin 4}
    (offset : (boardIndex y).val = (boardIndex x).val + axisStride axis ∨
      (boardIndex y).val + axisStride axis = (boardIndex x).val)
    (stable : Stable s c y) : SafeAxis s c axis x := by
  rintro t reached a legal turn d rfl ⟨n, hn, hn6, endpoint, run, k, hk, hkn, ray⟩
  have adj : Along x y d 1 ∨ Along y x d 1 := by
    obtain ⟨z, endRay, _⟩ := endpoint
    exact bracket_offset_neighbor hk hkn ray endRay offset
  have color : t.board y = some (Opposite t.turn) := by
    simpa [turn, Opposite] using stable t reached
  apply stable_not_captured stable reached legal turn
  rcases adj with fwd | back
  · exact captured_neighbor_forward hn hn6 endpoint run hk hkn ray fwd color
  · exact captured_neighbor_backward legal.1 hn hn6 endpoint run hk hkn ray back color

private theorem line_trans {axis : Fin 4} {x y z : Nat}
    (xy : sameLine axis x y) (yz : sameLine axis y z) : sameLine axis x z := by
  rcases axis with ⟨i, hi⟩
  match i with
  | 0 | 1 | 2 | _ + 3 => simp [sameLine] at * <;> omega

def Between (a y x : Nat) : Prop := (a < y ∧ y < x) ∨ (x < y ∧ y < a)
instance (a y x : Nat) : Decidable (Between a y x) := by unfold Between; infer_instance

private theorem between_on_ray {a y x : Square} {d : Direction} {k : Nat}
    (ray : Along a x d k)
    (line : sameLine (directionAxis d) (boardIndex x).val (boardIndex y).val)
    (between : Between (boardIndex a).val (boardIndex y).val (boardIndex x).val) :
    ∃ j, 1 ≤ j ∧ j < k ∧ Along a y d j := by
  have := a.2.isLt; have := x.2.isLt; have := y.2.isLt
  refine ⟨if d.delta.1 = 0 then max (a.2.val - y.2.val) (y.2.val - a.2.val)
    else max (a.1.val - y.1.val) (y.1.val - a.1.val), ?_⟩
  cases d <;>
    simp [sameLine, directionAxis, boardIndex_row, boardIndex_col] at line <;>
    simp [Between, boardIndex] at between <;> simp [Along, Direction.delta] at ray ⊢ <;> omega

theorem shield_safe_axis {s : State} {c : Color} {axis : Fin 4} {x hole shield : Square}
    (unique : ∀ z, sameLine axis (boardIndex x).val (boardIndex z).val →
      s.board z = none → z = hole)
    (line : sameLine axis (boardIndex x).val (boardIndex shield).val)
    (between : Between (boardIndex hole).val (boardIndex shield).val (boardIndex x).val)
    (initial : s.board shield = some (Opposite c))
    (other : ∀ b, b ≠ axis → SafeAxis s (Opposite c) b shield) : SafeAxis s c axis x := by
  rintro t reached a legal turn d rfl ⟨n, _, _, _, run, k, _, hkn, ray⟩
  have eq := unique a (along_sameLine ray) (empty_before reached legal.1)
  subst hole
  have color := color_or_move reached initial (· = a) (by
    rintro u prior z move opp ⟨dir, cap⟩
    by_cases same : directionAxis dir = directionAxis d
    · obtain ⟨_, _, _, _, _, j, _, _, path⟩ := cap
      exact unique z (line_trans line (same ▸ along_sameLine path)) (empty_before prior move.1)
    · exact False.elim (other _ same u prior z move opp dir rfl cap))
  rcases color with kept | ⟨_, rfl, col, occupied⟩
  · obtain ⟨j, hj, hjk, path⟩ := between_on_ray ray line between
    obtain ⟨y, hy, flipped⟩ := run j hj (by omega)
    rw [along_unique hy path, kept, turn] at flipped
    cases c <;> simp [Opposite] at flipped
  · simp [legal.1] at occupied

end NeuralReversi.Stability.Semantics

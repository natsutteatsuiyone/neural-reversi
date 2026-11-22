import Std

namespace NeuralReversi.Stability.Semantics
abbrev Square := Fin 8 × Fin 8
abbrev Color := Bool
abbrev Board := Square → Option Color
def Opposite (c : Color) : Color := !c
inductive Direction where
  | n | ne | e | se | s | sw | w | nw
  deriving DecidableEq, Repr
def Direction.delta : Direction → Int × Int
  | .n => (-1, 0) | .ne => (-1, 1) | .e => (0, 1) | .se => (1, 1)
  | .s => (1, 0) | .sw => (1, -1) | .w => (0, -1) | .nw => (-1, -1)
def Along (a x : Square) (d : Direction) (k : Nat) : Prop :=
  (x.1.val : Int) = a.1.val + (k : Int) * d.delta.1 ∧
  (x.2.val : Int) = a.2.val + (k : Int) * d.delta.2
def Captured (b : Board) (c : Color) (a x : Square) : Prop :=
  ∃ d : Direction, ∃ n : Nat, 1 ≤ n ∧ n ≤ 6 ∧
    (∃ endpoint : Square, Along a endpoint d (n+1) ∧ b endpoint = some c) ∧
    (∀ k : Nat, 1 ≤ k → k ≤ n →
      ∃ y : Square, Along a y d k ∧ b y = some (Opposite c)) ∧
    (∃ k : Nat, 1 ≤ k ∧ k ≤ n ∧ Along a x d k)
noncomputable def place (b : Board) (c : Color) (a : Square) : Board := by
  classical
  exact fun x => if x = a ∨ Captured b c a x then some c else b x
def LegalMove (b : Board) (c : Color) (a : Square) : Prop :=
  b a = none ∧ ∃ x, Captured b c a x
structure State where
  board : Board
  turn : Color
inductive Step : State → State → Prop
  | move (s : State) (a : Square) (h : LegalMove s.board s.turn a) :
      Step s ⟨place s.board s.turn a, Opposite s.turn⟩
  | pass (s : State)
      (hno : ∀ a, ¬ LegalMove s.board s.turn a)
      (hyes : ∃ a, LegalMove s.board (Opposite s.turn) a) :
      Step s ⟨s.board, Opposite s.turn⟩
inductive Reach : State → State → Prop
  | refl (s) : Reach s s
  | tail {s t u} : Reach s t → Step t u → Reach s u
def Occupied (b : Board) (x : Square) : Prop := ∃ c, b x = some c
def Stable (s : State) (c : Color) (x : Square) : Prop :=
  ∀ t, Reach s t → t.board x = some c
theorem along_unique {a x y : Square} {d : Direction} {k : Nat}
    (hx : Along a x d k) (hy : Along a y d k) : x = y := by
  rcases x with ⟨xr, xc⟩
  rcases y with ⟨yr, yc⟩
  simp only [Along] at hx hy
  have hr : xr.val = yr.val := by omega
  have hc : xc.val = yc.val := by omega
  exact Prod.ext (Fin.ext hr) (Fin.ext hc)
theorem captured_color {b : Board} {c : Color} {a x : Square}
    (h : Captured b c a x) : b x = some (Opposite c) := by
  rcases h with ⟨d,n,_,_,_,hr,k,hk,hkn,hx⟩
  obtain ⟨y,hy,hcolor⟩ := hr k hk hkn
  have : x = y := along_unique hx hy
  simpa [this] using hcolor
theorem place_occupied {b : Board} {c : Color} {a x : Square}
    (h : Occupied b x) : Occupied (place b c a) x := by
  classical
  unfold Occupied place
  split
  · exact ⟨c,rfl⟩
  · exact h
theorem step_occupied {s t : State} (hs : Step s t) {x : Square}
    (h : Occupied s.board x) : Occupied t.board x := by
  cases hs with
  | move a _ => exact place_occupied h
  | pass _ _ => exact h
theorem reach_occupied {s t : State} (hs : Reach s t) {x : Square}
    (h : Occupied s.board x) : Occupied t.board x := by
  induction hs with
  | refl => exact h
  | tail _ hstep ih => exact step_occupied hstep ih
theorem reach_trans {s t u : State} (hst : Reach s t) (htu : Reach t u) : Reach s u := by
  induction htu with
  | refl => exact hst
  | tail _ hstep ih => exact Reach.tail ih hstep
theorem stable_now {s : State} {c : Color} {x : Square} (h : Stable s c x) :
    s.board x = some c := h s (.refl s)
theorem stable_future {s t : State} {c : Color} {x : Square}
    (h : Stable s c x) (hst : Reach s t) : Stable t c x := by
  intro u htu
  exact h u (reach_trans hst htu)
theorem stable_of_preserved {s : State} {c : Color} {x : Square}
    (hinit : s.board x = some c)
    (hstep : ∀ t u, Reach s t → Step t u →
      t.board x = some c → u.board x = some c) : Stable s c x := by
  intro t hst
  induction hst with
  | refl => exact hinit
  | tail hreach hmove ih => exact hstep _ _ hreach hmove ih
theorem stable_not_captured {s t : State} {c : Color} {x a : Square}
    (hs : Stable s c x) (hst : Reach s t)
    (hmove : LegalMove t.board t.turn a) (ht : t.turn = Opposite c) :
    ¬ Captured t.board t.turn a x := by
  intro hcap
  have hf := hs _ (Reach.tail hst (Step.move t a hmove))
  classical
  have hx : (place t.board t.turn a) x = some t.turn := by
    simp [place, hcap]
  change (place t.board t.turn a) x = some c at hf
  rw [hx] at hf
  have hc : t.turn = c := Option.some.inj hf
  cases c <;> simp [Opposite] at ht hc <;> simp_all


end NeuralReversi.Stability.Semantics

module
public import NeuralReversi.Stability.EdgeGeometry

@[expose] public section

namespace NeuralReversi.Stability.Semantics

abbrev Line := Fin 8 → Option Color

noncomputable def linePlace (line : Line) (c : Color) (a : Fin 8) : Line := by
  classical
  exact fun x => if x = a ∨ LineCaptured line c a x then some c else line x

inductive LineStep : Line → Line → Prop
  | place (line : Line) (c : Color) (a : Fin 8) (empty : line a = none) :
      LineStep line (linePlace line c a)

inductive LineReach : Line → Line → Prop
  | refl (line) : LineReach line line
  | tail {a b c} : LineReach a b → LineStep b c → LineReach a c

theorem lineReach_trans {a b c : Line} (hab : LineReach a b) (hbc : LineReach b c) :
    LineReach a c := by
  induction hbc with
  | refl => exact hab
  | tail _ step ih => exact .tail ih step

theorem row_place_line (b : Board) (c : Color) (r a : Fin 8)
    (edge : r.val = 0 ∨ r.val = 7) :
    (fun i => place b c (r,a) (r,i)) = linePlace (fun i => b (r,i)) c a := by
  classical
  funext i
  simp only [place, linePlace, Prod.mk.injEq, true_and,
    row_captured_iff b c r a i edge]

theorem col_place_line (b : Board) (c : Color) (r a : Fin 8)
    (edge : r.val = 0 ∨ r.val = 7) :
    (fun i => place b c (a,r) (i,r)) = linePlace (fun i => b (i,r)) c a := by
  classical
  funext i
  simp only [place, linePlace, Prod.mk.injEq, and_true,
    col_captured_iff b c r a i edge]

theorem step_row_line {s t : State} (h : Step s t) (r : Fin 8)
    (edge : r.val = 0 ∨ r.val = 7) :
    LineReach (fun i => s.board (r,i)) (fun i => t.board (r,i)) := by
  cases h with
  | pass _ _ => exact .refl _
  | move a legal =>
    by_cases same : a.1 = r
    · have ae : a = (r,a.2) := Prod.ext same rfl
      rw [ae] at legal ⊢
      change LineReach _ (fun i => place s.board s.turn (r,a.2) (r,i))
      rw [row_place_line _ _ _ _ edge]
      exact .tail (.refl _) (.place _ _ _ legal.1)
    · have unchanged : (fun i => place s.board s.turn a (r,i)) =
          (fun i => s.board (r,i)) := by
        funext i
        exact place_off_row_edge edge same
      change LineReach _ (fun i => place s.board s.turn a (r,i))
      rw [unchanged]
      exact .refl _

theorem step_col_line {s t : State} (h : Step s t) (r : Fin 8)
    (edge : r.val = 0 ∨ r.val = 7) :
    LineReach (fun i => s.board (i,r)) (fun i => t.board (i,r)) := by
  cases h with
  | pass _ _ => exact .refl _
  | move a legal =>
    by_cases same : a.2 = r
    · have ae : a = (a.1,r) := Prod.ext rfl same
      rw [ae] at legal ⊢
      change LineReach _ (fun i => place s.board s.turn (a.1,r) (i,r))
      rw [col_place_line _ _ _ _ edge]
      exact .tail (.refl _) (.place _ _ _ legal.1)
    · have unchanged : (fun i => place s.board s.turn a (i,r)) =
          (fun i => s.board (i,r)) := by
        funext i
        exact place_off_col_edge edge same
      change LineReach _ (fun i => place s.board s.turn a (i,r))
      rw [unchanged]
      exact .refl _

theorem reach_row_line {s t : State} (h : Reach s t) (r : Fin 8)
    (edge : r.val = 0 ∨ r.val = 7) :
    LineReach (fun i => s.board (r,i)) (fun i => t.board (r,i)) := by
  induction h with
  | refl => exact .refl _
  | tail _ step ih => exact lineReach_trans ih (step_row_line step r edge)

theorem reach_col_line {s t : State} (h : Reach s t) (r : Fin 8)
    (edge : r.val = 0 ∨ r.val = 7) :
    LineReach (fun i => s.board (i,r)) (fun i => t.board (i,r)) := by
  induction h with
  | refl => exact .refl _
  | tail _ step ih => exact lineReach_trans ih (step_col_line step r edge)

end NeuralReversi.Stability.Semantics

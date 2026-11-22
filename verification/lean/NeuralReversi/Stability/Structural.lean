import NeuralReversi.Stability.Semantics

namespace NeuralReversi.Stability.Semantics

def FullAxes (b : Board) (x : Square) : Prop :=
  ∀ (d : Direction) (a : Square) (k : Nat), Along a x d k → Occupied b a

theorem full_axes_stable {s : State} {c : Color} {x : Square}
    (hx : s.board x = some c) (full : FullAxes s.board x) : Stable s c x := by
  apply stable_of_preserved hx
  intro t u hreach hstep hcolor
  cases hstep with
  | pass _ _ => exact hcolor
  | move a hlegal =>
    have hne : x ≠ a := by
      intro heq
      rw [heq, hlegal.1] at hcolor
      cases hcolor
    have hncap : ¬ Captured t.board t.turn a x := by
      rintro ⟨d,n,_,_,_,_,k,_,_,hax⟩
      obtain ⟨col,hcol⟩ := reach_occupied hreach (full d a k hax)
      rw [hlegal.1] at hcol
      cases hcol
    classical
    change (place t.board t.turn a) x = some c
    simpa [place, hne, hncap] using hcolor

theorem along_zero {a x : Square} (d : Direction) (h : Along a x d 0) : x = a := by
  apply Prod.ext <;> apply Fin.ext <;> unfold Along at h <;> simp at h <;> omega

theorem along_succ {a x y : Square} {d : Direction} {k : Nat}
    (hax : Along a x d k) (hxy : Along x y d 1) : Along a y d (k+1) := by
  cases d <;> simp [Along, Direction.delta] at * <;> omega

theorem along_pred {a x y : Square} {d : Direction} {k : Nat} (hk : 1 ≤ k)
    (hax : Along a x d k) (hyx : Along y x d 1) : Along a y d (k-1) := by
  cases d <;> simp [Along, Direction.delta] at * <;> omega

theorem captured_neighbor_forward {b : Board} {c : Color} {a x y : Square}
    {d : Direction} {n k : Nat}
    (hn : 1 ≤ n) (hn6 : n ≤ 6)
    (hend : ∃ endpoint, Along a endpoint d (n+1) ∧ b endpoint = some c)
    (hr : ∀ j, 1 ≤ j → j ≤ n → ∃ z, Along a z d j ∧ b z = some (Opposite c))
    (hk : 1 ≤ k) (hkn : k ≤ n) (hax : Along a x d k)
    (hxy : Along x y d 1) (hy : b y = some (Opposite c)) : Captured b c a y := by
  have hay := along_succ hax hxy
  have hlt : k < n := by
    apply Classical.byContradiction
    intro h
    have heq : k = n := by omega
    obtain ⟨endpoint, he, hc⟩ := hend
    have heqy : y = endpoint := along_unique (heq ▸ hay) he
    rw [heqy, hc] at hy
    cases c <;> simp [Opposite] at hy
  exact ⟨d,n,hn,hn6,hend,hr,k+1,by omega,by omega,hay⟩

theorem captured_neighbor_backward {b : Board} {c : Color} {a x y : Square}
    {d : Direction} {n k : Nat}
    (ha : b a = none) (hn : 1 ≤ n) (hn6 : n ≤ 6)
    (hend : ∃ endpoint, Along a endpoint d (n+1) ∧ b endpoint = some c)
    (hr : ∀ j, 1 ≤ j → j ≤ n → ∃ z, Along a z d j ∧ b z = some (Opposite c))
    (hk : 1 ≤ k) (hkn : k ≤ n) (hax : Along a x d k)
    (hyx : Along y x d 1) (hy : b y = some (Opposite c)) : Captured b c a y := by
  have hay := along_pred hk hax hyx
  have hgt : 1 < k := by
    apply Classical.byContradiction
    intro h
    have heq : k = 1 := by omega
    have hay0 : Along a y d 0 := by simpa [heq] using hay
    have heqy : y = a := along_zero d hay0
    rw [heqy, ha] at hy
    cases hy
  exact ⟨d,n,hn,hn6,hend,hr,k-1,by omega,by omega,hay⟩

def Guarded (s : State) (c : Color) (x : Square) : Prop :=
  ∀ d : Direction,
    (∀ a k, Along a x d k → Occupied s.board a) ∨
    ∃ y, (Along x y d 1 ∨ Along y x d 1) ∧ Stable s c y

theorem contact_stable {s : State} {c : Color} {x : Square}
    (initial : s.board x = some c) (guard : Guarded s c x) : Stable s c x := by
  apply stable_of_preserved initial
  intro t u hreach hstep hcolor
  cases hstep with
  | pass _ _ => exact hcolor
  | move a hlegal =>
    have hne : x ≠ a := by
      intro heq
      rw [heq, hlegal.1] at hcolor
      cases hcolor
    have hncap : ¬ Captured t.board t.turn a x := by
      intro hcap
      have hv := captured_color hcap
      have hturn : t.turn = Opposite c := by
        rw [hcolor] at hv
        cases hc : c <;> cases ht : t.turn <;> simp_all [Opposite]
      rcases hcap with ⟨d,n,hn,hn6,hend,hr,k,hk,hkn,hax⟩
      rcases guard d with hfull | ⟨y,hadj,hstable⟩
      · obtain ⟨col,hcol⟩ := reach_occupied hreach (hfull a k hax)
        rw [hlegal.1] at hcol
        cases hcol
      · have hy : t.board y = some (Opposite t.turn) := by
          have hys := hstable t hreach
          simpa [hturn,Opposite] using hys
        have hycap : Captured t.board t.turn a y := by
          rcases hadj with hforward | hbackward
          · exact captured_neighbor_forward hn hn6 hend hr hk hkn hax hforward hy
          · exact captured_neighbor_backward hlegal.1 hn hn6 hend hr hk hkn hax hbackward hy
        exact stable_not_captured hstable hreach hlegal hturn hycap
    classical
    change (place t.board t.turn a) x = some c
    simpa [place, hne, hncap] using hcolor

theorem row_edge_capture_parallel {a x endpoint : Square} {d : Direction} {n k : Nat}
    (hxedge : x.1.val = 0 ∨ x.1.val = 7)
    (hk : 1 ≤ k) (hkn : k ≤ n)
    (hx : Along a x d k) (he : Along a endpoint d (n+1)) : d.delta.1 = 0 := by
  have ha0 := a.1.isLt
  have he0 := endpoint.1.isLt
  cases d <;> simp [Along,Direction.delta] at * <;> omega

theorem col_edge_capture_parallel {a x endpoint : Square} {d : Direction} {n k : Nat}
    (hxedge : x.2.val = 0 ∨ x.2.val = 7)
    (hk : 1 ≤ k) (hkn : k ≤ n)
    (hx : Along a x d k) (he : Along a endpoint d (n+1)) : d.delta.2 = 0 := by
  have ha0 := a.2.isLt
  have he0 := endpoint.2.isLt
  cases d <;> simp [Along,Direction.delta] at * <;> omega
theorem corner_not_captured {b : Board} {c : Color} {a x : Square}
    (row : x.1.val = 0 ∨ x.1.val = 7) (col : x.2.val = 0 ∨ x.2.val = 7) :
    ¬ Captured b c a x := by
  rintro ⟨d, n, _, _, ⟨endpoint, he, _⟩, _, k, hk, hkn, hx⟩
  have hr := row_edge_capture_parallel row hk hkn hx he
  have hc := col_edge_capture_parallel col hk hkn hx he
  cases d <;> simp [Direction.delta] at hr hc

theorem corner_stable {s : State} {c : Color} {x : Square}
    (initial : s.board x = some c)
    (row : x.1.val = 0 ∨ x.1.val = 7) (col : x.2.val = 0 ∨ x.2.val = 7) : Stable s c x := by
  apply stable_of_preserved initial
  intro t u _ step color
  cases step with
  | pass _ _ => exact color
  | move a legal =>
    have notPlaced : x ≠ a := by
      intro eq
      rw [eq, legal.1] at color
      cases color
    have notCaptured : ¬ Captured t.board t.turn a x := corner_not_captured row col
    classical
    change place t.board t.turn a x = some c
    simpa [place, notPlaced, notCaptured] using color

def GuardedBy (s : State) (known : Square → Prop) (x : Square) : Prop :=
  ∀ d : Direction,
    (∀ a k, Along a x d k → Occupied s.board a) ∨
    ∃ y, (Along x y d 1 ∨ Along y x d 1) ∧ known y

theorem guardedBy_sound {s : State} {c : Color} {known : Square → Prop} {x : Square}
    (knownSound : ∀ y, known y → Stable s c y) (guard : GuardedBy s known x) :
    Guarded s c x := by
  intro d
  rcases guard d with full | ⟨y, neighbor, marked⟩
  · exact Or.inl full
  · exact Or.inr ⟨y, neighbor, knownSound y marked⟩

theorem expansion_sound {s : State} {c : Color} {known next : Square → Prop}
    (knownSound : ∀ x, known x → Stable s c x)
    (rule : ∀ x, next x → known x ∨ (s.board x = some c ∧ GuardedBy s known x)) :
    ∀ x, next x → Stable s c x := by
  intro x hx
  rcases rule x hx with old | ⟨initial, guard⟩
  · exact knownSound x old
  · exact contact_stable initial (guardedBy_sound knownSound guard)

def ContactClosure (s : State) (c : Color) (seed : Square → Prop) : Nat → Square → Prop
  | 0, x => seed x
  | n + 1, x => ContactClosure s c seed n x ∨
      (s.board x = some c ∧ GuardedBy s (ContactClosure s c seed n) x)

theorem closure_monotone (s : State) (c : Color) (seed : Square → Prop) (n : Nat) :
    ∀ x, ContactClosure s c seed n x → ContactClosure s c seed (n + 1) x := by
  intro x hx
  exact Or.inl hx

theorem closure_sound {s : State} {c : Color} {seed : Square → Prop}
    (seedSound : ∀ x, seed x → Stable s c x) (n : Nat) :
    ∀ x, ContactClosure s c seed n x → Stable s c x := by
  induction n with
  | zero => exact seedSound
  | succ n ih => exact expansion_sound ih (fun _ hx => hx)

theorem closure_never_flips {s t : State} {c : Color} {seed : Square → Prop}
    (seedSound : ∀ x, seed x → Stable s c x) {n : Nat} {x : Square}
    (marked : ContactClosure s c seed n x) (continuation : Reach s t) :
    t.board x = some c := closure_sound seedSound n x marked t continuation

end NeuralReversi.Stability.Semantics

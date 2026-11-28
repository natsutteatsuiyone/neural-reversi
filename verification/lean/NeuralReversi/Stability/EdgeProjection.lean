module
public import NeuralReversi.Stability.Structural

@[expose] public section

namespace NeuralReversi.Stability.Semantics
theorem captured_row_same {b : Board} {c : Color} {a x : Square}
 (hedge : x.1.val = 0 ∨ x.1.val = 7) (hc : Captured b c a x) : a.1 = x.1 := by
 obtain ⟨d,n,_,_,⟨ep,he,_⟩,_,k,hk,hkn,hx⟩ := hc
 have hd := row_edge_capture_parallel hedge hk hkn hx he
 have hr := hx.1
 rw [hd] at hr
 simp only [Int.mul_zero, Int.add_zero] at hr
 apply Fin.ext
 omega

theorem captured_col_same {b : Board} {c : Color} {a x : Square}
 (hedge : x.2.val = 0 ∨ x.2.val = 7) (hc : Captured b c a x) : a.2 = x.2 := by
 obtain ⟨d,n,_,_,⟨ep,he,_⟩,_,k,hk,hkn,hx⟩ := hc
 have hd := col_edge_capture_parallel hedge hk hkn hx he
 have hr := hx.2
 rw [hd] at hr
 simp only [Int.mul_zero, Int.add_zero] at hr
 apply Fin.ext
 omega

theorem place_off_row_edge {b : Board} {c : Color} {a x : Square}
 (hedge : x.1.val = 0 ∨ x.1.val = 7) (hne : a.1 ≠ x.1) :
 place b c a x = b x := by
 classical
 have hx : x ≠ a := by intro h; subst x; exact hne rfl
 have hc : ¬ Captured b c a x := fun h => hne (captured_row_same hedge h)
 simp [place, hx, hc]

theorem place_off_col_edge {b : Board} {c : Color} {a x : Square}
 (hedge : x.2.val = 0 ∨ x.2.val = 7) (hne : a.2 ≠ x.2) :
 place b c a x = b x := by
 classical
 have hx : x ≠ a := by intro h; subst x; exact hne rfl
 have hc : ¬ Captured b c a x := fun h => hne (captured_col_same hedge h)
 simp [place, hx, hc]
end NeuralReversi.Stability.Semantics

module
public import NeuralReversi.Stability.EdgeGeometry

@[expose] public section

namespace NeuralReversi.Stability.Semantics

def LineBracket (line : Fin 8 → Option Color) (c : Color) (a x : Fin 8) : Prop :=
  ∃ z : Fin 8, line z = some c ∧
    ((a.val < x.val ∧ x.val < z.val ∧
      ∀ y : Fin 8, a.val < y.val → y.val < z.val → line y = some (Opposite c)) ∨
     (z.val < x.val ∧ x.val < a.val ∧
      ∀ y : Fin 8, z.val < y.val → y.val < a.val → line y = some (Opposite c)))

theorem lineCaptured_bracket (line : Fin 8 → Option Color) (c : Color) (a x : Fin 8) :
    LineCaptured line c a x ↔ LineBracket line c a x := by
  rw [line_finite_iff]
  constructor
  · rintro ⟨positive,n,hn,⟨z,hz,hcolor⟩,run,k,hk,hkn,hx⟩
    refine ⟨z,hcolor,?_⟩
    cases positive with
    | true =>
      simp only [↓reduceIte, Int.mul_one] at hz hx run
      refine Or.inl ⟨by omega,by omega,?_⟩
      intro y hay hyz
      have hj : y.val - a.val < 8 := by omega
      obtain ⟨v,hv,hvc⟩ := run ⟨y.val-a.val,hj⟩ (by simp; omega) (by simp; omega)
      have eq : v = y := by apply Fin.ext; dsimp at hv; omega
      simpa [eq] using hvc
    | false =>
      simp only [Bool.false_eq_true, ↓reduceIte, Int.mul_neg, Int.mul_one] at hz hx run
      refine Or.inr ⟨by omega,by omega,?_⟩
      intro y hzy hya
      have hj : a.val - y.val < 8 := by omega
      obtain ⟨v,hv,hvc⟩ := run ⟨a.val-y.val,hj⟩ (by simp; omega) (by simp; omega)
      have eq : v = y := by apply Fin.ext; dsimp at hv; omega
      simpa [eq] using hvc
  · rintro ⟨z,hcolor,forward | backward⟩
    · rcases forward with ⟨hax,hxz,between⟩
      have hn : z.val-a.val-1 < 7 := by omega
      refine ⟨true,⟨z.val-a.val-1,hn⟩,by simp; omega,?_,?_,?_⟩
      · refine ⟨z,?_,hcolor⟩
        simp only [↓reduceIte, Int.mul_one]
        omega
      · intro j hj hjn
        have hy : a.val+j.val < 8 := by dsimp at hjn; omega
        refine ⟨⟨a.val+j.val,hy⟩,?_,between _ (by simp; omega) (by simp; dsimp at hjn; omega)⟩
        simp
      · have hk : x.val-a.val < 8 := by omega
        refine ⟨⟨x.val-a.val,hk⟩,by simp; omega,by simp; omega,?_⟩
        simp only [↓reduceIte, Int.mul_one]
        omega
    · rcases backward with ⟨hzx,hxa,between⟩
      have hn : a.val-z.val-1 < 7 := by omega
      refine ⟨false,⟨a.val-z.val-1,hn⟩,by simp; omega,?_,?_,?_⟩
      · refine ⟨z,?_,hcolor⟩
        simp only [Bool.false_eq_true, ↓reduceIte, Int.mul_neg, Int.mul_one]
        omega
      · intro j hj hjn
        have hy : a.val-j.val < 8 := by omega
        refine ⟨⟨a.val-j.val,hy⟩,?_,between _ (by simp; dsimp at hjn; omega) (by simp; omega)⟩
        simp only [Bool.false_eq_true, ↓reduceIte, Int.mul_neg, Int.mul_one]
        dsimp at hjn
        omega
      · have hk : a.val-x.val < 8 := by omega
        refine ⟨⟨a.val-x.val,hk⟩,by simp; omega,by simp; omega,?_⟩
        simp only [Bool.false_eq_true, ↓reduceIte, Int.mul_neg, Int.mul_one]
        omega

end NeuralReversi.Stability.Semantics

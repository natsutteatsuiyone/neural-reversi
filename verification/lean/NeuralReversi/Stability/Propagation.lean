import NeuralReversi.Stability.Geometry
namespace NeuralReversi.Stability

theorem contactStep_contains (central stable : Word) (full : FullLines) (i : Nat)
    (hi : stable.getLsbD i = true) : (contactStep central stable full).getLsbD i = true := by
  simp [contactStep, hi, -BitVec.getLsbD_eq_getElem]

theorem contactStep_decreases (central stable : Word) (full : FullLines)
    (h : contactStep central stable full ≠ stable) :
    (~~~(contactStep central stable full)).toNat < (~~~stable).toNat := by
  have hle : stable.toNat ≤ (contactStep central stable full).toNat := by
    apply Nat.le_of_testBit
    intro i hi
    exact contactStep_contains central stable full i hi
  have hne : (contactStep central stable full).toNat ≠ stable.toNat := by
    intro he; exact h (BitVec.eq_of_toNat_eq he)
  have hs := stable.isLt
  have hn := (contactStep central stable full).isLt
  simp only [BitVec.toNat_not]
  omega

def contactClosure (central stable : Word) (full : FullLines) : Word :=
  let next := contactStep central stable full
  if _h : next = stable then stable else contactClosure central next full
termination_by (~~~stable).toNat
decreasing_by exact contactStep_decreases central stable full _h

def getStableByContact (central previous : Word) (full : FullLines) : Word :=
  let seed := previous ||| (full.intersection &&& central)
  if seed = 0 then seed else contactClosure central seed full

-- On a full board with D4 as the only friendly disc, the edge seed is empty.
#guard getStableByContact (1 <<< 27) 0 (getFullLines (~~~0)) = 1 <<< 27

theorem contactClosure_preserves (central stable : Word) (full : FullLines)
    (P : Word → Prop) (step : ∀ x, P x → P (contactStep central x full))
    (initial : P stable) : P (contactClosure central stable full) := by
  induction stable using contactClosure.induct central full with
  | case1 x next h =>
      change contactStep central x full = x at h
      rw [contactClosure]
      simp only [h, ↓reduceDIte]
      exact initial
  | case2 x next h ih =>
      change contactStep central x full ≠ x at h
      rw [contactClosure]
      simp only [h, ↓reduceDIte]
      exact ih (step x initial)

theorem getStableByContact_preserves (central previous : Word) (full : FullLines)
    (P : Word → Prop) (initial : P (previous ||| (full.intersection &&& central)))
    (step : ∀ x, P x → P (contactStep central x full)) :
    P (getStableByContact central previous full) := by
  dsimp only [getStableByContact]
  split
  · exact initial
  · exact contactClosure_preserves central _ full P step initial

theorem contactClosure_fixed (central stable : Word) (full : FullLines) :
    contactStep central (contactClosure central stable full) full =
      contactClosure central stable full := by
  induction stable using contactClosure.induct central full with
  | case1 x next h =>
      change contactStep central x full = x at h
      have hx : contactClosure central x full = x := by
        rw [contactClosure]; simp only [h, ↓reduceDIte]
      rw [hx]; exact h
  | case2 x next h ih =>
      change contactStep central x full ≠ x at h
      have hx : contactClosure central x full = contactClosure central (contactStep central x full) full := by
        rw [contactClosure]; simp only [h, ↓reduceDIte]
      rw [hx]; exact ih

def stride (axis : Fin 4) : Nat :=
  match axis.val with | 0 => 1 | 1 => 8 | 2 => 7 | _ => 9

def ContactAt (stable : Word) (full : FullLines) (axis : Fin 4) (sq : Fin 64) : Prop :=
  ((stable >>> stride axis) ||| (stable <<< stride axis) ||| field full axis).getLsbD sq.val = true

theorem contactStep_new (central stable : Word) (full : FullLines) (sq : Fin 64)
    (h : (contactStep central stable full).getLsbD sq.val = true)
    (old : stable.getLsbD sq.val = false) :
    central.getLsbD sq.val = true ∧ ∀ axis : Fin 4, ContactAt stable full axis sq := by
  simp only [contactStep, BitVec.getLsbD_or, BitVec.getLsbD_and, old,
    Bool.false_or, Bool.and_eq_true] at h
  refine ⟨h.2, ?_⟩
  intro axis
  rcases axis with ⟨a, ha⟩
  match a with
  | 0 => simpa [ContactAt, stride, field, BitVec.getLsbD_or] using h.1.1.1.1
  | 1 => simpa [ContactAt, stride, field, BitVec.getLsbD_or] using h.1.1.1.2
  | 2 => simpa [ContactAt, stride, field, BitVec.getLsbD_or] using h.1.1.2
  | 3 => simpa [ContactAt, stride, field, BitVec.getLsbD_or] using h.1.2
  | a + 4 => omega

theorem contactAt_support (stable : Word) (full : FullLines) (axis : Fin 4) (sq : Fin 64)
    (h : ContactAt stable full axis sq) :
    (field full axis).getLsbD sq.val = true ∨
      ∃ j : Fin 64, (j.val = sq.val + stride axis ∨ j.val + stride axis = sq.val) ∧
        stable.getLsbD j.val = true := by
  simp only [ContactAt, BitVec.getLsbD_or, Bool.or_eq_true] at h
  rcases h with (h | h) | h
  · right
    have hb : stable.getLsbD (sq.val + stride axis) = true := by
      simpa only [BitVec.getLsbD_ushiftRight, Nat.add_comm] using h
    exact ⟨⟨sq.val + stride axis, BitVec.lt_of_getLsbD hb⟩, Or.inl rfl, hb⟩
  · right
    simp only [BitVec.getLsbD_shiftLeft, Bool.and_eq_true, decide_eq_true_eq] at h
    have hle : stride axis ≤ sq.val := by
      by_cases ht : sq.val < stride axis
      · simp [ht] at h
      · omega
    exact ⟨⟨sq.val - stride axis, BitVec.lt_of_getLsbD h.2⟩, Or.inr (by change sq.val - stride axis + stride axis = sq.val; omega), h.2⟩
  · exact Or.inl h

theorem center_interior (sq : Fin 64) (h : center.getLsbD sq.val = true) :
    0 < sq.val / 8 ∧ sq.val / 8 < 7 ∧ 0 < sq.val % 8 ∧ sq.val % 8 < 7 := by
  have all : ∀ s : Fin 64, center.getLsbD s.val = true →
      0 < s.val / 8 ∧ s.val / 8 < 7 ∧ 0 < s.val % 8 ∧ s.val % 8 < 7 := by decide
  exact all sq h


end NeuralReversi.Stability

module
public import NeuralReversi.Stability.Horizontal

@[expose] public section

namespace NeuralReversi.Stability

private abbrev Support := Nat → List Nat

private def Matches (w : Word) (disc : Word) (s : Support) : Prop :=
  ∀ i, i < 64 → w.getLsbD i = (s i).all disc.getLsbD

private theorem matches_self (disc : Word) : Matches disc disc (fun i => [i]) := by
  intro i _
  simp

private theorem matches_and {w₁ w₂ disc : Word} {s₁ s₂ : Support}
    (h₁ : Matches w₁ disc s₁) (h₂ : Matches w₂ disc s₂) :
    Matches (w₁ &&& w₂) disc (fun i => s₁ i ++ s₂ i) := by
  intro i hi
  rw [BitVec.getLsbD_and, h₁ i hi, h₂ i hi, List.all_append]

private theorem matches_shr_and {w disc : Word} {s : Support}
    (h : Matches w disc s) (k : Nat) :
    Matches (w &&& (w >>> k)) disc
      (fun i => if i + k < 64 then s i ++ s (i + k) else [64]) := by
  intro i hi
  rw [BitVec.getLsbD_and, BitVec.getLsbD_ushiftRight, h i hi]
  by_cases hlt : i + k < 64
  · simp only [hlt, ↓reduceIte, List.all_append, Nat.add_comm k i, h (i + k) hlt]
  · have hge : w.getLsbD (k + i) = false := BitVec.getLsbD_of_ge w _ (by omega)
    simp [hlt, hge]

private theorem matches_rotateRight {w disc : Word} {s : Support}
    (h : Matches w disc s) (k : Nat) (hk0 : 0 < k) (hk64 : k < 64) :
    Matches (rotateRight w k) disc (fun i => s ((i + k) % 64)) := by
  intro i hi
  unfold rotateRight
  rw [BitVec.getLsbD_or, BitVec.getLsbD_ushiftRight, BitVec.getLsbD_shiftLeft]
  by_cases hlt : k + i < 64
  · have hmod : (i + k) % 64 = k + i := by omega
    have hlt2 : i < 64 - k := by omega
    simp only [hi, decide_true, Bool.true_and, hlt2, Bool.not_true, Bool.false_and, Bool.or_false,
      hmod, h (k + i) hlt]
  · have hge : w.getLsbD (k + i) = false := BitVec.getLsbD_of_ge w _ (by omega)
    have hmod : (i + k) % 64 = i - (64 - k) := by omega
    have hnot : ¬ (i < 64 - k) := by omega
    simp only [hge, Bool.false_or, hi, decide_true, Bool.true_and, hnot, decide_false,
      Bool.not_false, hmod, h (i - (64 - k)) (by omega)]

private theorem matches_rotateLeft {w disc : Word} {s : Support}
    (h : Matches w disc s) (k : Nat) (hk0 : 0 < k) (hk64 : k < 64) :
    Matches (rotateLeft w k) disc (fun i => s ((i + (64 - k)) % 64)) := by
  intro i hi
  unfold rotateLeft
  rw [BitVec.getLsbD_or, BitVec.getLsbD_shiftLeft, BitVec.getLsbD_ushiftRight]
  by_cases hlt : i < k
  · have hmod : (i + (64 - k)) % 64 = 64 - k + i := by omega
    simp only [hi, decide_true, Bool.true_and, hlt, Bool.not_true, Bool.false_and, Bool.false_or,
      hmod, h (64 - k + i) (by omega)]
  · have hge : w.getLsbD (64 - k + i) = false := BitVec.getLsbD_of_ge w _ (by omega)
    have hmod : (i + (64 - k)) % 64 = i - k := by omega
    simp only [hi, decide_true, Bool.true_and, hlt, decide_false, Bool.not_false, hge,
      Bool.or_false, hmod, h (i - k) (by omega)]

private theorem matches_mask_shr {w disc : Word} {s : Support}
    (h : Matches w disc s) (mask : Word) (k : Nat)
    (hsafe : ∀ i, i < 64 → mask.getLsbD i = false → i + k < 64) :
    Matches (mask ||| (w >>> k)) disc (fun i => if mask.getLsbD i then [] else s (i + k)) := by
  intro i hi
  dsimp only
  rw [BitVec.getLsbD_or, BitVec.getLsbD_ushiftRight]
  cases hm : mask.getLsbD i
  · have hlt := hsafe i hi hm
    simp only [Bool.false_or, Bool.false_eq_true, ↓reduceIte, Nat.add_comm k i, h (i + k) hlt]
  · simp

private theorem matches_mask_shl {w disc : Word} {s : Support}
    (h : Matches w disc s) (mask : Word) (k : Nat)
    (hsafe : ∀ i, i < 64 → mask.getLsbD i = false → k ≤ i) :
    Matches (mask ||| (w <<< k)) disc (fun i => if mask.getLsbD i then [] else s (i - k)) := by
  intro i hi
  dsimp only
  rw [BitVec.getLsbD_or, BitVec.getLsbD_shiftLeft]
  cases hm : mask.getLsbD i
  · have hle := hsafe i hi hm
    have hnot : ¬ i < k := by omega
    simp only [Bool.false_or, Bool.false_eq_true, ↓reduceIte, hi, decide_true, Bool.true_and,
      hnot, decide_false, Bool.not_false, h (i - k) (by omega)]
  · simp

private theorem matches_mask_shr_shl {w₁ w₂ disc : Word} {s₁ s₂ : Support}
    (h₁ : Matches w₁ disc s₁) (h₂ : Matches w₂ disc s₂) (mask : Word) (k : Nat)
    (hsafe : ∀ i, i < 64 → mask.getLsbD i = false →
      (i + k < 64 ∧ i < k) ∨ (k ≤ i ∧ 64 ≤ i + k)) :
    Matches (mask ||| (w₁ >>> k) ||| (w₂ <<< k)) disc
      (fun i => if mask.getLsbD i then [] else if i + k < 64 then s₁ (i + k) else s₂ (i - k)) := by
  intro i hi
  dsimp only
  rw [BitVec.getLsbD_or, BitVec.getLsbD_or, BitVec.getLsbD_ushiftRight, BitVec.getLsbD_shiftLeft]
  cases hm : mask.getLsbD i
  · rcases hsafe i hi hm with ⟨hlt1, hlt2⟩ | ⟨hle, hge⟩
    · simp only [Bool.false_or, Bool.false_eq_true, ↓reduceIte, hlt1, hi, decide_true, Bool.true_and,
        hlt2, Bool.not_true, Bool.false_and, Bool.or_false, Nat.add_comm k i, h₁ (i + k) hlt1]
    · have h1 : w₁.getLsbD (k + i) = false := BitVec.getLsbD_of_ge w₁ _ (by omega)
      have hnot1 : ¬ i + k < 64 := by omega
      have hnot2 : ¬ i < k := by omega
      simp only [Bool.false_or, Bool.false_eq_true, ↓reduceIte, h1, hi, decide_true, Bool.true_and,
        hnot1, hnot2, decide_false, Bool.not_false, h₂ (i - k) (by omega)]
  · simp

private theorem word255_getLsbD (j : Nat) (hj : j < 64) :
    (255 : Word).getLsbD j = decide (j < 8) := by
  have hall : ∀ x : Fin 64, (255 : Word).getLsbD x.val = decide (x.val < 8) := by decide
  exact hall ⟨j, hj⟩

private theorem spread_getLsbD (p : Word) (i : Nat) (hi : i < 64) :
    (spread p [0,1,2,3,4,5,6,7]).getLsbD i = p.getLsbD (8 * (i / 8)) := by
  have hrow : ∀ (r : Nat),
      ((if p.getLsbD (8 * r) then ((255 : Word) <<< (8 * r)) else 0).getLsbD i) =
        (decide (i / 8 = r) && p.getLsbD (8 * r)) := by
    intro r
    cases hbit : p.getLsbD (8 * r)
    · simp
    · simp only [↓reduceIte, Bool.and_true, BitVec.getLsbD_shiftLeft, hi, decide_true, Bool.true_and]
      by_cases hlt : i < 8 * r
      · have hne : i / 8 ≠ r := by omega
        simp [hlt, hne]
      · have hsub64 : i - 8 * r < 64 := by omega
        have hlt8 : (i - 8 * r < 8) ↔ (i / 8 = r) := by omega
        simp only [hlt, decide_false, Bool.not_false, Bool.true_and, word255_getLsbD _ hsub64, hlt8]
  simp only [spread, BitVec.getLsbD_or, hrow]
  have hr : i / 8 = 0 ∨ i / 8 = 1 ∨ i / 8 = 2 ∨ i / 8 = 3 ∨
      i / 8 = 4 ∨ i / 8 = 5 ∨ i / 8 = 6 ∨ i / 8 = 7 := by omega
  rcases hr with hr | hr | hr | hr | hr | hr | hr | hr <;> simp [hr]

private def stepAnd (s₁ s₂ : Support) : Support := fun i => s₁ i ++ s₂ i
private def stepShrAnd (s : Support) (k : Nat) : Support :=
  fun i => if i + k < 64 then s i ++ s (i + k) else [64]
private def stepRotR (s : Support) (k : Nat) : Support := fun i => s ((i + k) % 64)
private def stepRotL (s : Support) (k : Nat) : Support := fun i => s ((i + (64 - k)) % 64)
private def stepMaskShr (s : Support) (mask : Word) (k : Nat) : Support :=
  fun i => if mask.getLsbD i then [] else s (i + k)
private def stepMaskShl (s : Support) (mask : Word) (k : Nat) : Support :=
  fun i => if mask.getLsbD i then [] else s (i - k)
private def stepMaskBoth (s₁ s₂ : Support) (mask : Word) (k : Nat) : Support :=
  fun i => if mask.getLsbD i then [] else if i + k < 64 then s₁ (i + k) else s₂ (i - k)

private def hSupport : Support :=
  let s0 : Support := fun i => [i]
  let s1 := stepShrAnd s0 1
  let s2 := stepShrAnd s1 2
  let s3 := stepShrAnd s2 4
  fun i => s3 (8 * (i / 8))

private def vSupport : Support :=
  let s0 : Support := fun i => [i]
  let s1 := stepAnd s0 (stepRotR s0 8)
  let s2 := stepAnd s1 (stepRotR s1 16)
  stepAnd s2 (stepRotL s2 32)

private def d7Support : Support :=
  let s0 : Support := fun i => [i]
  let l1 := stepAnd s0 (stepMaskShr s0 0xff01010101010101 7)
  let r1 := stepAnd s0 (stepMaskShl s0 0x80808080808080ff 7)
  let l2 := stepAnd l1 (stepMaskShr l1 0xffff030303030303 14)
  let r2 := stepAnd r1 (stepMaskShl r1 0xc0c0c0c0c0c0ffff 14)
  let l3 := stepAnd l2 (stepMaskShr l2 0xffffffff0f0f0f0f 28)
  let r3 := stepAnd r2 (stepMaskShl r2 0xf0f0f0f0ffffffff 28)
  stepAnd l3 r3

private def d9Support : Support :=
  let s0 : Support := fun i => [i]
  let l1 := stepAnd s0 (stepMaskShr s0 0xff80808080808080 9)
  let r1 := stepAnd s0 (stepMaskShl s0 0x01010101010101ff 9)
  let l2 := stepAnd l1 (stepMaskShr l1 0xffffc0c0c0c0c0c0 18)
  let r2 := stepAnd r1 (stepMaskShl r1 0x030303030303ffff 18)
  stepAnd (stepAnd l2 r2) (stepMaskBoth l2 r2 0x0f0f0f0ff0f0f0f0 36)

private def axisSupport (axis : Fin 4) : Support :=
  match axis.val with
  | 0 => hSupport
  | 1 => vSupport
  | 2 => d7Support
  | _ => d9Support

private theorem mask_bounds_ok :
    (∀ i : Fin 64, (0xff01010101010101 : Word).getLsbD i.val = false → i.val + 7 < 64) ∧
    (∀ i : Fin 64, (0x80808080808080ff : Word).getLsbD i.val = false → 7 ≤ i.val) ∧
    (∀ i : Fin 64, (0xffff030303030303 : Word).getLsbD i.val = false → i.val + 14 < 64) ∧
    (∀ i : Fin 64, (0xc0c0c0c0c0c0ffff : Word).getLsbD i.val = false → 14 ≤ i.val) ∧
    (∀ i : Fin 64, (0xffffffff0f0f0f0f : Word).getLsbD i.val = false → i.val + 28 < 64) ∧
    (∀ i : Fin 64, (0xf0f0f0f0ffffffff : Word).getLsbD i.val = false → 28 ≤ i.val) ∧
    (∀ i : Fin 64, (0xff80808080808080 : Word).getLsbD i.val = false → i.val + 9 < 64) ∧
    (∀ i : Fin 64, (0x01010101010101ff : Word).getLsbD i.val = false → 9 ≤ i.val) ∧
    (∀ i : Fin 64, (0xffffc0c0c0c0c0c0 : Word).getLsbD i.val = false → i.val + 18 < 64) ∧
    (∀ i : Fin 64, (0x030303030303ffff : Word).getLsbD i.val = false → 18 ≤ i.val) ∧
    (∀ i : Fin 64, (0x0f0f0f0ff0f0f0f0 : Word).getLsbD i.val = false →
      (i.val + 36 < 64 ∧ i.val < 36) ∨ (36 ≤ i.val ∧ 64 ≤ i.val + 36)) := by decide +kernel

private theorem field_matches (disc : Word) (axis : Fin 4) :
    Matches (field (getFullLines disc) axis) disc (axisSupport axis) := by
  have mb := mask_bounds_ok
  have h0 := matches_self disc
  rcases axis with ⟨a, ha⟩
  match a with
  | 0 =>
    intro i hi
    simp only [field, getFullLines, Id.run, pure]
    rw [dilate_eq_spread, spread_getLsbD _ i hi]
    have h1 := matches_shr_and h0 1
    have h2 := matches_shr_and h1 2
    have h3 := matches_shr_and h2 4
    exact h3 (8 * (i / 8)) (by omega)
  | 1 =>
    simp only [field, getFullLines, Id.run, pure, axisSupport, vSupport]
    have v1 := matches_and h0 (matches_rotateRight h0 8 (by decide) (by decide))
    have v2 := matches_and v1 (matches_rotateRight v1 16 (by decide) (by decide))
    exact matches_and v2 (matches_rotateLeft v2 32 (by decide) (by decide))
  | 2 =>
    simp only [field, getFullLines, Id.run, pure, axisSupport, d7Support]
    have l1 := matches_and h0 (matches_mask_shr h0 _ 7 (fun i hi h => mb.1 ⟨i, hi⟩ h))
    have r1 := matches_and h0 (matches_mask_shl h0 _ 7 (fun i hi h => mb.2.1 ⟨i, hi⟩ h))
    have l2 := matches_and l1 (matches_mask_shr l1 _ 14 (fun i hi h => mb.2.2.1 ⟨i, hi⟩ h))
    have r2 := matches_and r1 (matches_mask_shl r1 _ 14 (fun i hi h => mb.2.2.2.1 ⟨i, hi⟩ h))
    have l3 := matches_and l2 (matches_mask_shr l2 _ 28 (fun i hi h => mb.2.2.2.2.1 ⟨i, hi⟩ h))
    have r3 := matches_and r2 (matches_mask_shl r2 _ 28 (fun i hi h => mb.2.2.2.2.2.1 ⟨i, hi⟩ h))
    exact matches_and l3 r3
  | 3 =>
    simp only [field, getFullLines, Id.run, pure, axisSupport, d9Support]
    have l1 := matches_and h0 (matches_mask_shr h0 _ 9 (fun i hi h => mb.2.2.2.2.2.2.1 ⟨i, hi⟩ h))
    have r1 := matches_and h0 (matches_mask_shl h0 _ 9 (fun i hi h => mb.2.2.2.2.2.2.2.1 ⟨i, hi⟩ h))
    have l2 := matches_and l1 (matches_mask_shr l1 _ 18 (fun i hi h => mb.2.2.2.2.2.2.2.2.1 ⟨i, hi⟩ h))
    have r2 := matches_and r1 (matches_mask_shl r1 _ 18 (fun i hi h => mb.2.2.2.2.2.2.2.2.2.1 ⟨i, hi⟩ h))
    have both := matches_mask_shr_shl l2 r2 _ 36 (fun i hi h => mb.2.2.2.2.2.2.2.2.2.2 ⟨i, hi⟩ h)
    exact matches_and (matches_and l2 r2) both
  | _ + 4 => omega

private theorem support_equiv_lineCells : ∀ (axis : Fin 4) (sq : Fin 64),
    (axisSupport axis sq.val).all (fun t => decide (t ∈ lineCells axis sq.val)) = true ∧
    (lineCells axis sq.val).all (fun t => decide (t ∈ axisSupport axis sq.val)) = true := by
  decide +kernel

private theorem list_all_eq_of_subset {α : Type} (p : α → Bool) (xs ys : List α)
    (h1 : ∀ x ∈ xs, x ∈ ys) (h2 : ∀ y ∈ ys, y ∈ xs) :
    xs.all p = ys.all p := by
  apply Bool.eq_iff_iff.mpr
  simp only [List.all_eq_true]
  exact ⟨fun h y hy => h y (h2 y hy), fun h x hx => h x (h1 x hx)⟩

theorem fullLines_bits (disc : Word) (axis : Fin 4) (sq : Fin 64) :
    (field (getFullLines disc) axis).getLsbD sq.val = lineBits disc axis sq.val := by
  rw [field_matches disc axis sq.val sq.isLt]
  unfold lineBits
  have heq := support_equiv_lineCells axis sq
  apply list_all_eq_of_subset
  · intro x hx
    exact of_decide_eq_true (List.all_eq_true.mp heq.1 x hx)
  · intro y hy
    exact of_decide_eq_true (List.all_eq_true.mp heq.2 y hy)

theorem fullLines_correct (disc : Word) (axis : Fin 4) (sq : Fin 64) :
    (field (getFullLines disc) axis).getLsbD sq.val = true ↔ FullAt disc axis sq := by
  rw [fullLines_bits, lineBits_iff]

end NeuralReversi.Stability

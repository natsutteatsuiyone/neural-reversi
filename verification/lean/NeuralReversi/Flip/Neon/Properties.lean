module
public import NeuralReversi.Flip.Neon.Model
public import NeuralReversi.Flip.Properties
public import NeuralReversi.Flip.Avx512.Properties

@[expose] public section

namespace NeuralReversi.Flip.Neon
open NeuralReversi.Flip
open NeuralReversi.Flip.Avx512 (mask66 indexedValue indexedValue_typed)

set_option maxRecDepth 8192
set_option maxHeartbeats 2000000

theorem mask66_disjoint_02 : ∀ pos : Fin 66, mask66 pos 0 &&& mask66 pos 2 = 0#64 := by decide
theorem mask66_disjoint_13 : ∀ pos : Fin 66, mask66 pos 1 &&& mask66 pos 3 = 0#64 := by decide
theorem mask66_disjoint_02_13 :
    ∀ pos : Fin 66, (mask66 pos 0 ||| mask66 pos 2) &&& (mask66 pos 1 ||| mask66 pos 3) = 0#64 := by
  decide
theorem mask66_disjoint_46 : ∀ pos : Fin 66, mask66 pos 4 &&& mask66 pos 6 = 0#64 := by decide
theorem mask66_disjoint_57 : ∀ pos : Fin 66, mask66 pos 5 &&& mask66 pos 7 = 0#64 := by decide
theorem mask66_disjoint_46_57 :
    ∀ pos : Fin 66, (mask66 pos 4 ||| mask66 pos 6) &&& (mask66 pos 5 ||| mask66 pos 7) = 0#64 := by
  decide

theorem qsub_one_eq (x : Word) :
    qsubU64 x 1#64 = if x = 0#64 then 0#64 else x - 1#64 := by
  unfold qsubU64
  by_cases hx : x = 0#64
  · subst hx; decide
  · have hne : x.toNat ≠ 0 := fun h => hx (BitVec.eq_of_toNat_eq h)
    have hpos : ¬ x.toNat < (1#64).toNat := by
      have h1 : (1#64).toNat = 1 := rfl
      omega
    rw [ite_eq_right hpos, ite_eq_right hx]

theorem neg_and_player_eq_ls1b (mask player opponent : Word)
    (hdisj : player &&& opponent = 0#64) :
    -(mask &&& ~~~opponent) &&& (mask &&& player) =
      ((~~~opponent &&& mask) &&& -(~~~opponent &&& mask)) &&& player := by
  have hcomm : mask &&& ~~~opponent = ~~~opponent &&& mask := BitVec.and_comm _ _
  rw [hcomm]
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [BitVec.getLsbD_and, BitVec.getLsbD_not, hi, decide_true, Bool.true_and]
  cases hp : player.getLsbD i
  · simp
  · cases hm : mask.getLsbD i
    · simp
    · have ho : opponent.getLsbD i = false := by
        have h0 := congrArg (fun v : Word => v.getLsbD i) hdisj
        simp only [BitVec.getLsbD_and, BitVec.getLsbD_zero, hp, Bool.true_and] at h0
        exact h0
      simp [ho]

theorem ls1b_unique (x : Word) (i j : Nat)
    (hi : (x &&& -x).getLsbD i = true) (hj : (x &&& -x).getLsbD j = true) : i = j := by
  simp only [BitVec.getLsbD_and, BitVec.getLsbD_neg, Bool.and_eq_true] at hi hj
  have hilt : i < 64 := BitVec.lt_of_getLsbD hi.1
  have hjlt : j < 64 := BitVec.lt_of_getLsbD hj.1
  by_cases hlt : i < j
  · have hex : ∃ k, k < j ∧ x.getLsbD k = true := ⟨i, hlt, hi.1⟩
    have hcontra := hj.2
    simp only [hj.1, hjlt, hex, decide_true, Bool.and_self, Bool.xor_self] at hcontra
    exact Bool.noConfusion hcontra
  · by_cases hgt : j < i
    · have hex : ∃ k, k < i ∧ x.getLsbD k = true := ⟨j, hgt, hj.1⟩
      have hcontra := hi.2
      simp only [hi.1, hilt, hex, decide_true, Bool.and_self, Bool.xor_self] at hcontra
      exact Bool.noConfusion hcontra
    · omega

theorem ls1b_and_player_eq (x player : Word) (hne : (x &&& -x) &&& player ≠ 0#64) :
    (x &&& -x) &&& player = x &&& -x := by
  have hex : ∃ k < 64, ((x &&& -x) &&& player).getLsbD k = true := by
    by_cases h : ∃ k < 64, ((x &&& -x) &&& player).getLsbD k = true
    · exact h
    · exfalso
      apply hne
      apply BitVec.eq_of_getLsbD_eq
      intro i hi
      cases hb : ((x &&& -x) &&& player).getLsbD i
      · simp
      · exfalso; exact h ⟨i, hi, hb⟩
  obtain ⟨k, _, hk⟩ := hex
  have hk_and : (x &&& -x).getLsbD k = true ∧ player.getLsbD k = true := by
    simpa only [BitVec.getLsbD_and, Bool.and_eq_true] using hk
  apply BitVec.eq_of_getLsbD_eq
  intro i _
  simp only [BitVec.getLsbD_and]
  cases hy : (x &&& -x).getLsbD i
  · simp [BitVec.getLsbD_and] at hy ⊢
    intro h1 h2
    simp [h1, h2] at hy
  · have hik : i = k := ls1b_unique x i k hy hk_and.1
    subst hik
    simp [hk_and.2]

theorem span_lane_eq_flipLeft (mask player opponent : Word)
    (hdisj : player &&& opponent = 0#64) :
    mask &&& qsubU64 (-(mask &&& ~~~opponent) &&& (mask &&& player)) 1#64 =
      Scalar.flipLeft mask player (~~~opponent) := by
  rw [neg_and_player_eq_ls1b mask player opponent hdisj, qsub_one_eq]
  unfold Scalar.flipLeft
  dsimp only
  by_cases h : ((~~~opponent &&& mask) &&& -(~~~opponent &&& mask)) &&& player = 0#64
  · simp [h]
  · have hflank : (~~~opponent &&& mask) &&& -(~~~opponent &&& mask) ≠ 0#64 := by
      intro hz
      apply h
      rw [hz, BitVec.zero_and]
    rw [ls1b_and_player_eq (~~~opponent &&& mask) player h]
    rw [ite_eq_right hflank]
    have hflank0 : (~~~opponent &&& mask) &&& -(~~~opponent &&& mask) ≠ 0 := hflank
    rw [ite_eq_left hflank0]

theorem carry_inv_eq_neg (mask opponent : Word) :
    (opponent ||| ~~~mask) + 1#64 = -(mask &&& ~~~opponent) := by
  have hdm : opponent ||| ~~~mask = ~~~(mask &&& ~~~opponent) := by
    apply BitVec.eq_of_getLsbD_eq
    intro i hi
    simp only [BitVec.getLsbD_or, BitVec.getLsbD_not, BitVec.getLsbD_and, hi, decide_true,
      Bool.true_and, Bool.not_and, Bool.not_not, Bool.or_comm]
  rw [hdm, BitVec.neg_eq_not_add]

theorem span_inv_lane_eq_span_lane (mask player opponent : Word) :
    qsubU64 (((opponent ||| ~~~mask) + 1#64) &&& (player &&& ~~~(~~~mask))) 1#64 =
      qsubU64 (-(mask &&& ~~~opponent) &&& (mask &&& player)) 1#64 := by
  rw [carry_inv_eq_neg, BitVec.not_not, BitVec.and_comm player mask]

theorem merge_lane_eq_or (sha3 : Bool) (maskA wA maskB wB : Word)
    (hdisj : maskA &&& maskB = 0#64) :
    (if sha3 then (maskA &&& wA) ^^^ (wB &&& ~~~(~~~maskB))
     else (~~~maskB &&& (maskA &&& wA)) ||| (~~~(~~~maskB) &&& wB)) =
      (maskA &&& wA) ||| (maskB &&& wB) := by
  rw [BitVec.not_not]
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  have h0 := congrArg (fun v : Word => v.getLsbD i) hdisj
  simp only [BitVec.getLsbD_and, BitVec.getLsbD_zero] at h0
  cases sha3
  · simp only [Bool.false_eq_true, ↓reduceIte, BitVec.getLsbD_or, BitVec.getLsbD_and,
      BitVec.getLsbD_not, hi, decide_true, Bool.true_and]
    cases ha : maskA.getLsbD i <;> cases hb : maskB.getLsbD i <;> simp [ha, hb] at h0 ⊢
  · simp only [↓reduceIte, BitVec.getLsbD_xor, BitVec.getLsbD_or, BitVec.getLsbD_and]
    cases ha : maskA.getLsbD i <;> cases hb : maskB.getLsbD i <;> simp [ha, hb] at h0 ⊢

private theorem carry_false_of_disjoint (a b : Word) (h : a &&& b = 0#64) (i : Nat) :
    BitVec.carry i a b false = false := by
  induction i with
  | zero => simp [BitVec.carry, Nat.mod_one]
  | succ i ih =>
    rw [BitVec.carry_succ, ih]
    have h0 := congrArg (fun v : Word => v.getLsbD i) h
    simp only [BitVec.getLsbD_and, BitVec.getLsbD_zero] at h0
    simp [h0]

theorem add_eq_or_of_disjoint (a b : Word) (h : a &&& b = 0#64) : a + b = a ||| b := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  rw [BitVec.getLsbD_add hi, carry_false_of_disjoint a b h i, BitVec.getLsbD_or]
  have h0 := congrArg (fun v : Word => v.getLsbD i) h
  simp only [BitVec.getLsbD_and, BitVec.getLsbD_zero] at h0
  cases ha : a.getLsbD i <;> cases hb : b.getLsbD i <;> simp [ha, hb] at h0 ⊢

theorem getLsbD_reverse_eq (a : Word) (i : Nat) (hi : i < 64) :
    a.reverse.getLsbD i = a.getLsbD (63 - i) := by
  simp only [BitVec.getLsbD_reverse, BitVec.getMsbD, hi, decide_true, Bool.true_and]

theorem reverse_and (a b : Word) : (a &&& b).reverse = a.reverse &&& b.reverse := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [getLsbD_reverse_eq _ i hi, BitVec.getLsbD_and]

theorem reverse_or (a b : Word) : (a ||| b).reverse = a.reverse ||| b.reverse := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [getLsbD_reverse_eq _ i hi, BitVec.getLsbD_or]

theorem reverse_not (a : Word) : (~~~a).reverse = ~~~a.reverse := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  have hlt : 63 - i < 64 := by omega
  simp only [getLsbD_reverse_eq _ i hi, BitVec.getLsbD_not, hi, hlt, decide_true, Bool.true_and]

theorem reverse_zero : (0#64).reverse = 0#64 := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [getLsbD_reverse_eq _ i hi, BitVec.getLsbD_zero]

theorem flipLeft_reverse_eq_flipRight (mask player notOpponent : Word) :
    (Scalar.flipLeft mask.reverse player.reverse notOpponent.reverse).reverse =
      Scalar.flipRightValue mask player notOpponent := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  apply Bool.eq_iff_iff.mpr
  rw [getLsbD_reverse_eq _ i hi, Scalar.left_contract, Scalar.right_contract]
  constructor
  · rintro ⟨j', ⟨hj'1, hj'2⟩, hpj', hm', hij'⟩
    have hj'lt : j' < 64 := BitVec.lt_of_getLsbD hj'1
    refine ⟨63 - j', ⟨?_, ?_⟩, ?_, ?_, ?_⟩
    · have h1 := hj'1
      rw [← reverse_and, getLsbD_reverse_eq _ j' hj'lt] at h1
      exact h1
    · intro k hk
      by_cases hklt : k < 64
      · have h2 := hj'2 (63 - k) (by omega)
        rw [← reverse_and, getLsbD_reverse_eq _ (63 - k) (by omega)] at h2
        have heq : 63 - (63 - k) = k := by omega
        rwa [heq] at h2
      · exact BitVec.getLsbD_of_ge _ k (by omega)
    · rw [getLsbD_reverse_eq _ j' hj'lt] at hpj'
      exact hpj'
    · rw [getLsbD_reverse_eq _ (63 - i) (by omega)] at hm'
      have heq : 63 - (63 - i) = i := by omega
      rwa [heq] at hm'
    · omega
  · rintro ⟨j, ⟨hj1, hj2⟩, hpj, hm, hji⟩
    have hjlt : j < 64 := BitVec.lt_of_getLsbD hj1
    refine ⟨63 - j, ⟨?_, ?_⟩, ?_, ?_, ?_⟩
    · rw [← reverse_and, getLsbD_reverse_eq _ (63 - j) (by omega)]
      have heq : 63 - (63 - j) = j := by omega
      rwa [heq]
    · intro k hk
      have hklt : k < 64 := by omega
      rw [← reverse_and, getLsbD_reverse_eq _ k hklt]
      exact hj2 (63 - k) (by omega)
    · rw [getLsbD_reverse_eq _ (63 - j) (by omega)]
      have heq : 63 - (63 - j) = j := by omega
      rwa [heq]
    · rw [getLsbD_reverse_eq _ (63 - i) (by omega)]
      have heq : 63 - (63 - i) = i := by omega
      rwa [heq]
    · omega

theorem subset_disjoint {a b m1 m2 : Word} (hd : m1 &&& m2 = 0#64)
    (ha : a &&& ~~~m1 = 0#64) (hb : b &&& ~~~m2 = 0#64) : a &&& b = 0#64 := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  have h1 := congrArg (fun v : Word => v.getLsbD i) hd
  have h2 := congrArg (fun v : Word => v.getLsbD i) ha
  have h3 := congrArg (fun v : Word => v.getLsbD i) hb
  simp only [BitVec.getLsbD_and, BitVec.getLsbD_not, BitVec.getLsbD_zero, hi, decide_true,
    Bool.true_and] at h1 h2 h3 ⊢
  cases hm1 : m1.getLsbD i <;> cases hm2 : m2.getLsbD i <;>
    cases ha' : a.getLsbD i <;> cases hb' : b.getLsbD i <;>
    simp [hm1, hm2, ha', hb'] at h1 h2 h3 ⊢

theorem flipLeft_subset_mask (mask player notOpponent : Word) :
    Scalar.flipLeft mask player notOpponent &&& ~~~mask = 0#64 := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [BitVec.getLsbD_and, BitVec.getLsbD_not, BitVec.getLsbD_zero, hi, decide_true,
    Bool.true_and]
  cases hm : mask.getLsbD i
  · cases hf : (Scalar.flipLeft mask player notOpponent).getLsbD i
    · rfl
    · obtain ⟨_, _, _, hm', _⟩ := (Scalar.left_contract mask player notOpponent i).mp hf
      simp [hm] at hm'
  · simp

theorem or_subset_union {a b m1 m2 : Word}
    (ha : a &&& ~~~m1 = 0#64) (hb : b &&& ~~~m2 = 0#64) :
    (a ||| b) &&& ~~~(m1 ||| m2) = 0#64 := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  have h2 := congrArg (fun v : Word => v.getLsbD i) ha
  have h3 := congrArg (fun v : Word => v.getLsbD i) hb
  simp only [BitVec.getLsbD_and, BitVec.getLsbD_or, BitVec.getLsbD_not, BitVec.getLsbD_zero,
    hi, decide_true, Bool.true_and] at h2 h3 ⊢
  cases hm1 : m1.getLsbD i <;> cases hm2 : m2.getLsbD i <;>
    cases ha' : a.getLsbD i <;> cases hb' : b.getLsbD i <;>
    simp [hm1, hm2, ha', hb'] at h2 h3 ⊢

private theorem merge_side_eq (sha3 : Bool) (m0 m1 m2 m3 player opponent : Word)
    (hdisj : player &&& opponent = 0#64)
    (hd02 : m0 &&& m2 = 0#64) (hd13 : m1 &&& m3 = 0#64) :
    mergeSpans sha3 (m0, m1)
        (flipSpanPair (m0, m1) (vdupqNU64 player) (vdupqNU64 opponent) (vdupqNU64 1#64))
        (~~~m2, ~~~m3)
        (flipSpanPairInv (~~~m2, ~~~m3) (vdupqNU64 player) (vdupqNU64 opponent) (vdupqNU64 1#64)) =
      (Scalar.flipLeft m0 player (~~~opponent) ||| Scalar.flipLeft m2 player (~~~opponent),
       Scalar.flipLeft m1 player (~~~opponent) ||| Scalar.flipLeft m3 player (~~~opponent)) := by
  unfold mergeSpans vbcaxqU64 vbslqU64 flipSpanPair flipSpanPairInv
  unfold vandqU64 vorrqU64 vbicqU64 vaddqU64 negU64 vqsubqU64 vdupqNU64
  cases sha3 <;> simp only [Bool.false_eq_true, ↓reduceIte]
  · apply Prod.ext <;> dsimp only
    · have h := merge_lane_eq_or false m0
        (qsubU64 (-(m0 &&& ~~~opponent) &&& (m0 &&& player)) 1#64) m2
        (qsubU64 (((opponent ||| ~~~m2) + 1#64) &&& (player &&& ~~~(~~~m2))) 1#64) hd02
      simp only [Bool.false_eq_true, ↓reduceIte] at h
      rw [h, span_inv_lane_eq_span_lane, span_lane_eq_flipLeft _ _ _ hdisj,
        span_lane_eq_flipLeft _ _ _ hdisj]
    · have h := merge_lane_eq_or false m1
        (qsubU64 (-(m1 &&& ~~~opponent) &&& (m1 &&& player)) 1#64) m3
        (qsubU64 (((opponent ||| ~~~m3) + 1#64) &&& (player &&& ~~~(~~~m3))) 1#64) hd13
      simp only [Bool.false_eq_true, ↓reduceIte] at h
      rw [h, span_inv_lane_eq_span_lane, span_lane_eq_flipLeft _ _ _ hdisj,
        span_lane_eq_flipLeft _ _ _ hdisj]
  · apply Prod.ext <;> dsimp only
    · have h := merge_lane_eq_or true m0
        (qsubU64 (-(m0 &&& ~~~opponent) &&& (m0 &&& player)) 1#64) m2
        (qsubU64 (((opponent ||| ~~~m2) + 1#64) &&& (player &&& ~~~(~~~m2))) 1#64) hd02
      simp only [↓reduceIte] at h
      rw [h, span_inv_lane_eq_span_lane, span_lane_eq_flipLeft _ _ _ hdisj,
        span_lane_eq_flipLeft _ _ _ hdisj]
    · have h := merge_lane_eq_or true m1
        (qsubU64 (-(m1 &&& ~~~opponent) &&& (m1 &&& player)) 1#64) m3
        (qsubU64 (((opponent ||| ~~~m3) + 1#64) &&& (player &&& ~~~(~~~m3))) 1#64) hd13
      simp only [↓reduceIte] at h
      rw [h, span_inv_lane_eq_span_lane, span_lane_eq_flipLeft _ _ _ hdisj,
        span_lane_eq_flipLeft _ _ _ hdisj]

theorem flipIndex_eq_indexedValue (sha3 : Bool) (pos : Fin 66) (player opponent : Word)
    (hdisj : player &&& opponent = 0#64) :
    flipIndex sha3 pos player opponent = indexedValue pos player opponent := by
  have hdisjRev : player.reverse &&& opponent.reverse = 0#64 := by
    rw [← reverse_and, hdisj, reverse_zero]
  have hd46r : (mask66 pos 4).reverse &&& (mask66 pos 6).reverse = 0#64 := by
    rw [← reverse_and, mask66_disjoint_46 pos, reverse_zero]
  have hd57r : (mask66 pos 5).reverse &&& (mask66 pos 7).reverse = 0#64 := by
    rw [← reverse_and, mask66_disjoint_57 pos, reverse_zero]
  unfold flipIndex BoardCtx.flip1 BoardCtx.flipPairs BoardCtx.new
  dsimp only
  have hm0 : neonMask pos 0 = mask66 pos 0 := rfl
  have hm1 : neonMask pos 1 = mask66 pos 1 := rfl
  have hm2 : neonMask pos 2 = ~~~mask66 pos 2 := rfl
  have hm3 : neonMask pos 3 = ~~~mask66 pos 3 := rfl
  have hm4 : neonMask pos 4 = (mask66 pos 4).reverse := rfl
  have hm5 : neonMask pos 5 = (mask66 pos 5).reverse := rfl
  have hm6 : neonMask pos 6 = ~~~(mask66 pos 6).reverse := rfl
  have hm7 : neonMask pos 7 = ~~~(mask66 pos 7).reverse := rfl
  rw [hm0, hm1, hm2, hm3, hm4, hm5, hm6, hm7]
  rw [merge_side_eq sha3 (mask66 pos 0) (mask66 pos 1) (mask66 pos 2) (mask66 pos 3)
        player opponent hdisj (mask66_disjoint_02 pos) (mask66_disjoint_13 pos)]
  rw [merge_side_eq sha3 (mask66 pos 4).reverse (mask66 pos 5).reverse
        (mask66 pos 6).reverse (mask66 pos 7).reverse
        player.reverse opponent.reverse hdisjRev hd46r hd57r]
  unfold foldAddp vpaddqU64
  dsimp only
  have hdl : (Scalar.flipLeft (mask66 pos 0) player (~~~opponent) |||
        Scalar.flipLeft (mask66 pos 2) player (~~~opponent)) &&&
      (Scalar.flipLeft (mask66 pos 1) player (~~~opponent) |||
        Scalar.flipLeft (mask66 pos 3) player (~~~opponent)) = 0#64 :=
    subset_disjoint (mask66_disjoint_02_13 pos)
      (or_subset_union (flipLeft_subset_mask _ _ _) (flipLeft_subset_mask _ _ _))
      (or_subset_union (flipLeft_subset_mask _ _ _) (flipLeft_subset_mask _ _ _))
  have hdr_mask : ((mask66 pos 4).reverse ||| (mask66 pos 6).reverse) &&&
      ((mask66 pos 5).reverse ||| (mask66 pos 7).reverse) = 0#64 := by
    rw [← reverse_or, ← reverse_or, ← reverse_and, mask66_disjoint_46_57 pos, reverse_zero]
  have hdr : (Scalar.flipLeft (mask66 pos 4).reverse player.reverse (~~~opponent.reverse) |||
        Scalar.flipLeft (mask66 pos 6).reverse player.reverse (~~~opponent.reverse)) &&&
      (Scalar.flipLeft (mask66 pos 5).reverse player.reverse (~~~opponent.reverse) |||
        Scalar.flipLeft (mask66 pos 7).reverse player.reverse (~~~opponent.reverse)) = 0#64 :=
    subset_disjoint hdr_mask
      (or_subset_union (flipLeft_subset_mask _ _ _) (flipLeft_subset_mask _ _ _))
      (or_subset_union (flipLeft_subset_mask _ _ _) (flipLeft_subset_mask _ _ _))
  rw [add_eq_or_of_disjoint _ _ hdl, add_eq_or_of_disjoint _ _ hdr]
  rw [reverse_or, reverse_or, reverse_or]
  rw [← reverse_not opponent]
  rw [flipLeft_reverse_eq_flipRight, flipLeft_reverse_eq_flipRight,
    flipLeft_reverse_eq_flipRight, flipLeft_reverse_eq_flipRight]
  unfold indexedValue
  ac_rfl

theorem flip_eq_scalarValue (sha3 : Bool) (sq : Square) (player opponent : Word)
    (hdisj : player &&& opponent = 0#64) :
    flip sha3 sq player opponent = value sq player opponent := by
  unfold flip
  rw [flipIndex_eq_indexedValue sha3 _ player opponent hdisj]
  exact indexedValue_typed ⟨sq.val, by have := sq.isLt; omega⟩ sq.isLt player opponent

theorem flip_coordinate (sha3 : Bool) (sq : Square) (player opponent : Word)
    (hdisj : player &&& opponent = 0#64) (i : Nat) :
    (flip sha3 sq player opponent).getLsbD i = true ↔ CoordinateBit sq player opponent i := by
  rw [flip_eq_scalarValue sha3 sq player opponent hdisj]
  exact value_coordinate sq player opponent i

theorem flip2_eq (sha3 : Bool) (x0 x1 : Fin 66) (player opponent : Word)
    (hdisj : player &&& opponent = 0#64) :
    (BoardCtx.new player opponent).flip2 sha3 x0 x1 =
      (indexedValue x0 player opponent, indexedValue x1 player opponent) := by
  unfold BoardCtx.flip2
  exact Prod.ext (flipIndex_eq_indexedValue sha3 x0 player opponent hdisj)
    (flipIndex_eq_indexedValue sha3 x1 player opponent hdisj)

theorem flip3_eq (sha3 : Bool) (x0 x1 x2 : Fin 66) (player opponent : Word)
    (hdisj : player &&& opponent = 0#64) :
    (BoardCtx.new player opponent).flip3 sha3 x0 x1 x2 =
      (indexedValue x0 player opponent, indexedValue x1 player opponent,
        indexedValue x2 player opponent) := by
  unfold BoardCtx.flip3
  exact Prod.ext (flipIndex_eq_indexedValue sha3 x0 player opponent hdisj)
    (Prod.ext (flipIndex_eq_indexedValue sha3 x1 player opponent hdisj)
      (flipIndex_eq_indexedValue sha3 x2 player opponent hdisj))

theorem flip4_eq (sha3 : Bool) (x0 x1 x2 x3 : Fin 66) (player opponent : Word)
    (hdisj : player &&& opponent = 0#64) :
    (BoardCtx.new player opponent).flip4 sha3 x0 x1 x2 x3 =
      (indexedValue x0 player opponent, indexedValue x1 player opponent,
        indexedValue x2 player opponent, indexedValue x3 player opponent) := by
  unfold BoardCtx.flip4
  exact Prod.ext (flipIndex_eq_indexedValue sha3 x0 player opponent hdisj)
    (Prod.ext (flipIndex_eq_indexedValue sha3 x1 player opponent hdisj)
      (Prod.ext (flipIndex_eq_indexedValue sha3 x2 player opponent hdisj)
        (flipIndex_eq_indexedValue sha3 x3 player opponent hdisj)))

def NeonFlipContract : Prop :=
  ∀ (sha3 : Bool) (sq : Square) (player opponent : Word),
    player &&& opponent = 0#64 →
    flip sha3 sq player opponent = value sq player opponent ∧
    ∀ i, (flip sha3 sq player opponent).getLsbD i = true ↔ CoordinateBit sq player opponent i

theorem neonFlipContract_proven : NeonFlipContract :=
  fun sha3 sq player opponent hdisj =>
    ⟨flip_eq_scalarValue sha3 sq player opponent hdisj,
     flip_coordinate sha3 sq player opponent hdisj⟩

end NeuralReversi.Flip.Neon

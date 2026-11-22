import NeuralReversi.Flip.ScalarSpec
import NeuralReversi.Bitboard.LeastBit

namespace NeuralReversi.Flip.Scalar

def one (j : Nat) : Word := 1#64 <<< j

theorem one_bit (j : Nat) (hj : j < 64) (i : Nat) :
    (one j).getLsbD i = decide (i = j) := by
  simp only [one, BitVec.getLsbD_shiftLeft, BitVec.getLsbD_one]
  simp only [show 0 < 64 by decide, decide_true, Bool.true_and]
  apply Bool.eq_iff_iff.mpr
  simp only [Bool.and_eq_true, decide_eq_true_eq, Bool.not_eq_true', decide_eq_false_iff_not]
  omega

theorem one_sub_bit (j : Nat) (hj : j < 64) (i : Nat) :
    (one j - 1#64).getLsbD i = decide (i < j) := by
  rw [← BitVec.not_neg]
  simp only [BitVec.getLsbD_not, BitVec.getLsbD_neg, one_bit j hj]
  have hex : (∃ k < i, decide (k = j) = true) ↔ j < i := by simp
  simp only [hex]
  rcases Nat.lt_trichotomy i j with h | h | h
  · have hw : i < 64 := by omega
    have he : ¬ i = j := by omega
    have hr : ¬ j < i := by omega
    simp [hw, he, hr, h]
  · subst i
    simp [hj]
  · have he : ¬ i = j := by omega
    have hr : ¬ i < j := by omega
    by_cases hw : i < 64 <;> simp [hw, he, hr, h]

theorem one_upper_bit (j : Nat) (hj : j < 64) (i : Nat) :
    (~~~(one j ||| (one j - 1#64))).getLsbD i = decide (j < i ∧ i < 64) := by
  simp only [BitVec.getLsbD_not, BitVec.getLsbD_or, one_bit j hj, one_sub_bit j hj]
  apply Bool.eq_iff_iff.mpr
  simp only [Bool.and_eq_true, Bool.not_eq_true', Bool.or_eq_false_iff,
    decide_eq_false_iff_not, decide_eq_true_eq]
  omega

theorem one_anchor (j : Nat) (hj : j < 64) (player : Word) :
    one j &&& player ≠ 0 ↔ player.getLsbD j = true := by
  constructor
  · intro hn
    cases hb : player.getLsbD j with
    | true => rfl
    | false =>
      exfalso
      apply hn
      apply BitVec.eq_of_getLsbD_eq
      intro i _
      simp only [BitVec.getLsbD_and, one_bit j hj]
      by_cases hi : i = j <;> simp_all
  · intro hp hz
    have hb := congrArg (fun x : Word => x.getLsbD j) hz
    simp [BitVec.getLsbD_and, one_bit j hj, hp] at hb

theorem isolate_bit (x : Word) (i : Nat) :
    (x &&& -x).getLsbD i = true ↔ Least x i := by
  simp only [BitVec.getLsbD_and, BitVec.getLsbD_neg, Least]
  by_cases hx : x.getLsbD i = true
  · have hi := BitVec.lt_of_getLsbD hx
    simp only [hx, hi, decide_true, Bool.true_and, Bool.true_xor,
      Bool.not_eq_true', decide_eq_false_iff_not]
    simp only [not_exists, not_and, Bool.not_eq_true]
    simp only [true_and]
  · simp [Bool.eq_false_iff.mpr hx]

theorem least_unique (x : Word) {j k : Nat} (hj : Least x j) (hk : Least x k) :
    j = k := by
  rcases hj with ⟨hj, hlj⟩
  rcases hk with ⟨hk, hlk⟩
  by_cases hjk : j < k
  · have := hlk j hjk
    simp_all
  · by_cases hkj : k < j
    · have := hlj k hkj
      simp_all
    · omega

theorem isolate_eq_one (x : Word) (j : Nat) (hj : Least x j) :
    x &&& -x = one j := by
  have hjw := BitVec.lt_of_getLsbD hj.1
  apply BitVec.eq_of_getLsbD_eq
  intro i _
  rw [one_bit j hjw]
  apply Bool.eq_iff_iff.mpr
  rw [isolate_bit]
  simp only [decide_eq_true_eq]
  constructor
  · exact fun hi => least_unique x hi hj
  · exact fun hi => hi ▸ hj

theorem least_ctz (x : Word) (hx : x ≠ 0) : Least x x.ctz.toNat := by
  exact ⟨BitVec.getLsbD_true_ctz_of_ne_zero hx,
    fun _ hk => BitVec.getLsbD_false_of_lt_ctz hk⟩

theorem greatest_clz (x : Word) (hx : x ≠ 0) : Greatest x (63 - x.clz.toNat) := by
  have hc : x.clz.toNat < 64 := by
    simpa [BitVec.lt_def] using (BitVec.clz_lt_iff_ne_zero.mpr hx)
  refine ⟨BitVec.getLsbD_true_clz_of_ne_zero (by decide) hx, ?_⟩
  intro k hk
  have hb := BitVec.toNat_lt_two_pow_sub_clz (x := x)
  have hp : 2 ^ (64 - x.clz.toNat) ≤ 2 ^ k := Nat.pow_le_pow_of_le (by decide) (by omega)
  exact Nat.testBit_lt_two_pow (by omega)

theorem greatest_unique (x : Word) {j k : Nat} (hj : Greatest x j) (hk : Greatest x k) :
    j = k := by
  rcases hj with ⟨hj, hlj⟩
  rcases hk with ⟨hk, hlk⟩
  by_cases hjk : j < k
  · have := hlj k hjk
    simp_all
  · by_cases hkj : k < j
    · have := hlk j hkj
      simp_all
    · omega

theorem count_arithmetic (x : Word) (hx : x ≠ 0) :
    (1#32).toNat ≤ (64#32).toNat ∧
    (leadingZeros32 x).toNat ≤ (64#32 - 1#32).toNat ∧
    (shiftCount32 x).toNat = 63 - x.clz.toNat ∧
    (shiftCount32 x).toNat < 64 := by
  have hc : x.clz.toNat < 64 := by
    simpa [BitVec.lt_def] using (BitVec.clz_lt_iff_ne_zero.mpr hx)
  have hsmall : x.clz.toNat < 2 ^ 32 := by omega
  have hl : (leadingZeros32 x).toNat = x.clz.toNat := by
    simp [leadingZeros32, Nat.mod_eq_of_lt hsmall]
  have hsub : 64#32 - 1#32 = 63#32 := by decide
  have hs : (shiftCount32 x).toNat = 63 - x.clz.toNat := by
    unfold shiftCount32
    rw [hsub, BitVec.toNat_sub_of_le]
    · simp [hl]
    · simp only [BitVec.le_def, BitVec.toNat_ofNat, hl]
      omega
  refine ⟨by decide, ?_, hs, ?_⟩
  · rw [hsub, hl]
    simp only [BitVec.toNat_ofNat]
    omega
  · rw [hs]
    omega

def flipRightValue (mask player notOpponent : Word) : Word :=
  let x := notOpponent &&& mask
  if x = 0 then 0
  else
    let flank := 1#64 <<< shiftCount32 x
    if flank &&& player ≠ 0 then mask &&& ~~~(flank ||| (flank - 1#64)) else 0

theorem least_exists_iff (x : Word) (j : Nat) (hj : Least x j) (P : Nat → Prop) :
    (∃ k, Least x k ∧ P k) ↔ P j := by
  constructor
  · rintro ⟨k, hk, hp⟩
    exact least_unique x hk hj ▸ hp
  · exact fun hp => ⟨j, hj, hp⟩

theorem greatest_exists_iff (x : Word) (j : Nat) (hj : Greatest x j) (P : Nat → Prop) :
    (∃ k, Greatest x k ∧ P k) ↔ P j := by
  constructor
  · rintro ⟨k, hk, hp⟩
    exact greatest_unique x hk hj ▸ hp
  · exact fun hp => ⟨j, hj, hp⟩

theorem left_contract (mask player notOpponent : Word) (i : Nat) :
    (flipLeft mask player notOpponent).getLsbD i = true ↔
      ∃ j, Least (notOpponent &&& mask) j ∧
        player.getLsbD j = true ∧ mask.getLsbD i = true ∧ i < j := by
  by_cases hx : notOpponent &&& mask = 0
  · simp [flipLeft, hx, Least]
  · have hj := least_ctz (notOpponent &&& mask) hx
    have hjw := BitVec.lt_of_getLsbD hj.1
    rw [least_exists_iff _ _ hj]
    simp only [flipLeft, isolate_eq_one _ _ hj, one_anchor _ hjw]
    by_cases hp : player.getLsbD (notOpponent &&& mask).ctz.toNat = true
    · simp only [hp, ite_true, BitVec.getLsbD_and, one_sub_bit _ hjw,
        Bool.and_eq_true, decide_eq_true_eq, true_and]
    · simp [hp]

theorem right_contract (mask player notOpponent : Word) (i : Nat) :
    (flipRightValue mask player notOpponent).getLsbD i = true ↔
      ∃ j, Greatest (notOpponent &&& mask) j ∧
        player.getLsbD j = true ∧ mask.getLsbD i = true ∧ j < i := by
  by_cases hx : notOpponent &&& mask = 0
  · simp [flipRightValue, hx, Greatest]
  · have hj := greatest_clz (notOpponent &&& mask) hx
    have hjw := BitVec.lt_of_getLsbD hj.1
    rw [greatest_exists_iff _ _ hj]
    have hc := (count_arithmetic (notOpponent &&& mask) hx).2.2.1
    have hf : 1#64 <<< shiftCount32 (notOpponent &&& mask) =
        one (63 - (notOpponent &&& mask).clz.toNat) := by
      rw [BitVec.shiftLeft_eq', hc]
      rfl
    simp only [flipRightValue, hx, ite_false, hf, one_anchor _ hjw]
    by_cases hp : player.getLsbD (63 - (notOpponent &&& mask).clz.toNat) = true
    · simp only [hp, ite_true, BitVec.getLsbD_and, one_upper_bit _ hjw,
        Bool.and_eq_true, decide_eq_true_eq, true_and]
      constructor
      · exact fun h => ⟨h.1, h.2.1⟩
      · exact fun h => ⟨h.1, h.2, BitVec.lt_of_getLsbD h.1⟩
    · simp [hp]

theorem left_zero_input (mask player notOpponent : Word)
    (hx : notOpponent &&& mask = 0) : flipLeft mask player notOpponent = 0 := by
  simp [flipLeft, hx]

theorem right_zero_input (mask player notOpponent : Word)
    (hx : notOpponent &&& mask = 0) : flipRightValue mask player notOpponent = 0 := by
  simp [flipRightValue, hx]

theorem left_no_anchor (mask player notOpponent : Word)
    (hp : ∀ j, Least (notOpponent &&& mask) j → player.getLsbD j = false) :
    flipLeft mask player notOpponent = 0 := by
  apply BitVec.eq_of_getLsbD_eq
  intro i _
  have hn : ¬ (flipLeft mask player notOpponent).getLsbD i = true := by
    intro h
    obtain ⟨j, hj, hpj, _⟩ := (left_contract mask player notOpponent i).mp h
    have := hp j hj
    simp_all
  simpa using hn

theorem right_no_anchor (mask player notOpponent : Word)
    (hp : ∀ j, Greatest (notOpponent &&& mask) j → player.getLsbD j = false) :
    flipRightValue mask player notOpponent = 0 := by
  apply BitVec.eq_of_getLsbD_eq
  intro i _
  have hn : ¬ (flipRightValue mask player notOpponent).getLsbD i = true := by
    intro h
    obtain ⟨j, hj, hpj, _⟩ := (right_contract mask player notOpponent i).mp h
    have := hp j hj
    simp_all
  simpa using hn



theorem flipRight_eq_value (checks : Bool) (mask player notOpponent : Word) :
    flipRight checks mask player notOpponent = .ok (flipRightValue mask player notOpponent) := by
  by_cases hx : notOpponent &&& mask = 0
  · simp [flipRight,flipRightValue,hx]
  · have safe := count_arithmetic (notOpponent &&& mask) hx
    have first : checkedSub32 checks 64#32 1#32 = .ok (64#32-1#32) := by
      simp only [checkedSub32,Nat.not_lt_of_ge safe.1,and_false,↓reduceIte]
    have second : checkedSub32 checks (64#32-1#32) (leadingZeros32 (notOpponent &&& mask)) =
        .ok (shiftCount32 (notOpponent &&& mask)) := by
      simp only [checkedSub32,Nat.not_lt_of_ge safe.2.1,and_false,↓reduceIte,shiftCount32]
    have shift : rustShl64 checks 1#64 (shiftCount32 (notOpponent &&& mask)) =
        .ok (1#64 <<< shiftCount32 (notOpponent &&& mask)) := by
      simp only [rustShl64,Nat.not_le_of_lt safe.2.2.2,and_false,↓reduceIte,
        Nat.mod_eq_of_lt safe.2.2.2,BitVec.shiftLeft_eq']
    rw [flipRight,ite_eq_right hx,flipRightValue,ite_eq_right hx,first]
    change (do
      let count ← checkedSub32 checks (64#32-1#32) (leadingZeros32 (notOpponent &&& mask))
      let flank ← rustShl64 checks 1#64 count
      pure (if flank &&& player ≠ 0 then mask &&& ~~~(flank ||| (flank-1#64)) else 0)) = _
    rw [second]
    change (do
      let flank ← rustShl64 checks 1#64 (shiftCount32 (notOpponent &&& mask))
      pure (if flank &&& player ≠ 0 then mask &&& ~~~(flank ||| (flank-1#64)) else 0)) = _
    rw [shift]
    rfl


def LeftContract : Prop :=
  ∀ mask player notOpponent i,
    (flipLeft mask player notOpponent).getLsbD i = true ↔ LeftBitSpec mask player notOpponent i

theorem leftContract_proven : LeftContract := by
  intro mask player notOpponent i
  exact left_contract mask player notOpponent i

def RightContract : Prop :=
  (∀ checks mask player notOpponent, ∃ value,
    flipRight checks mask player notOpponent = .ok value ∧
    ∀ i, value.getLsbD i = true ↔ RightBitSpec mask player notOpponent i) ∧
  (∀ x : Word, x ≠ 0 →
    (1#32).toNat ≤ (64#32).toNat ∧
    (leadingZeros32 x).toNat ≤ (64#32-1#32).toNat ∧
    (shiftCount32 x).toNat = 63-x.clz.toNat ∧ (shiftCount32 x).toNat < 64)

theorem rightContract_proven : RightContract := by
  refine ⟨?_,count_arithmetic⟩
  intro checks mask player notOpponent
  exact ⟨flipRightValue mask player notOpponent,flipRight_eq_value checks mask player notOpponent,
    fun i => right_contract mask player notOpponent i⟩

end NeuralReversi.Flip.Scalar

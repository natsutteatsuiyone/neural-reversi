import NeuralReversi.Flip.ScalarProperties
import NeuralReversi.Simd.TernaryProperties

namespace NeuralReversi.Flip.Avx512.Algebra
open NeuralReversi.Flip.Scalar
open NeuralReversi.Simd.X86 (ternaryBit ternaryWord wordFromBits_bit)

set_option maxRecDepth 8192
set_option maxHeartbeats 2000000

theorem ternaryBit_80 : ∀ a b c, ternaryBit 0x80 a b c = (a && b && c) := by decide
theorem ternaryBit_28 : ∀ a b c, ternaryBit 0x28 a b c = (c && (a ^^ b)) := by decide
theorem ternaryBit_f2 : ∀ a b c, ternaryBit 0xf2 a b c = (a || (c && !b)) := by decide

theorem ternaryWord_80 (a b c : Word) : ternaryWord a b c 0x80 = a &&& b &&& c := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  rw [ternaryWord, wordFromBits_bit _ i hi, ternaryBit_80]
  simp only [BitVec.getLsbD_and]

theorem ternaryWord_28 (a b c : Word) : ternaryWord a b c 0x28 = c &&& (a ^^^ b) := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  rw [ternaryWord, wordFromBits_bit _ i hi, ternaryBit_28]
  simp only [BitVec.getLsbD_and, BitVec.getLsbD_xor]

theorem ternaryWord_f2 (a b c : Word) : ternaryWord a b c 0xf2 = a ||| (c &&& ~~~b) := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  rw [ternaryWord, wordFromBits_bit _ i hi, ternaryBit_f2]
  simp only [BitVec.getLsbD_and, BitVec.getLsbD_or, BitVec.getLsbD_not,
    hi, decide_true, Bool.true_and]

-- Value semantics of raw packed VPSRLVQ; the count is a full u64 lane.
def rawShift (x count : Word) : Word :=
  if count.toNat < 64 then x >>> count.toNat else 0

def msb : Word := 0x8000000000000000#64

theorem msb_eq_one : msb = one 63 := by decide

theorem rawShift_large (x count : Word) (hc : 64 ≤ count.toNat) : rawShift x count = 0 := by
  simp only [rawShift, Nat.not_lt_of_ge hc, ↓reduceIte]

theorem rawShift_count64 (x : Word) : rawShift x 64#64 = 0 := by
  apply rawShift_large
  decide

theorem clz_zero : (0 : Word).clz = 64#64 := BitVec.clz_eq_iff_eq_zero.mpr rfl

theorem rawShift_msb_zero : rawShift msb (0 : Word).clz = 0 := by
  rw [clz_zero, rawShift_count64]

theorem rawShift_highest (x : Word) (hx : x ≠ 0) :
    rawShift msb x.clz = one (63 - x.clz.toNat) := by
  have hc : x.clz.toNat < 64 := by
    simpa [BitVec.lt_def] using (BitVec.clz_lt_iff_ne_zero.mpr hx)
  simp only [rawShift, hc, ↓reduceIte, msb_eq_one]
  apply BitVec.eq_of_getLsbD_eq
  intro i _
  rw [BitVec.getLsbD_ushiftRight, one_bit 63 (by decide), one_bit _ (by omega)]
  apply Bool.eq_iff_iff.mpr
  simp only [decide_eq_true_eq]
  omega

theorem rawShift_selector (x : Word) :
    rawShift msb x.clz = if x = 0 then 0 else one (63 - x.clz.toNat) := by
  by_cases hx : x = 0
  · subst x
    simp only [rawShift_msb_zero, ↓reduceIte]
  · rw [ite_eq_right hx]
    exact rawShift_highest x hx

theorem one_ne_zero (j : Nat) (hj : j < 64) : one j ≠ 0 := by
  intro h
  have hb := congrArg (fun w : Word => w.getLsbD j) h
  rw [one_bit j hj] at hb
  simp at hb

theorem one_and_player (j : Nat) (hj : j < 64) (player : Word) :
    one j &&& player = if player.getLsbD j = true then one j else 0 := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  by_cases hp : player.getLsbD j = true
  · simp only [hp, ↓reduceIte, BitVec.getLsbD_and, one_bit j hj]
    by_cases heq : i = j <;> simp_all
  · simp only [hp, BitVec.getLsbD_and, one_bit j hj]
    by_cases heq : i = j <;> simp_all

def anchor (x player : Word) : Word := ternaryWord x (0#64 - x) player 0x80

theorem anchor_form (x player : Word) : anchor x player = (x &&& -x) &&& player := by
  rw [anchor, ternaryWord_80, BitVec.zero_sub]

def ZeroOrOne (a : Word) : Prop := a = 0 ∨ ∃ j, j < 64 ∧ a = one j

theorem anchor_zeroOrOne (x player : Word) : ZeroOrOne (anchor x player) := by
  by_cases hx : x = 0
  · left
    simp [anchor_form, hx]
  · have hj := least_ctz x hx
    have hjw := BitVec.lt_of_getLsbD hj.1
    rw [anchor_form, isolate_eq_one x _ hj, one_and_player _ hjw]
    by_cases hp : player.getLsbD x.ctz.toNat = true
    · simp only [hp, ↓reduceIte]
      exact .inr ⟨x.ctz.toNat, hjw, rfl⟩
    · simp only [hp]
      exact .inl rfl

def cmpFlank (a : Word) : Word := (if a = 0 then BitVec.allOnes 64 else 0#64) - a
def signedMin (a b : Word) : Word := if a.toInt ≤ b.toInt then a else b
def minFlank (a : Word) : Word := signedMin (0#64 - a) (BitVec.allOnes 64)

theorem signedMin_strict (a b : Word) :
    signedMin a b = if a.toInt < b.toInt then a else b := by
  by_cases hlt : a.toInt < b.toInt
  · simp only [signedMin, hlt, Int.le_of_lt hlt, ↓reduceIte]
  · by_cases hle : a.toInt ≤ b.toInt
    · have heq : a = b := BitVec.eq_of_toInt_eq (by omega)
      subst b
      simp [signedMin]
    · simp only [signedMin, hlt, hle, ↓reduceIte]

theorem cmpFlank_zero : cmpFlank 0 = BitVec.allOnes 64 := by simp [cmpFlank]

theorem cmpFlank_one (j : Nat) (hj : j < 64) : cmpFlank (one j) = -one j := by
  rw [cmpFlank, ite_eq_right (one_ne_zero j hj), BitVec.zero_sub]

theorem negative_one_msb (j : Nat) (hj : j < 64) : (-one j).msb = true := by
  rw [BitVec.msb_eq_getLsbD_last]
  simp only [show 64 - 1 = 63 by decide, BitVec.getLsbD_neg, one_bit j hj]
  have hex : (∃ k < 63, decide (k = j) = true) ↔ j < 63 := by simp
  simp only [hex]
  by_cases heq : j = 63
  · subst j
    decide
  · have hlt : j < 63 := by omega
    have hne : ¬ 63 = j := by omega
    simp [hlt, hne]

theorem negative_one_toInt (j : Nat) (hj : j < 64) : (-one j).toInt ≤ -1 := by
  have h := BitVec.toInt_neg_of_msb_true (negative_one_msb j hj)
  omega

theorem minFlank_eq_cmpFlank (a : Word) (ha : ZeroOrOne a) : minFlank a = cmpFlank a := by
  rcases ha with rfl | ⟨j, hj, rfl⟩
  · simp [minFlank, signedMin, cmpFlank]
  · have hle : (-one j).toInt ≤ (BitVec.allOnes 64).toInt := by
      have hones : (BitVec.allOnes 64).toInt = -1 := by decide
      rw [hones]
      exact negative_one_toInt j hj
    rw [minFlank, BitVec.zero_sub, signedMin, ite_eq_left hle, cmpFlank_one j hj]

theorem minFlank_not_unrestricted :
    minFlank 0x8000000000000001#64 ≠ cmpFlank 0x8000000000000001#64 := by decide

theorem neg_xor_bit (a : Word) (i : Nat) :
    (-a ^^^ a).getLsbD i = true ↔
      i < 64 ∧ ∃ k, k < i ∧ a.getLsbD k = true := by
  simp only [BitVec.getLsbD_xor, BitVec.getLsbD_neg]
  have cancel : ∀ b c : Bool, ((b ^^ c) ^^ b) = c := by decide
  rw [cancel]
  simp only [Bool.and_eq_true, decide_eq_true_eq]

theorem one_negative_suffix (j : Nat) (hj : j < 64) :
    -one j ^^^ one j = ~~~(one j ||| (one j - 1#64)) := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  apply Bool.eq_iff_iff.mpr
  rw [neg_xor_bit, one_upper_bit j hj]
  simp [one_bit j hj, hi]

def leftCmpWord (m player notOpponent : Word) : Word :=
  m &&& ~~~cmpFlank (anchor (notOpponent &&& m) player)

def leftMinWord (m player notOpponent : Word) : Word :=
  m &&& ~~~minFlank (anchor (notOpponent &&& m) player)

theorem leftCmpWord_scalar (m player notOpponent : Word) :
    leftCmpWord m player notOpponent = flipLeft m player notOpponent := by
  by_cases hx : notOpponent &&& m = 0
  · simp [leftCmpWord, flipLeft, hx, anchor_form, cmpFlank]
  · have hj := least_ctz (notOpponent &&& m) hx
    have hjw := BitVec.lt_of_getLsbD hj.1
    have hf := isolate_eq_one (notOpponent &&& m) _ hj
    have ha : anchor (notOpponent &&& m) player =
        if player.getLsbD (notOpponent &&& m).ctz.toNat = true then
          one (notOpponent &&& m).ctz.toNat else 0 := by
      rw [anchor_form, hf, one_and_player _ hjw]
    by_cases hp : player.getLsbD (notOpponent &&& m).ctz.toNat = true
    · simp only [leftCmpWord, ha, hp, ↓reduceIte, cmpFlank_one _ hjw, BitVec.not_neg,
        flipLeft, hf, one_anchor _ hjw]
    · simp only [leftCmpWord, ha, hp, Bool.false_eq_true, ↓reduceIte, cmpFlank_zero, BitVec.not_allOnes,
        BitVec.and_zero, flipLeft, hf, one_anchor _ hjw]
      rfl

theorem leftMinWord_scalar (m player notOpponent : Word) :
    leftMinWord m player notOpponent = flipLeft m player notOpponent := by
  unfold leftMinWord
  rw [minFlank_eq_cmpFlank _ (anchor_zeroOrOne _ _)]
  exact leftCmpWord_scalar m player notOpponent

def rightAnchor (x player : Word) : Word := rawShift msb x.clz &&& player

theorem rightAnchor_zero (player : Word) : rightAnchor 0 player = 0 := by
  rw [rightAnchor, rawShift_msb_zero]
  simp

def rightWord (m player notOpponent : Word) : Word :=
  let a := rightAnchor (notOpponent &&& m) player
  ternaryWord (0#64 - a) a m 0x28

theorem rightWord_scalar (m player notOpponent : Word) :
    rightWord m player notOpponent = flipRightValue m player notOpponent := by
  by_cases hx : notOpponent &&& m = 0
  · simp only [rightWord, hx, rightAnchor_zero, flipRightValue, ↓reduceIte]
    change ternaryWord (0#64 - 0#64) 0#64 m 0x28 = 0#64
    rw [ternaryWord_28, BitVec.sub_self, BitVec.xor_self, BitVec.and_zero]
  · have hj := greatest_clz (notOpponent &&& m) hx
    have hjw := BitVec.lt_of_getLsbD hj.1
    have hc := (count_arithmetic (notOpponent &&& m) hx).2.2.1
    have hf : 1#64 <<< shiftCount32 (notOpponent &&& m) =
        one (63 - (notOpponent &&& m).clz.toNat) := by
      rw [BitVec.shiftLeft_eq', hc]
      rfl
    have ha : rightAnchor (notOpponent &&& m) player =
        if player.getLsbD (63 - (notOpponent &&& m).clz.toNat) = true then
          one (63 - (notOpponent &&& m).clz.toNat) else 0 := by
      rw [rightAnchor, rawShift_highest _ hx, one_and_player _ hjw]
    by_cases hp : player.getLsbD (63 - (notOpponent &&& m).clz.toNat) = true
    · simp only [rightWord, ha, hp, ↓reduceIte, ternaryWord_28, BitVec.zero_sub,
        one_negative_suffix _ hjw, flipRightValue, hx, hf, one_anchor _ hjw]
    · simp only [rightWord, ha, hp, ↓reduceIte, flipRightValue, hx, hf, one_anchor _ hjw]
      change ternaryWord (0#64 - 0#64) 0#64 m 0x28 = 0#64
      rw [ternaryWord_28, BitVec.sub_self, BitVec.xor_self, BitVec.and_zero]

def combinedCmp (leftMask rightMask player notOpponent : Word) : Word :=
  ternaryWord (rightWord rightMask player notOpponent)
    (cmpFlank (anchor (notOpponent &&& leftMask) player)) leftMask 0xf2

def combinedMin (leftMask rightMask player notOpponent : Word) : Word :=
  ternaryWord (rightWord rightMask player notOpponent)
    (minFlank (anchor (notOpponent &&& leftMask) player)) leftMask 0xf2

theorem combinedCmp_scalar (leftMask rightMask player notOpponent : Word) :
    combinedCmp leftMask rightMask player notOpponent =
      flipRightValue rightMask player notOpponent ||| flipLeft leftMask player notOpponent := by
  rw [combinedCmp, ternaryWord_f2]
  change rightWord rightMask player notOpponent ||| leftCmpWord leftMask player notOpponent = _
  rw [rightWord_scalar, leftCmpWord_scalar]

theorem combinedMin_scalar (leftMask rightMask player notOpponent : Word) :
    combinedMin leftMask rightMask player notOpponent =
      flipRightValue rightMask player notOpponent ||| flipLeft leftMask player notOpponent := by
  rw [combinedMin, ternaryWord_f2]
  change rightWord rightMask player notOpponent ||| leftMinWord leftMask player notOpponent = _
  rw [rightWord_scalar, leftMinWord_scalar]

end NeuralReversi.Flip.Avx512.Algebra

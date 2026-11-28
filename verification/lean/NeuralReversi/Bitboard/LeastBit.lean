module
public import NeuralReversi.Bitboard.Model
public import NeuralReversi.Square.Model

@[expose] public section

namespace NeuralReversi.Bitboard

theorem clear_bit_spec (bits : BitVec width) (i : Nat) :
    (bits &&& (bits - 1#width)).getLsbD i = true ↔
      bits.getLsbD i = true ∧ ∃ j < i, bits.getLsbD j = true := by
  rw [← BitVec.not_neg]
  simp only [BitVec.getLsbD_and, BitVec.getLsbD_not, BitVec.getLsbD_neg]
  by_cases hi : i < width
  · simp only [hi, decide_true, Bool.true_and]
    cases h : bits.getLsbD i <;> simp
  · have hz : bits.getLsbD i = false := BitVec.getLsbD_of_ge bits i (by omega)
    simp [hz]

def ClearLeastBit : Prop :=
  ∀ (bits : Bits) (i : Nat),
    (clearLsb bits).getLsbD i = true ↔
      bits.getLsbD i = true ∧ ∃ j < i, bits.getLsbD j = true

theorem clearLeastBit_proven : ClearLeastBit := by
  intro bits i
  exact clear_bit_spec bits i

theorem clearLsb_ctz (bits : Bits) (nonzero : bits ≠ 0) (i : Nat) :
    (clearLsb bits).getLsbD i = true ↔
      bits.getLsbD i = true ∧ bits.ctz.toNat < i := by
  rw [clearLeastBit_proven]
  constructor
  · rintro ⟨hi, j, hji, hj⟩
    refine ⟨hi, ?_⟩
    have : bits.ctz.toNat ≤ j := by
      by_cases h : bits.ctz.toNat ≤ j
      · exact h
      · have hz := BitVec.getLsbD_false_of_lt_ctz (x := bits) (i := j) (by omega)
        simp [hj] at hz
    omega
  · rintro ⟨hi, hci⟩
    exact ⟨hi, bits.ctz.toNat, hci, BitVec.getLsbD_true_ctz_of_ne_zero nonzero⟩

theorem clearLsb_zero_iff_atMostOne (bits : Bits) :
    clearLsb bits = 0 ↔
      ∀ i j, bits.getLsbD i = true → bits.getLsbD j = true → i = j := by
  constructor
  · intro hz i j hi hj
    have not_lower : ∀ a b, bits.getLsbD a = true → bits.getLsbD b = true → ¬ b < a := by
      intro a b ha hb hba
      have hc := (clearLeastBit_proven bits a).mpr ⟨ha, b, hba, hb⟩
      simp [hz] at hc
    have := not_lower i j hi hj
    have := not_lower j i hj hi
    omega
  · intro unique
    apply BitVec.eq_of_getLsbD_eq
    intro i hiw
    have hn : ¬ (clearLsb bits).getLsbD i = true := by
      intro hc
      obtain ⟨hi, j, hji, hj⟩ := (clearLeastBit_proven bits i).mp hc
      have := unique i j hi hj
      omega
    simpa using hn

def SingleBitContract : Prop :=
  ∀ bits : Bits,
    (hasSingleBitNonzero bits = true ↔
      ∀ i j, bits.getLsbD i = true → bits.getLsbD j = true → i = j) ∧
    (bits ≠ 0 → (hasSingleBitNonzero bits = true ↔
      ∃ i, bits.getLsbD i = true ∧ ∀ j, bits.getLsbD j = true → j = i))

theorem singleBitContract_proven : SingleBitContract := by
  intro bits
  have criterion : hasSingleBitNonzero bits = true ↔
      ∀ i j, bits.getLsbD i = true → bits.getLsbD j = true → i = j := by
    simpa [hasSingleBitNonzero, clearLsb] using clearLsb_zero_iff_atMostOne bits
  refine ⟨criterion, ?_⟩
  intro nonzero
  rw [criterion]
  constructor
  · intro unique
    exact ⟨bits.ctz.toNat, BitVec.getLsbD_true_ctz_of_ne_zero nonzero,
      fun j hj => unique j bits.ctz.toNat hj (BitVec.getLsbD_true_ctz_of_ne_zero nonzero)⟩
  · rintro ⟨i, _, unique⟩ j k hj hk
    exact (unique j hj).trans (unique k hk).symm

theorem trailingZeros_zero : (0 : Bits).ctz.toNat = 64 := by
  have zero : ((0 : Bits).reverse).clz = (64 : Bits) :=
    BitVec.clz_eq_iff_eq_zero.mpr (by simp)
  have number := congrArg BitVec.toNat zero
  simpa [BitVec.ctz] using number

theorem trailingZeros_bound (bits : Bits) : bits.ctz.toNat ≤ 64 := by
  by_cases zero : bits = 0
  · subst bits
    exact Nat.le_of_eq trailingZeros_zero
  · have bound := (BitVec.ctz_lt_iff_ne_zero (x := bits)).mpr zero
    have : bits.ctz.toNat < 64 := by simpa [BitVec.lt_def] using bound
    omega

def lsbSquare (bits : Bits) : Option NeuralReversi.Square.Square :=
  if h : bits.ctz.toNat < 64 then some ⟨bits.ctz.toNat, h⟩ else none

def lsbSquareUnchecked (debugAssertions : Bool) (bits : Bits) :
    Except String NeuralReversi.Square.Square :=
  if debugAssertions && bits == 0 then .error "lsb_square_unchecked called on empty bitboard"
  else if h : bits.ctz.toNat < 64 then .ok ⟨bits.ctz.toNat, h⟩
  else .error "invalid Square discriminant: 64"

def IsLeastSetSquare (bits : Bits) (sq : NeuralReversi.Square.Square) : Prop :=
  sq.val < 64 ∧ bits.getLsbD sq.val = true ∧ ∀ i < sq.val, bits.getLsbD i = false

def LeastSquareContract : Prop :=
  lsbSquareUnchecked true 0 = .error "lsb_square_unchecked called on empty bitboard" ∧
  lsbSquareUnchecked false 0 = .error "invalid Square discriminant: 64" ∧
  lsbSquare 0 = none ∧
  ∀ bits : Bits, bits ≠ 0 → ∃ sq,
    lsbSquare bits = some sq ∧
    lsbSquareUnchecked false bits = .ok sq ∧
    lsbSquareUnchecked true bits = .ok sq ∧ IsLeastSetSquare bits sq

set_option maxRecDepth 4096 in
theorem leastSquareContract_proven : LeastSquareContract := by
  have h0 : (0#64).ctz.toNat = 64 := trailingZeros_zero
  refine ⟨by simp [lsbSquareUnchecked], ?_, ?_, ?_⟩
  · simp [lsbSquareUnchecked, h0]
  · simp [lsbSquare, h0]
  · intro bits nonzero
    have hlt : bits.ctz.toNat < 64 := by
      have bound := (BitVec.ctz_lt_iff_ne_zero (x := bits)).mpr nonzero
      simpa [BitVec.lt_def] using bound
    let sq : NeuralReversi.Square.Square := ⟨bits.ctz.toNat, hlt⟩
    refine ⟨sq, ?_, ?_, ?_, ?_⟩
    · simp [lsbSquare, hlt, sq]
    · simp [lsbSquareUnchecked, hlt, sq]
    · simp_all [lsbSquareUnchecked, sq]
    · exact ⟨hlt, BitVec.getLsbD_true_ctz_of_ne_zero nonzero,
        fun _ hi => BitVec.getLsbD_false_of_lt_ctz hi⟩

def popLsb (debugAssertions : Bool) (bits : Bits) :
    Except String (NeuralReversi.Square.Square × Bits) :=
  if debugAssertions && bits == 0 then .error "pop_lsb called on empty bitboard"
  else (lsbSquareUnchecked debugAssertions bits).map fun sq => (sq, clearLsb bits)

def PopLeastContract : Prop :=
  popLsb true 0 = .error "pop_lsb called on empty bitboard" ∧
  popLsb false 0 = .error "invalid Square discriminant: 64" ∧
  ∀ bits : Bits, bits ≠ 0 → ∃ sq rest,
    popLsb false bits = .ok (sq, rest) ∧ popLsb true bits = .ok (sq, rest) ∧
    IsLeastSetSquare bits sq ∧
    ∀ i, rest.getLsbD i = true ↔ bits.getLsbD i = true ∧ sq.val < i

set_option maxRecDepth 4096 in
theorem popLeastContract_proven : PopLeastContract := by
  have h0 : (0#64).ctz.toNat = 64 := trailingZeros_zero
  refine ⟨by simp [popLsb], ?_, ?_⟩
  · simp [popLsb, lsbSquareUnchecked, h0, Except.map]
  · intro bits nonzero
    have hlt : bits.ctz.toNat < 64 := by
      have bound := (BitVec.ctz_lt_iff_ne_zero (x := bits)).mpr nonzero
      simpa [BitVec.lt_def] using bound
    let sq : NeuralReversi.Square.Square := ⟨bits.ctz.toNat, hlt⟩
    refine ⟨sq, clearLsb bits, ?_, ?_, ?_, ?_⟩
    · simp [popLsb, lsbSquareUnchecked, hlt, sq, Except.map]
    · simp_all [popLsb, lsbSquareUnchecked, sq, Except.map]
    · exact ⟨hlt, BitVec.getLsbD_true_ctz_of_ne_zero nonzero,
        fun _ hi => BitVec.getLsbD_false_of_lt_ctz hi⟩
    · exact clearLsb_ctz bits nonzero

end NeuralReversi.Bitboard

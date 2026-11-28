module
public import NeuralReversi.Eval.PatternUpdate.Encoding
public import NeuralReversi.Bitboard.LeastBit

@[expose] public section

namespace NeuralReversi.Eval.PatternUpdate

def patterns : Vector (List Nat) 32 := #v[
  [10, 11, 12, 13, 18, 19, 20, 21],
  [50, 51, 52, 53, 42, 43, 44, 45],
  [17, 25, 33, 41, 18, 26, 34, 42],
  [22, 30, 38, 46, 21, 29, 37, 45],
  [0, 9, 18, 27, 36, 45, 54, 63],
  [7, 14, 21, 28, 35, 42, 49, 56],
  [26, 27, 28, 29, 34, 35, 36, 37],
  [19, 20, 27, 28, 35, 36, 43, 44],
  [0, 1, 2, 3, 4, 5, 6, 7],
  [56, 57, 58, 59, 60, 61, 62, 63],
  [0, 8, 16, 24, 32, 40, 48, 56],
  [7, 15, 23, 31, 39, 47, 55, 63],
  [1, 2, 3, 4, 9, 10, 11, 12],
  [6, 5, 4, 3, 14, 13, 12, 11],
  [57, 58, 59, 60, 49, 50, 51, 52],
  [62, 61, 60, 59, 54, 53, 52, 51],
  [8, 16, 24, 32, 9, 17, 25, 33],
  [48, 40, 32, 24, 49, 41, 33, 25],
  [15, 23, 31, 39, 14, 22, 30, 38],
  [55, 47, 39, 31, 54, 46, 38, 30],
  [0, 1, 2, 8, 9, 10, 16, 17, 18],
  [7, 6, 5, 15, 14, 13, 23, 22, 21],
  [56, 57, 58, 48, 49, 50, 40, 41, 42],
  [63, 62, 61, 55, 54, 53, 47, 46, 45],
  [9, 10, 11, 17, 18, 19, 25, 26, 27],
  [14, 13, 12, 22, 21, 20, 30, 29, 28],
  [49, 50, 51, 41, 42, 43, 33, 34, 35],
  [54, 53, 52, 46, 45, 44, 38, 37, 36],
  [1, 10, 19, 28, 37, 46, 55],
  [8, 17, 26, 35, 44, 53, 62],
  [6, 13, 20, 27, 34, 41, 48],
  [15, 22, 29, 36, 43, 50, 57]
]

def pattern (i : Fin 32) : List Nat := patterns[i]

def PatternGeometryContract : Prop :=
  ∀ i : Fin 32,
    (pattern i).Nodup ∧ (pattern i).length ≤ 9 ∧
    (pattern i).all (fun sq => sq < 64) = true

theorem patternGeometryContract_proven : PatternGeometryContract := by
  unfold PatternGeometryContract
  decide

def patternCount (sq : Nat) : Nat := patterns.toList.countP (·.contains sq)

def SquareCoverageContract : Prop :=
  ∀ sq : Fin 64, 0 < patternCount sq.val ∧ patternCount sq.val ≤ 5

theorem squareCoverageContract_proven : SquareCoverageContract := by
  unfold SquareCoverageContract
  decide

open Bitboard (Bits clearLsb)

def flipDigits (flipped : Bits) (i : Nat) : Int := bit (flipped.getLsbD i)

def flipCode (squares : List Nat) (flipped : Bits) : Int := code (flipDigits flipped) squares

def chunkMask (bits : Bits) (chunk : Nat) : BitVec 16 :=
  (bits >>> (16 * chunk)).setWidth 16

def ChunkIndexContract : Prop :=
  ∀ bits chunk, chunk < 4 →
    let mask := chunkMask bits chunk
    mask.toNat = ((bits >>> (16 * chunk)) &&& 65535#64).toNat ∧
    chunk * 65536 + mask.toNat < 4 * 65536 ∧
    (chunk * 65536 + mask.toNat + 1) * 64 ≤ 16777216

theorem chunkIndexContract_proven : ChunkIndexContract := by
  intro bits chunk valid
  let mask := chunkMask bits chunk
  have limit := mask.isLt
  refine ⟨?_, by omega, by omega⟩
  unfold chunkMask
  rw [BitVec.toNat_setWidth, BitVec.toNat_and]
  change (bits >>> (16 * chunk)).toNat % 2^16 =
    (bits >>> (16 * chunk)).toNat &&& (2^16 - 1)
  rw [Nat.and_two_pow_sub_one_eq_mod]

def chunkDigit (bits : Bits) (chunk i : Nat) : Int :=
  if i / 16 = chunk then bit ((chunkMask bits chunk).getLsbD (i % 16)) else 0

private theorem chunkDigit_eq (bits : Bits) (chunk i : Nat) :
    chunkDigit bits chunk i = if i / 16 = chunk then flipDigits bits i else 0 := by
  unfold chunkDigit
  split
  · rename_i h
    unfold chunkMask flipDigits
    rw [BitVec.getLsbD_setWidth, BitVec.getLsbD_ushiftRight]
    have lt : i % 16 < 16 := Nat.mod_lt _ (by decide)
    simp only [lt, decide_true, Bool.true_and]
    rw [← h, Nat.div_add_mod]
  · rfl

def ChunkSumContract : Prop :=
  ∀ squares bits, (∀ i ∈ squares, i < 64) →
    code (chunkDigit bits 0) squares + code (chunkDigit bits 1) squares +
      code (chunkDigit bits 2) squares + code (chunkDigit bits 3) squares =
      flipCode squares bits

theorem chunkSumContract_proven : ChunkSumContract := by
  intro squares bits valid
  rw [← code_add, ← code_add, ← code_add]
  apply code_congr
  intro i hi
  have bound := valid i hi
  have loc : i / 16 = 0 ∨ i / 16 = 1 ∨ i / 16 = 2 ∨ i / 16 = 3 := by omega
  simp only [chunkDigit_eq]
  rcases loc with h | h | h | h <;> simp [h]

private theorem clear_decreases (bits : Bits) (nonzero : bits ≠ 0) :
    (clearLsb bits).toNat < bits.toNat := by
  have nonzeroNat : bits.toNat ≠ 0 := by
    intro zero
    apply nonzero
    apply BitVec.eq_of_toNat_eq
    simpa using zero
  unfold clearLsb
  rw [BitVec.toNat_and, BitVec.toNat_sub_of_le (by rw [BitVec.le_def]; simp; omega)]
  change (bits.toNat &&& (bits.toNat - 1)) < bits.toNat
  have bound : bits.toNat &&& (bits.toNat - 1) ≤ bits.toNat - 1 := Nat.and_le_right
  omega

def sumSetBits (weight : Nat → Int) (bits : Bits) : Int :=
  if _h : bits = 0 then 0
  else weight bits.ctz.toNat + sumSetBits weight (clearLsb bits)
termination_by bits.toNat
decreasing_by exact clear_decreases bits _h

private theorem flipDigits_clear (bits : Bits) (nonzero : bits ≠ 0) (i : Nat) :
    flipDigits bits i = squareIndicator bits.ctz.toNat i + flipDigits (clearLsb bits) i := by
  have keep := Bitboard.clearLsb_ctz bits nonzero i
  by_cases h : bits.ctz.toNat = i
  · subst i
    have rest : (clearLsb bits).getLsbD bits.ctz.toNat = false := by
      cases hk : (clearLsb bits).getLsbD bits.ctz.toNat
      · rfl
      · exact False.elim ((Nat.lt_irrefl _) (keep.mp hk).2)
    simp [flipDigits, squareIndicator, bit, BitVec.getLsbD_true_ctz_of_ne_zero nonzero, rest]
  · have eq : (clearLsb bits).getLsbD i = bits.getLsbD i := by
      cases old : bits.getLsbD i
      · simpa [old] using keep
      · have order : bits.ctz.toNat < i := by
          by_cases past : i < bits.ctz.toNat
          · have z := BitVec.getLsbD_false_of_lt_ctz past
            simp [old] at z
          · omega
        exact keep.mpr ⟨old, order⟩
    simp [squareIndicator, bit, flipDigits, h, eq]

def FlipEnumerationContract : Prop :=
  ∀ squares, squares.Nodup → ∀ bits,
    sumSetBits (fun q => lookupWeight q squares) bits = flipCode squares bits

theorem flipEnumerationContract_proven : FlipEnumerationContract := by
  intro squares unique
  exact go squares unique
where
  go (squares : List Nat) (unique : squares.Nodup) (bits : Bits) :
      sumSetBits (fun q => lookupWeight q squares) bits = flipCode squares bits := by
    rw [sumSetBits]
    split
    · rename_i h
      subst bits
      symm
      unfold flipCode
      rw [code_congr squares _ (fun _ => 0) (fun i _ => by simp [flipDigits, bit]), code_zero]
    · rename_i h
      rw [go squares unique (clearLsb bits), lookupWeightContract_proven squares _ unique]
      unfold flipCode
      rw [← code_add]
      exact code_congr squares _ _ (fun i _ => (flipDigits_clear bits h i).symm)
  termination_by bits.toNat
  decreasing_by apply clear_decreases; assumption

end NeuralReversi.Eval.PatternUpdate

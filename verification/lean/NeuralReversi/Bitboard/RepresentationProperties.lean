import NeuralReversi.Bitboard.RepresentationModel
import NeuralReversi.Bitboard.SquareProperties
import NeuralReversi.Bitboard.Counting

namespace NeuralReversi.Bitboard

def RawRepresentationContract : Prop :=
  (∀ raw : Bits,
    bits (new raw) = raw ∧ intoU64 (new raw) = raw ∧
    bits (fromU64 raw) = raw ∧ intoU64 (fromU64 raw) = raw ∧
    (∀ (sq : Square) (checked : Bool),
      valueContains checked (new raw) sq = .ok (raw.toNat.testBit sq.val) ∧
      valueContains checked (fromU64 raw) sq = .ok (raw.toNat.testBit sq.val)) ∧
    (valueCount (new raw)).toNat = (List.range 64).countP raw.toNat.testBit ∧
    (valueCount (fromU64 raw)).toNat = (List.range 64).countP raw.toNat.testBit) ∧
  (∀ value : Value, new (bits value) = value ∧ fromU64 (intoU64 value) = value)

theorem rawRepresentationContract_proven : RawRepresentationContract := by
  constructor
  · intro raw
    refine ⟨rfl, rfl, rfl, rfl, ?_, ?_, ?_⟩
    · intro sq checked
      exact ⟨squareMembershipContract_proven raw sq checked checked,
        squareMembershipContract_proven raw sq checked checked⟩
    · exact (populationCountContract_proven raw).1
    · exact (populationCountContract_proven raw).1
  · intro value
    cases value
    exact ⟨rfl, rfl⟩

theorem new_injective (a b : Bits) (equal : new a = new b) : a = b := by
  exact congrArg bits equal

theorem intoU64_injective (a b : Value) (equal : intoU64 a = intoU64 b) : a = b := by
  have reconstruct := rawRepresentationContract_proven.2
  rw [← (reconstruct a).2, ← (reconstruct b).2, equal]

def SquareValueConversionContract : Prop :=
  ∀ (sq : Square) (checked : Bool), ∃ value : Value,
    fromSquareValue checked sq = .ok value ∧
    (∀ i, value.storage.getLsbD i = decide (i = sq.val)) ∧
    (valueCount value).toNat = 1

theorem squareValueConversionContract_proven : SquareValueConversionContract := by
  intro sq checked
  refine ⟨⟨squareMask sq⟩, ?_, squareMask_bit sq, ?_⟩
  · rw [fromSquareValue, (squareMaskContract_proven sq checked checked).2.1]
    rfl
  · change (count (squareMask sq)).toNat = 1
    rw [(populationCountContract_proven (squareMask sq)).1]
    have cells : (squareMask sq).getLsbD = fun i => decide (i = sq.val) := funext (squareMask_bit sq)
    rw [cells]
    simpa [List.count_eq_countP, Bool.beq_eq_decide_eq, sq.isLt] using
      (List.count_range (a := sq.val) (n := 64))

theorem displayChar_eq (bits : Bits) (rank file : Nat) :
    displayChar bits rank file = if bits.getLsbD (rank * 8 + file) then '1' else '.' := by
  unfold displayChar
  dsimp only
  have hbit : (((bits >>> (rank * 8 + file)) &&& 1#64) != 0#64) = bits.getLsbD (rank * 8 + file) := by
    cases h : bits.getLsbD (rank * 8 + file)
    · have hzero : ((bits >>> (rank * 8 + file)) &&& 1#64) = 0#64 := by
        apply BitVec.eq_of_getLsbD_eq
        intro i hi
        simp only [BitVec.getLsbD_and, BitVec.getLsbD_ushiftRight, BitVec.getLsbD_zero]
        by_cases hi0 : i = 0
        · subst hi0
          simp [h]
        · have h1 : (1#64).getLsbD i = false := by
            simp [BitVec.getLsbD_one, hi0]
          simp [h1]
      simp [hzero]
    · have hne : ((bits >>> (rank * 8 + file)) &&& 1#64) ≠ 0#64 := by
        intro hzero
        have h0 := congrArg (fun v : Bits => v.getLsbD 0) hzero
        simp [h] at h0
      simp [hne]
  rw [hbit]

theorem displayRank_length (bits : Bits) (rank : Nat) :
    (displayRank bits rank).length = 9 := by
  simp [displayRank]

private theorem flatten_uniform_length {α : Type} (f : Nat → List α) (k : Nat)
    (hlen : ∀ r, (f r).length = k) (n : Nat) :
    ((List.range n).map f).flatten.length = n * k := by
  induction n with
  | zero => simp
  | succ n ih =>
    simp [List.range_succ, List.flatten_append, ih, hlen, Nat.succ_mul]

theorem displayChars_length (value : Value) :
    (displayChars value).length = 72 := by
  exact flatten_uniform_length (displayRank value.storage) 9 (displayRank_length value.storage) 8

private theorem flatten_uniform_getElem {α : Type} (f : Nat → List α) (k : Nat)
    (hlen : ∀ r, (f r).length = k) (n : Nat) :
    ∀ (r c : Nat), r < n → (hc : c < k) →
      (hidx : r * k + c < ((List.range n).map f).flatten.length) →
      ((List.range n).map f).flatten[r * k + c]'hidx = (f r)[c]'(by rw [hlen]; exact hc) := by
  induction n with
  | zero => intro r _ hr; omega
  | succ n ih =>
    intro r c hr hc hidx
    have hflat : ((List.range (n + 1)).map f).flatten =
        ((List.range n).map f).flatten ++ f n := by
      simp [List.range_succ, List.flatten_append]
    have hpre_len : ((List.range n).map f).flatten.length = n * k :=
      flatten_uniform_length f k hlen n
    by_cases hlt : r < n
    · have hlt_idx : r * k + c < ((List.range n).map f).flatten.length := by
        rw [hpre_len]
        have hle : (r + 1) * k ≤ n * k := Nat.mul_le_mul_right k hlt
        rw [Nat.add_mul, Nat.one_mul] at hle
        omega
      have h1 : ((List.range (n + 1)).map f).flatten[r * k + c]'hidx =
          (((List.range n).map f).flatten ++ f n)[r * k + c]'(by rw [← hflat]; exact hidx) := by
        congr 1
      rw [h1, List.getElem_append_left hlt_idx]
      exact ih r c hlt hc hlt_idx
    · have hrn : r = n := by omega
      subst r
      have hge_idx : ((List.range n).map f).flatten.length ≤ n * k + c := by
        rw [hpre_len]; omega
      have h1 : ((List.range (n + 1)).map f).flatten[n * k + c]'hidx =
          (((List.range n).map f).flatten ++ f n)[n * k + c]'(by rw [← hflat]; exact hidx) := by
        congr 1
      rw [h1, List.getElem_append_right hge_idx]
      simp [hpre_len]

theorem displayChars_getElem (value : Value) (rank file : Nat) (hr : rank < 8) (hf : file < 8) :
    (displayChars value)[rank * 9 + file]'(by rw [displayChars_length]; omega) =
      if value.storage.getLsbD (rank * 8 + file) then '1' else '.' := by
  have h := flatten_uniform_getElem (displayRank value.storage) 9
    (displayRank_length value.storage) 8 rank file hr (by omega)
    (by change rank * 9 + file < (displayChars value).length; rw [displayChars_length]; omega)
  refine h.trans ?_
  simp [displayRank, List.getElem_append_left, hf, displayChar_eq]

theorem displayChars_newline (value : Value) (rank : Nat) (hr : rank < 8) :
    (displayChars value)[rank * 9 + 8]'(by rw [displayChars_length]; omega) = '\n' := by
  have h := flatten_uniform_getElem (displayRank value.storage) 9
    (displayRank_length value.storage) 8 rank 8 hr (by omega)
    (by change rank * 9 + 8 < (displayChars value).length; rw [displayChars_length]; omega)
  refine h.trans ?_
  simp [displayRank]

theorem display_injective (a b : Value) (h : display a = display b) : a = b := by
  have hchars : displayChars a = displayChars b := by
    have := congrArg String.toList h
    simpa [display] using this
  cases a with | mk sa =>
  cases b with | mk sb =>
  congr 1
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  let rank := i / 8
  let file := i % 8
  have hr : rank < 8 := by omega
  have hf : file < 8 := by omega
  have hidx : rank * 8 + file = i := by omega
  have ha := displayChars_getElem ⟨sa⟩ rank file hr hf
  have hb := displayChars_getElem ⟨sb⟩ rank file hr hf
  have heq : (displayChars ⟨sa⟩)[rank * 9 + file]'(by rw [displayChars_length]; omega) =
      (displayChars ⟨sb⟩)[rank * 9 + file]'(by rw [displayChars_length]; omega) := by
    simp [hchars]
  rw [ha, hb, hidx] at heq
  cases hsa : sa.getLsbD i <;> cases hsb : sb.getLsbD i <;> simp [hsa, hsb] at heq ⊢

def DisplayContract : Prop :=
  (∀ value : Value,
    (displayChars value).length = 72 ∧
    (∀ rank file, (hr : rank < 8) → (hf : file < 8) →
      (displayChars value)[rank * 9 + file]'(by rw [displayChars_length]; omega) =
        if value.storage.getLsbD (rank * 8 + file) then '1' else '.') ∧
    (∀ rank, (hr : rank < 8) →
      (displayChars value)[rank * 9 + 8]'(by rw [displayChars_length]; omega) = '\n')) ∧
  (∀ a b : Value, display a = display b → a = b)

theorem displayContract_proven : DisplayContract :=
  ⟨fun value => ⟨displayChars_length value,
    fun rank file hr hf => displayChars_getElem value rank file hr hf,
    fun rank hr => displayChars_newline value rank hr⟩,
   display_injective⟩

end NeuralReversi.Bitboard

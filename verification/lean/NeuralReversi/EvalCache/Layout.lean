import NeuralReversi.ScaledScore.ArithmeticProperties

namespace NeuralReversi.EvalCache

abbrev Key := BitVec 64
abbrev Word := BitVec 64
abbrev Way := Fin 4

def bucketBits (sizeLog2 : Nat) : Nat := sizeLog2 - 2

def bucketCount (sizeLog2 : Nat) : Nat := 2 ^ bucketBits sizeLog2

def checkedCount (pointerBits sizeLog2 : Nat) : Option Nat :=
  if sizeLog2 < pointerBits then some (bucketCount sizeLog2) else none

def lowMask (n : Nat) : Word := (BitVec.allOnes n).setWidth 64

theorem lowMask_value {n : Nat} (width : n ≤ 64) : (lowMask n).toNat = 2 ^ n - 1 := by
  rw [lowMask, BitVec.toNat_setWidth, BitVec.toNat_allOnes]
  have limit := Nat.pow_le_pow_right (by decide : 0 < 2) width
  have pos : 0 < 2 ^ n := Nat.two_pow_pos n
  exact Nat.mod_eq_of_lt (by omega)

theorem mask_width (word : Word) (n : Nat) :
    word &&& lowMask n = (word.setWidth n).setWidth 64 := by
  apply BitVec.eq_of_getLsbD_eq
  intro i permitted
  by_cases present : i < n
  · simp [lowMask, permitted, present]
  · simp [lowMask, present]

def bucket (sizeLog2 : Nat) (key : Key) : Nat :=
  (key.rotateLeft 16 &&& lowMask (bucketBits sizeLog2)).toNat

def victim (key : Key) : Nat := (key.rotateLeft 16 >>> 62).toNat

def tag (key : Key) : Word := key &&& lowMask 48

def GeometryContract : Prop :=
  ∀ (pointerBits sizeLog2 : Nat) (key : Key), (pointerBits = 32 ∨ pointerBits = 64) →
    (sizeLog2 ≥ pointerBits → checkedCount pointerBits sizeLog2 = none) ∧
    (sizeLog2 < pointerBits →
      checkedCount pointerBits sizeLog2 = some (bucketCount sizeLog2) ∧
      bucket sizeLog2 key < bucketCount sizeLog2 ∧
      bucketCount sizeLog2 < 2 ^ pointerBits ∧ victim key < 4 ∧
      (lowMask (bucketBits sizeLog2)).toNat = bucketCount sizeLog2 - 1)

theorem victim_lt (key : Key) : victim key < 4 := by
  exact BitVec.toNat_ushiftRight_lt (key.rotateLeft 16) 62 (by decide)

theorem bucket_lt (sizeLog2 : Nat) (key : Key) : bucket sizeLog2 key < bucketCount sizeLog2 := by
  unfold bucket bucketCount
  rw [mask_width, BitVec.toNat_setWidth]
  exact Nat.lt_of_le_of_lt (Nat.mod_le ..) (key.rotateLeft 16 |>.setWidth (bucketBits sizeLog2)).isLt

theorem geometryContract_proven : GeometryContract := by
  intro pointerBits sizeLog2 key widths
  refine ⟨fun invalid => by simp [checkedCount, show ¬sizeLog2 < pointerBits by omega], ?_⟩
  intro admitted
  have count : bucketCount sizeLog2 < 2 ^ pointerBits := by
    unfold bucketCount bucketBits
    exact Nat.pow_lt_pow_right (by decide) (by omega)
  exact ⟨by simp [checkedCount, admitted], bucket_lt _ _, count, victim_lt _,
    lowMask_value (by unfold bucketBits; omega)⟩

def KeyContract : Prop :=
  ∀ (sizeLog2 : Nat) (left right : Key), 18 ≤ sizeLog2 →
    (bucket sizeLog2 left = bucket sizeLog2 right ∧ tag left = tag right ↔ left = right)

theorem keyContract_proven : KeyContract := by
  intro sizeLog2 left right enough
  constructor
  · rintro ⟨sameBucket, sameTag⟩
    apply BitVec.eq_of_getLsbD_eq
    intro i hi
    by_cases low : i < 48
    · have bits := congrArg (fun v : Word => v.getLsbD i) sameTag
      simpa [tag, mask_width, hi, low] using bits
    · have sameWord : left.rotateLeft 16 &&& lowMask (bucketBits sizeLog2) =
          right.rotateLeft 16 &&& lowMask (bucketBits sizeLog2) := BitVec.eq_of_toNat_eq sameBucket
      have small : i - 48 < 16 := by omega
      have inBucket : i - 48 < bucketBits sizeLog2 := by unfold bucketBits; omega
      have bits := congrArg (fun v : Word => v.getLsbD (i - 48)) sameWord
      have rot : 48 + (i - 48) = i := by omega
      simpa [mask_width, small, inBucket, hi, show i - 48 < 64 by omega,
        BitVec.getLsbD_rotateLeft_of_le (by decide : 16 < 64), rot] using bits
  · rintro rfl
    exact ⟨rfl, rfl⟩

def emptyWord : Word := 0x8000

def pack (key : Key) (score : BitVec 32) : Word :=
  (tag key <<< 16) ||| (score.setWidth 16).setWidth 64

def isHit (key : Key) (word : Word) : Bool :=
  decide (word >>> 16 = tag key ∧ word.setWidth 16 ≠ (0x8000 : BitVec 16))

def read (key : Key) (word : Word) : Option Int :=
  if isHit key word then some (word.setWidth 16).toInt else none

def Admitted (score : Int) : Prop := -32767 ≤ score ∧ score ≤ 32767

instance (score : Int) : Decidable (Admitted score) := inferInstanceAs (Decidable (_ ∧ _))

def encode (key : Key) (score : BitVec 32) : Option Word :=
  if Admitted score.toInt then some (pack key score) else none

theorem pack_score (key : Key) (score : BitVec 32) :
    (pack key score).setWidth 16 = score.setWidth 16 := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp [pack, hi]

theorem pack_tag (key : Key) (score : BitVec 32) : pack key score >>> 16 = tag key := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  by_cases h : i < 48
  · simp [pack, hi, show 16 + i < 64 by omega]
  · have upper := BitVec.getLsbD_of_ge (x := pack key score) (i := 16 + i) (by omega)
    have right : (tag key).getLsbD i = false := by simp [tag, mask_width, h]
    simpa [BitVec.getLsbD_ushiftRight, right] using upper

theorem narrow_value {score : BitVec 32} (admitted : Admitted score.toInt) :
    (score.setWidth 16).toInt = score.toInt := by
  obtain ⟨lo, hi⟩ := admitted
  calc
    _ = score.toInt.bmod (2 ^ 16) := by
      rw [BitVec.toInt_setWidth, BitVec.toInt_eq_toNat_bmod,
        Int.bmod_bmod_of_dvd (by decide : 2^16 ∣ 2^32)]
    _ = _ := Int.bmod_eq_of_le (by omega) (by omega)

theorem matches_pack {key original : Key} {score : BitVec 32} (valid : Admitted score.toInt) :
    isHit key (pack original score) = true ↔ tag original = tag key := by
  have usable : score.setWidth 16 ≠ (0x8000 : BitVec 16) := by
    intro bad
    have h : (score.setWidth 16).toInt = -32768 := congrArg BitVec.toInt bad
    have lo := valid.1
    rw [narrow_value valid] at h
    omega
  simp only [isHit, decide_eq_true_eq, pack_tag, pack_score, and_iff_left usable]

theorem read_empty (key : Key) : read key emptyWord = none := by
  simp [read, isHit, emptyWord]

def EncodingContract : Prop :=
  ∀ (key : Key) (score : BitVec 32),
    (¬ Admitted score.toInt → encode key score = none) ∧
    (Admitted score.toInt → encode key score = some (pack key score) ∧
      (∀ lookup, read lookup (pack key score) =
        if tag lookup = tag key then some score.toInt else none)) ∧ read key emptyWord = none

theorem encodingContract_proven : EncodingContract := by
  intro key score
  refine ⟨fun bad => by simp [encode, bad], fun ok => ?_, read_empty key⟩
  refine ⟨by simp [encode, ok], fun lookup => ?_⟩
  have branch := (matches_pack (key := lookup) (original := key) ok).trans eq_comm
  simp only [read, pack_score, narrow_value ok, branch]

def DecodeContract : Prop :=
  ∀ key word score, read key word = some score → Admitted score

theorem decodeContract_proven : DecodeContract := by
  intro key word score found
  unfold read at found
  split at found
  · rename_i usable
    have raw := Option.some.inj found
    have low := BitVec.le_toInt (word.setWidth 16)
    have high := BitVec.toInt_lt (x := word.setWidth 16)
    have notEmpty : (word.setWidth 16).toInt ≠ -32768 := by
      intro empty
      have guard := (of_decide_eq_true usable).2
      apply guard
      exact BitVec.eq_of_toInt_eq empty
    unfold Admitted
    omega
  · cases found

example : bucket 4 0 = bucket 4 0x0004000000000000 ∧ tag 0 = tag 0x0004000000000000 := by decide
example : read 0 emptyWord = none := by decide
example : encode 0 0x8000 = none := by decide

end NeuralReversi.EvalCache
